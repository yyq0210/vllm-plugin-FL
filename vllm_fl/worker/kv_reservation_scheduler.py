# Copyright (c) 2026 BAAI. All rights reserved.
"""Conservative, lossless KV growth admission for a single full-attention pool.

This adds a read-only capacity gate before the original vLLM allocator. It
reserves capacity logically, never allocates future blocks or modifies KV.
"""

import json
import time

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, UniformTypeKVCacheSpecs
from vllm.v1.request import RequestStatus

logger = init_logger("vllm.leanscale.kv_reservation")


def _supported_request(request):
    return (
        request.sampling_params is not None
        and request.pooling_params is None
        and not request.mm_features
        and not request.resumable
        and request.streaming_queue is None
        and request.kv_transfer_params is None
        and request.max_tokens > 0
        and not request.spec_token_ids
    )


def supported_cache(kv_cache_config):
    groups = kv_cache_config.kv_cache_groups
    if len(groups) != 1 or groups[0].is_eagle_group:
        return False
    spec = groups[0].kv_cache_spec
    specs = (
        list(spec.kv_cache_specs.values())
        if isinstance(spec, UniformTypeKVCacheSpecs) else [spec]
    )
    return bool(specs) and all(
        type(s) is FullAttentionSpec
        and s.sliding_window is None
        and s.attention_chunk_size is None
        and not s.non_causal
        for s in specs
    )


class KVAdmissionController:
    """Instance-local allocator facade; all mutations stay in the base manager."""

    def __init__(self, manager, scheduler):
        self.manager = manager
        self.scheduler = scheduler
        self.stats = dict(checked=0, deferred=0, admitted=0, first_request=0,
                          unsupported=0, peak_running=0, peak_required_blocks=0)

    def __getattr__(self, name):
        return getattr(self.manager, name)

    def _remaining_blocks(self, request, new_blocks=None, new_computed_tokens=0):
        manager = self.manager
        # Upper bound includes the final sampled token conservatively. Async
        # placeholders do not extend this bound: the base scheduler stops at
        # prompt length + max_tokens, including outstanding outputs.
        target = min(request.num_prompt_tokens + request.max_tokens,
                     manager.max_model_len)
        blocks = (manager.empty_kv_cache_blocks.blocks if new_blocks is None
                  else new_blocks.blocks)
        return manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=target,
            new_computed_blocks=blocks,
            num_encoder_tokens=0,
            total_computed_tokens=min(
                request.num_computed_tokens + new_computed_tokens,
                manager.max_model_len),
            num_tokens_main_model=target,
            apply_admission_cap=True,
        )

    def allocate_slots(
        self, request, num_new_tokens, num_new_computed_tokens=0,
        new_computed_blocks=None, num_lookahead_tokens=0,
        num_external_computed_tokens=0, delay_cache_blocks=False,
        num_encoder_tokens=0, full_sequence_must_fit=False,
        reserved_blocks=0, has_scheduled_reqs=True,
    ):
        is_admission = request.status in (
            RequestStatus.WAITING, RequestStatus.PREEMPTED)
        if is_admission:
            running = self.scheduler.running
            self.stats["peak_running"] = max(
                self.stats["peak_running"], len(running))
            supported = (
                full_sequence_must_fit
                and not num_lookahead_tokens
                and not num_external_computed_tokens
                and not delay_cache_blocks
                and not num_encoder_tokens
                # Nonzero reservations belong to other, unsupported policies.
                and not reserved_blocks
                and _supported_request(request)
                and all(_supported_request(r) for r in running)
            )
            if not supported:
                self.stats["unsupported"] += 1
            elif not running:
                # Preserve base progress when the pool is otherwise idle.
                self.stats["first_request"] += 1
            else:
                self.stats["checked"] += 1
                required = sum(self._remaining_blocks(r) for r in running)
                required += self._remaining_blocks(
                    request, new_computed_blocks, num_new_computed_tokens)
                if has_scheduled_reqs:
                    required += self.manager.watermark_blocks
                self.stats["peak_required_blocks"] = max(
                    self.stats["peak_required_blocks"], required)
                if required > self.manager.block_pool.get_num_free_blocks():
                    # No touching prefix blocks, request state, or free lists.
                    self.stats["deferred"] += 1
                    return None
        result = self.manager.allocate_slots(
            request, num_new_tokens,
            num_new_computed_tokens=num_new_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            num_lookahead_tokens=num_lookahead_tokens,
            num_external_computed_tokens=num_external_computed_tokens,
            delay_cache_blocks=delay_cache_blocks,
            num_encoder_tokens=num_encoder_tokens,
            full_sequence_must_fit=full_sequence_must_fit,
            reserved_blocks=reserved_blocks,
            has_scheduled_reqs=has_scheduled_reqs,
        )
        if is_admission and result is not None:
            self.stats["admitted"] += 1
        return result


class KVReservationAsyncScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        parallel = self.parallel_config
        enabled = (
            supported_cache(self.kv_cache_config)
            and self.scheduler_config.async_scheduling is True
            and self.scheduler_reserve_full_isl
            and not self.is_encoder_decoder
            and self.connector is None
            and self.ec_connector is None
            and self.vllm_config.speculative_config is None
            and not self.max_num_encoder_input_tokens
            and all(
                getattr(parallel, name, None) == 1
                for name in (
                    "tensor_parallel_size", "pipeline_parallel_size",
                    "data_parallel_size", "decode_context_parallel_size",
                    "prefill_context_parallel_size",
                )
            )
        )
        self._kv_admission = (
            KVAdmissionController(self.kv_cache_manager, self) if enabled else None
        )
        self._kv_admission_last_log = 0.0
        logger.info("LeanScale KV growth admission enabled=%s policy=full_max_tokens",
                    enabled)

    def schedule(self):
        controller = self._kv_admission
        if controller is None:
            return super().schedule()
        # Keep the public manager unchanged outside this synchronous scheduling
        # call. The original AsyncScheduler still owns all token accounting.
        original = self.kv_cache_manager
        self.kv_cache_manager = controller
        try:
            output = super().schedule()
        finally:
            self.kv_cache_manager = original
        controller.stats["peak_running"] = max(
            controller.stats["peak_running"], len(self.running))
        now = time.monotonic()
        if now - self._kv_admission_last_log >= 60:
            self._kv_admission_last_log = now
            logger.info("LeanScale KV admission stats %s", json.dumps({
                **controller.stats, "running": len(self.running),
                "free_blocks": original.block_pool.get_num_free_blocks(),
            }, sort_keys=True))
        return output
