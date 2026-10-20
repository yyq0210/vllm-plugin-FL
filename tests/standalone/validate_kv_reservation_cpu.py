"""Use the installed vLLM allocator on CPU; never allocate a CUDA tensor."""
import hashlib
import json
import pickle
import random
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

# Only the platform's import-time capability query is stubbed in this test.
# _lazy_init remains forbidden for the entire test process.
torch.cuda._lazy_init = lambda: (_ for _ in ()).throw(
    AssertionError("CPU validation must not initialize CUDA"))
with patch("torch.cuda.get_device_capability", return_value=(7, 0)):
    from vllm import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import (
        get_request_block_hasher, init_none_hash)
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec)
    from vllm.v1.request import Request, RequestStatus
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm_fl.worker.kv_reservation_config import (
        configure_kv_reservation, SCHEDULER_PATH)
    from vllm_fl.worker.kv_reservation_scheduler import (
        KVAdmissionController, KVReservationAsyncScheduler, supported_cache)

hash_fn = lambda x: hashlib.sha256(pickle.dumps(x)).digest()
init_none_hash(hash_fn)
hasher = get_request_block_hasher(16, hash_fn)


def request(name, prompt=64, output=64, salt=0):
    return Request(name, [salt] * prompt,
                   SamplingParams(max_tokens=output), None, block_hasher=hasher)


def setup(blocks=20, caching=False, watermark=0.0):
    spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=128,
                             dtype=torch.bfloat16)
    config = KVCacheConfig(blocks, [], [KVCacheGroupSpec(["layer"], spec)])
    manager = KVCacheManager(
        config, max_model_len=1024, scheduler_block_size=16,
        hash_block_size=16, max_num_batched_tokens=64,
        enable_caching=caching, watermark=watermark)
    scheduler = NS(running=[])
    control = KVAdmissionController(manager, scheduler)
    return manager, scheduler, control, config


def admit(control, req, count=None, prefix=None, hits=0):
    return control.allocate_slots(
        req, count or (req.num_tokens - hits),
        num_new_computed_tokens=hits, new_computed_blocks=prefix,
        full_sequence_must_fit=True,
        has_scheduled_reqs=bool(control.scheduler.running))


def mark_running(scheduler, req, computed):
    req.status = RequestStatus.RUNNING
    req.num_computed_tokens = computed
    scheduler.running.append(req)


def snapshot(manager):
    return {
        "free": manager.block_pool.get_num_free_blocks(),
        "blocks": [(b.block_id, b.ref_cnt, b.block_hash,
                    getattr(b.prev_free_block, "block_id", None),
                    getattr(b.next_free_block, "block_id", None))
                   for b in manager.block_pool.blocks],
        "requests": [dict((k, [b.block_id for b in v]) for k, v in m.req_to_blocks.items())
                     for m in manager.coordinator.single_type_managers],
    }


def test_growth_rejection_is_read_only():
    m, s, c, _ = setup(13)  # 12 usable blocks, one null block
    a, b = request("a"), request("b", salt=1)
    assert admit(c, a) is not None
    mark_running(s, a, 64)
    before = snapshot(m)
    assert admit(c, b) is None  # a needs 4 more + b needs 8, only 8 free
    assert snapshot(m) == before
    assert b.status == RequestStatus.WAITING and b.num_computed_tokens == 0
    # Demonstrate the base gate admits the same request and overcommits future
    # decode capacity. This is a counterexample, not an alternative oracle.
    assert m.allocate_slots(b, 64, full_sequence_must_fit=True) is not None


def test_chunked_prefill_reserves_unallocated_input():
    m, s, c, _ = setup(21)  # 20 usable
    a, b = request("a", 256, 16), request("b", 64, 16, salt=1)
    assert admit(c, a, count=16) is not None
    mark_running(s, a, 16)
    before = snapshot(m)
    assert admit(c, b, count=16) is None  # a's remaining 16 + b's 5 > 19
    assert snapshot(m) == before


def test_release_and_resume():
    m, s, c, _ = setup(13)
    a, b = request("a"), request("b", salt=1)
    assert admit(c, a) is not None
    mark_running(s, a, 64)
    b.status = RequestStatus.PREEMPTED
    assert admit(c, b) is None
    m.free(a)
    s.running.remove(a)
    a.status = RequestStatus.FINISHED_ABORTED
    assert admit(c, b) is not None
    assert c.stats["first_request"] == 2


def test_running_allocation_unchanged():
    m, s, c, _ = setup(13)
    a = request("a")
    assert admit(c, a) is not None
    mark_running(s, a, 64)
    a.append_output_token_ids([2])
    assert c.allocate_slots(a, 1) is not None
    assert len(m.coordinator.get_blocks("a")[0]) == 5
    assert c.stats["checked"] == 0


def test_exact_boundary_and_async_placeholders():
    m, s, c, _ = setup(17)  # 16 blocks fits two complete lifetimes of 8 each
    a, b = request("a"), request("b", salt=1)
    assert admit(c, a) is not None
    mark_running(s, a, 64)
    a.num_output_placeholders = 2
    assert admit(c, b) is not None
    mark_running(s, b, 64)
    # Reserve a conservative final-token slot too, independently of placeholders.
    for r in (a, b):
        assert c.allocate_slots(r, 64) is not None
        r.num_computed_tokens = 128
    assert m.block_pool.get_num_free_blocks() == 0
    assert c.stats["deferred"] == 0


def test_shared_prefix_counts_physical_blocks():
    m, s, c, _ = setup(10, caching=True)  # 9 usable
    a, b = request("a", 65, 15), request("b", 65, 15)
    assert admit(c, a) is not None
    mark_running(s, a, 65)
    prefix, hits = m.get_computed_blocks(b)
    assert hits == 64
    assert c._remaining_blocks(b, prefix, hits) == 1  # four shared + one private
    assert admit(c, b, prefix=prefix, hits=hits) is not None
    assert m.block_pool.get_num_free_blocks() == 3
    assert all(x.ref_cnt == 2 for x in prefix.blocks[0])


def test_free_cached_prefix_costs_capacity():
    m, s, c, _ = setup(13, caching=True)
    cached = request("cached", 65, 15)
    assert admit(c, cached) is not None
    m.free(cached)  # hashed prefix blocks are on the free queue, ref_cnt=0
    a, b = request("a", 64, 64, salt=1), request("b", 65, 15)
    assert admit(c, a) is not None
    mark_running(s, a, 64)
    prefix, hits = m.get_computed_blocks(b)
    assert hits == 64
    assert c._remaining_blocks(b, prefix, hits) == 5
    before = snapshot(m)
    assert admit(c, b, prefix=prefix, hits=hits) is None  # 4+5 > 8
    assert snapshot(m) == before
    assert all(x.ref_cnt == 0 for x in prefix.blocks[0])


def test_watermark_and_first_request_progress():
    m, s, c, _ = setup(17, watermark=0.1)
    a, b = request("a"), request("b", salt=1)
    assert admit(c, a) is not None
    mark_running(s, a, 64)
    assert admit(c, b) is None  # 16 lifetime blocks + existing watermark >16
    assert m.watermark_blocks == 1
    m2, _, c2, _ = setup(5)
    # Lifetime cannot fit in this artificial pool, but the sole request keeps
    # the base allocator's progress semantics.
    assert admit(c2, request("first", 32, 200)) is not None


def test_unsupported_request_falls_back():
    for field, value in [("resumable", True), ("spec_token_ids", [9]),
                         ("kv_transfer_params", {"x": 1})]:
        m, s, c, _ = setup(13)
        a, b = request("a"), request("b", salt=1)
        assert admit(c, a) is not None
        mark_running(s, a, 64)
        setattr(b, field, value)
        assert admit(c, b) is not None
        assert c.stats["unsupported"] == 1


def test_supported_cache_guard():
    _, _, _, cfg = setup()
    assert supported_cache(cfg)
    spec = cfg.kv_cache_groups[0].kv_cache_spec
    for changed in [replace(spec, sliding_window=64),
                    replace(spec, attention_chunk_size=64),
                    replace(spec, non_causal=True)]:
        assert not supported_cache(KVCacheConfig(
            20, [], [KVCacheGroupSpec(["layer"], changed)]))
    assert not supported_cache(KVCacheConfig(
        20, [], cfg.kv_cache_groups * 2))


def config():
    return NS(
        scheduler_config=NS(scheduler_cls=None, async_scheduling=True,
                            scheduler_reserve_full_isl=True),
        model_config=NS(is_encoder_decoder=False, is_multimodal_model=False,
                        runner_type="generate"),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
                           data_parallel_size=1, decode_context_parallel_size=1,
                           prefill_context_parallel_size=1),
        speculative_config=None, kv_transfer_config=None, ec_transfer_config=None)


def test_config_and_async_inheritance():
    v = config()
    assert configure_kv_reservation(v, "iluvatar")
    assert v.scheduler_config.scheduler_cls == SCHEDULER_PATH
    assert issubclass(KVReservationAsyncScheduler, AsyncScheduler)
    assert not configure_kv_reservation(v, "iluvatar")  # preserve explicit class
    for group, name, value in [
        ("scheduler_config", "async_scheduling", False),
        ("scheduler_config", "async_scheduling", None),
        ("scheduler_config", "scheduler_cls", "custom.Scheduler"),
        ("parallel_config", "tensor_parallel_size", 2),
        ("model_config", "is_multimodal_model", True),
    ]:
        v = config()
        setattr(getattr(v, group), name, value)
        assert not configure_kv_reservation(v, "iluvatar")
    assert not configure_kv_reservation(config(), "nvidia")


def test_manager_restored_on_schedule_failure():
    obj = KVReservationAsyncScheduler.__new__(KVReservationAsyncScheduler)
    m, _, c, _ = setup()
    obj.kv_cache_manager = m
    obj._kv_admission = c
    with patch.object(AsyncScheduler, "schedule", side_effect=RuntimeError("test")):
        try:
            obj.schedule()
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected failure")
    assert obj.kv_cache_manager is m


def test_varied_request_lifetimes_never_preempt_running():
    """Drive the real allocator through chunked prefill, decode and cancellation."""
    rng = random.Random(20261020)
    for caching in (False, True):
        m, s, c, cfg = setup(81, caching=caching)
        pending = [
            request(str(i), rng.randint(32, 256), rng.randint(1, 129), salt=i % 8)
            for i in range(96)
        ]
        finished = set()
        for step in range(20000):
            for req in list(s.running):
                if int(req.request_id) % 11 == 0 and step % 17 == 0:
                    m.free(req)
                    s.running.remove(req)
                    req.status = RequestStatus.FINISHED_ABORTED
                    finished.add(req.request_id)
                    continue
                n = min(16, req.num_tokens - req.num_computed_tokens)
                assert n > 0
                assert c.allocate_slots(req, n) is not None, (
                    "running request exhausted reserved capacity", req.request_id)
                req.num_computed_tokens += n
                if req.num_computed_tokens == req.num_tokens:
                    req.append_output_token_ids([7])
                    if req.num_output_tokens == req.max_tokens:
                        m.free(req)
                        s.running.remove(req)
                        req.status = RequestStatus.FINISHED_LENGTH_CAPPED
                        finished.add(req.request_id)
            while pending:
                req = pending[0]
                prefix, hits = m.get_computed_blocks(req)
                n = min(16, req.num_tokens - hits)
                if admit(c, req, count=n, prefix=prefix, hits=hits) is None:
                    break
                pending.pop(0)
                mark_running(s, req, hits + n)
                if req.num_computed_tokens == req.num_tokens:
                    req.append_output_token_ids([7])
                    if req.max_tokens == 1:
                        m.free(req)
                        s.running.remove(req)
                        req.status = RequestStatus.FINISHED_LENGTH_CAPPED
                        finished.add(req.request_id)
            if not pending and not s.running:
                break
        else:
            raise AssertionError("lifecycle simulation made no progress")
        assert len(finished) == 96
        assert m.block_pool.get_num_free_blocks() == cfg.num_blocks - 1
        assert c.stats["deferred"] > 0


def main():
    result = {"status": "running", "checks": [], "cuda_initialized": False}
    for name, fn in sorted(globals().copy().items()):
        if name.startswith("test_") and callable(fn):
            start = time.perf_counter()
            try:
                fn()
                item = {"name": name, "passed": True}
            except Exception as exc:
                item = {"name": name, "passed": False, "error": repr(exc),
                        "traceback": traceback.format_exc()}
            item["seconds"] = time.perf_counter() - start
            result["checks"].append(item)
    result["cuda_initialized"] = torch.cuda.is_initialized()
    result["status"] = "passed" if (
        all(x["passed"] for x in result["checks"])
        and not result["cuda_initialized"]) else "failed"
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
