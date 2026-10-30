# SPDX-License-Identifier: Apache-2.0
"""Opt-in BI-V150 top-p row concurrency; keep the original vLLM JIT kernel.

Launcher and scratch management adapted from the installed vLLM
v1/sample/ops/topk_topp_triton.py (Apache-2.0). No filtering or RNG changes.
"""
import functools
import hashlib
import os
from pathlib import Path

import torch
from vllm.logger import init_logger

ENV_NAME = 'VLLM_FL_TOPP_ROW_PARALLEL'
SHADOW_ENV = 'VLLM_FL_TOPP_ROW_PARALLEL_SHADOW'
SHADOW_ARM_ENV = 'VLLM_FL_TOPP_ROW_PARALLEL_SHADOW_ARM_FILE'
EXPECTED_SOURCE_SHA256 = '64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8'
logger = init_logger('vllm.leanscale.topp_row_parallel')
_stats = {'candidate_calls': 0, 'shadow_checks': 0}
_logged_batches = set()


class _ShadowBudget:
    """Per-batch diagnostic quota, armed only after service warmup."""
    def __init__(self, arm_file):
        if not arm_file or not Path(arm_file).is_absolute():
            raise ValueError('Shadow diagnostics require an absolute arm file path')
        self.arm_file = Path(arm_file)
        if self.arm_file.exists():
            raise ValueError('Shadow arm file must not exist before service startup')
        self.armed = False
        self.passed = {}

    def should_check(self, batch):
        if not 17 <= batch <= 64:
            return False
        if not self.armed:
            self.armed = self.arm_file.is_file()
        return self.armed and self.passed.get(batch, 0) < 2

    def record_pass(self, batch):
        if not self.armed or not 17 <= batch <= 64 or self.passed.get(batch, 0) >= 2:
            raise RuntimeError('Invalid shadow budget update')
        self.passed[batch] = self.passed.get(batch, 0) + 1


def register_topp_row_parallel_env():
    from vllm import envs
    for key in (ENV_NAME, SHADOW_ENV):
        envs.environment_variables.setdefault(key, lambda name=key: os.environ.get(name, '0') == '1')
    envs.environment_variables.setdefault(SHADOW_ARM_ENV, lambda: os.environ.get(SHADOW_ARM_ENV, ''))


@functools.lru_cache(maxsize=8)
def _supported_device(index):
    return torch.cuda.device_count() == 1 and torch.cuda.get_device_name(index) == 'Iluvatar BI-V150'


def _eligible(logits, k, p, mask_value):
    return (
        logits.device.type == 'cuda' and logits.dtype == torch.float32
        and logits.ndim == 2 and 17 <= logits.shape[0] <= 64
        and logits.shape[1] == 130560 and logits.stride(1) == 1
        and logits.stride(0) >= logits.shape[1]
        and k is None and p is not None and p.device == logits.device
        and p.dtype == torch.float32 and p.ndim == 1
        and p.shape[0] == logits.shape[0] and p.stride(0) == 1
        and mask_value == -float('inf')
        and _supported_device(logits.device.index)
    )


def install_topp_row_parallel():
    """Install both references once; default off and fail closed on source drift."""
    register_topp_row_parallel_env()
    if os.environ.get(ENV_NAME, '0') != '1':
        return False
    from vllm.platforms import current_platform
    if getattr(current_platform, 'vendor_name', None) != 'iluvatar':
        return False
    import vllm.v1.sample.ops.topk_topp_triton as base
    import vllm.v1.sample.ops.topk_topp_sampler as sampler
    original = base.apply_top_k_top_p_triton
    if getattr(original, '_leanscale_topp_row_parallel', False):
        return sampler.apply_top_k_top_p_triton is original
    if (hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() != EXPECTED_SOURCE_SHA256
            or sampler.apply_top_k_top_p_triton is not original):
        logger.warning('LeanScale top-p row parallel disabled: unverified implementation or alias')
        return False
    shadow = os.environ.get(SHADOW_ENV, '0') == '1'
    shadow_budget = _ShadowBudget(os.environ.get(SHADOW_ARM_ENV)) if shadow else None

    @functools.wraps(original)
    def apply(logits, k, p, mask_value=-float('inf')):
        if not _eligible(logits, k, p, mask_value):
            return original(logits, k, p, mask_value)
        num_sm = base.num_compute_units(logits.device.index)
        if num_sm != 16:
            return original(logits, k, p, mask_value)
        batch, vocab = logits.shape
        programs = min(num_sm * 4, batch)
        key = (logits.device, logits.dtype, vocab)
        buffer = base._TRITON_BUFFER_CACHE.get(key)
        tables = base._TRITON_TABLE_CACHE.get(logits.device)
        capturing = torch.cuda.is_current_stream_capturing()
        # Warm up outside capture before changing shared storage or host tables.
        if capturing and (buffer is None or buffer.shape[0] < 64 or tables is None or shadow):
            return original(logits, k, p, mask_value)
        if buffer is None or buffer.shape[0] < 64:
            buffer = logits.new_empty((64, vocab))
            base._TRITON_BUFFER_CACHE[key] = buffer
        if tables is None:
            tables = (logits.new_tensor(base._NORMAL_CDF_TO_SIGMA_TABLE),
                      logits.new_tensor(base._PERCENTILE_TO_STD_TABLE))
            base._TRITON_TABLE_CACHE[logits.device] = tables
        normal, percentile = tables
        do_shadow = shadow_budget is not None and shadow_budget.should_check(batch)
        reference = logits.clone() if do_shadow else None
        rng_before = torch.cuda.get_rng_state(logits.device) if do_shadow else None
        base._topk_topp_kernel[(programs,)](
            logits, logits.stride(0), buffer, percentile, normal, logits, p,
            BATCH_SIZE=batch, MASK_VALUE=mask_value, VOCAB_SIZE=vocab,
            BLOCK_SIZE=8192, BLOCK_SIZE_TRUNC=4096,
            TOPK_ENABLED=False, TOPP_ENABLED=True,
        )
        _stats['candidate_calls'] += 1
        if do_shadow:
            original(reference, None, p, mask_value)
            if not torch.equal(logits, reference):
                raise RuntimeError('Top-p row concurrency changed masked logits')
            if not torch.equal(rng_before, torch.cuda.get_rng_state(logits.device)):
                raise RuntimeError('Top-p masking changed RNG state')
            shadow_budget.record_pass(batch)
            _stats['shadow_checks'] += 1
            values = p.detach().cpu().tolist()
            logger.info('LeanScale top-p shadow passed: B=%d V=%d stride=%s p_min=%s p_max=%s exact_mask=True rng_unchanged=True check=%d',
                        batch, vocab, tuple(logits.stride()), min(values), max(values), _stats['shadow_checks'])
        if not capturing and (batch not in _logged_batches or _stats['candidate_calls'] % 1024 == 0):
            _logged_batches.add(batch)
            logger.info('LeanScale top-p grid active: B=%d V=%d programs=%d scratch_rows=%d scratch_bytes=%d k=None calls=%d shadow=%s',
                        batch, vocab, programs, buffer.shape[0], buffer.numel()*buffer.element_size(),
                        _stats['candidate_calls'], shadow)
        return logits

    apply._leanscale_topp_row_parallel = True
    apply._leanscale_original = original
    base.apply_top_k_top_p_triton = apply
    sampler.apply_top_k_top_p_triton = apply
    logger.info('LeanScale top-p grid hook installed: max_programs=64 B=17..64 V=130560 top_p_only=True shadow=%s', shadow)
    return True
