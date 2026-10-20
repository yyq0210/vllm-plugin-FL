# SPDX-License-Identifier: Apache-2.0
"""Functional Q/K results with an ordered, single KV-cache side effect."""
import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.v1.attention.backend import AttentionType
from vllm.utils.torch_utils import (
    LayerNameType, _resolve_layer_name, direct_register_custom_op,
)
from vllm_fl.dispatch import CachedOp, get_default_manager, is_dump_enabled

logger = init_logger("vllm.leanscale.rope_kv_dispatch")
_rope = CachedOp("rotary_embedding")


def rope_kv_dispatch(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                     cos: torch.Tensor, sin: torch.Tensor, positions: torch.Tensor,
                     interleaved: bool, layer_name: LayerNameType
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    name = _resolve_layer_name(layer_name)
    _, layer, cache, slots = get_attention_context(name)
    from vllm_fl.dispatch.backends.flaggems.impl.iluvatar_gqa import IluvatarGQAImpl

    manager = get_default_manager()
    selected = manager._resolve_impl("rotary_embedding")
    supported = (
        selected.impl_id == "default.flagos" and not is_dump_enabled()
        and isinstance(layer.impl, IluvatarGQAImpl)
        and layer.impl.attn_type == AttentionType.DECODER
        and layer.impl.kv_cache_dtype in ("auto", "float16", "bfloat16")
        and not layer.impl._is_per_token_head_quant
        and slots is not None and slots.ndim == 1
        and slots.dtype in (torch.int32, torch.int64)
        and query.ndim == key.ndim == value.ndim == 3
        and query.shape[1:] == (16, 128)
        and key.shape[1:] == value.shape[1:] == (2, 128)
        and query.shape[0] == key.shape[0] == value.shape[0] == positions.numel()
        and query.dtype in (torch.float16, torch.bfloat16)
        and query.dtype == key.dtype == value.dtype == cache.dtype
        and query.device.type == "cuda"
        and query.device == key.device == value.device == cache.device
        and query.device == cos.device == sin.device == positions.device == slots.device
        and query.stride(-1) == key.stride(-1) == value.stride(-1) == 1
        and cache.ndim == 5 and cache.shape[1] == 2
        and cache.shape[-2:] == (2, 128) and cache.stride(-1) == 1
        and cos.ndim == sin.ndim == 2 and cos.shape == sin.shape
        and cos.shape[-1] == 64 and cos.stride(-1) == sin.stride(-1) == 1
        and positions.ndim == 1 and positions.dtype in (torch.int32, torch.int64)
    )
    if supported:
        from flag_gems.runtime.backend._iluvatar.fused.rope_kv_cache import rope_kv_cache
        manager._record_first_use("rotary_embedding", selected)
        kc, vc = cache.unbind(1)
        qout, kout = rope_kv_cache(
            query, key, value, cos, sin, positions, kc, vc, slots,
            interleaved=interleaved,
        )
        if not getattr(layer.impl, "_leanscale_rope_kv_logged", False):
            logger.info("LeanScale fused RoPE+KV active: layer=%s", name)
            layer.impl._leanscale_rope_kv_logged = True
    else:
        qout, kout = _rope(None, query, key, cos, sin, positions, interleaved, False)
        if slots is not None:
            layer.impl.do_kv_cache_update(layer, kout, value, cache, slots)
    # As in unified_kv_cache_update, this dummy creates an explicit dependency
    # consumed by attention. Projection inputs remain functional and unaliased.
    return qout, kout, torch.empty(0, device=cache.device, dtype=cache.dtype)


def rope_kv_dispatch_fake(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                          cos: torch.Tensor, sin: torch.Tensor, positions: torch.Tensor,
                          interleaved: bool, layer_name: LayerNameType
                          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (torch.empty(query.shape, device=query.device, dtype=query.dtype),
            torch.empty(key.shape, device=key.device, dtype=key.dtype),
            torch.empty(0, device=key.device, dtype=key.dtype))


direct_register_custom_op(
    op_name="fl_rope_kv_dispatch", op_func=rope_kv_dispatch, mutates_args=[],
    fake_impl=rope_kv_dispatch_fake,
)
