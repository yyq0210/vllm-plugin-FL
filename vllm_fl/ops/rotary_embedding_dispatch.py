# Copyright (c) 2026 BAAI. All rights reserved.
"""Compiler boundary for functional RoPE through the existing FL dispatcher."""
import torch

from vllm.utils.torch_utils import direct_register_custom_op
from vllm_fl.dispatch import CachedOp

_rotary_embedding = CachedOp("rotary_embedding")


def rotary_embedding_dispatch(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rotary_interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    # This boundary is functional: backends must not mutate the QKV projection.
    # CachedOp retains policy changes, strict mode, fallback and IO-dump hooks.
    return _rotary_embedding(
        None, query, key, cos, sin, positions, rotary_interleaved, False
    )


def rotary_embedding_dispatch_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rotary_interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.empty(query.shape, device=query.device, dtype=query.dtype),
            torch.empty(key.shape, device=key.device, dtype=key.dtype))


direct_register_custom_op(
    op_name="fl_rotary_embedding_dispatch",
    op_func=rotary_embedding_dispatch,
    mutates_args=[],
    fake_impl=rotary_embedding_dispatch_fake,
)
