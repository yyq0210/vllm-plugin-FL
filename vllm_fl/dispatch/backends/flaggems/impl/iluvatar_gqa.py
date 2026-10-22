# Copyright (c) 2026 LeanScale.
"""FlagOS-selected Triton backend with a bounded Iluvatar GQA specialization."""
import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend, TritonAttentionImpl,
)
from vllm.v1.attention.ops.triton_unified_attention import is_batch_invariant

logger = init_logger(__name__)


class IluvatarGQABackend(TritonAttentionBackend):
    @staticmethod
    def get_impl_cls():
        return IluvatarGQAImpl


class IluvatarGQAImpl(TritonAttentionImpl):
    def _supports_gqa(self, query, kv_cache, meta, output, output_scale, output_block_scale):
        if meta is None or output is None or is_batch_invariant:
            return False
        if not (self.num_heads == 16 and self.num_kv_heads == 2 and self.head_size == 128):
            return False
        if (self.attn_type != AttentionType.DECODER or self.alibi_slopes is not None
                or self.sinks is not None or self.use_alibi_sqrt
                or self.use_td
                or self.sliding_window != (-1, -1) or self.chunk_lookback != -1
                or self.logits_soft_cap != 0 or self._is_per_token_head_quant
                or self.kv_cache_dtype not in ("auto", "float16", "bfloat16")
                or output_scale is not None or output_block_scale is not None):
            return False
        if (meta.causal is not True or meta.use_cascade
                or meta.mm_prefix_range_tensor is not None):
            return False
        if (query.dtype not in (torch.bfloat16, torch.float16)
                or query.dtype != kv_cache.dtype or output.dtype != query.dtype
                or query.device.type != "cuda" or query.stride(-1) != 1
                or output.stride(-1) != 1 or kv_cache.stride(-1) != 1):
            return False
        if (query.ndim != 3 or query.shape[1:] != (16, 128)
                or output.shape != query.shape or kv_cache.ndim != 5
                or kv_cache.shape[1] != 2 or kv_cache.shape[-2:] != (2, 128)):
            return False
        nseq = meta.seq_lens.numel()
        buffers = (meta.softmax_segm_output, meta.softmax_segm_max, meta.softmax_segm_expsum)
        if any(x is None or not x.is_contiguous() or x.dtype != torch.float32 for x in buffers):
            return False
        if not (buffers[0].shape[0] >= nseq and buffers[0].shape[1:] == (16, 16, 128)
                and buffers[1].shape == buffers[2].shape == buffers[0].shape[:-1]):
            return False
        return (meta.query_start_loc.numel() == nseq + 1
                and meta.query_start_loc.is_contiguous() and meta.seq_lens.is_contiguous()
                and meta.block_table.stride(-1) == 1)

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        if not self._supports_gqa(query, kv_cache, attn_metadata, output,
                                  output_scale, output_block_scale):
            return super().forward(layer, query, key, value, kv_cache, attn_metadata,
                                   output, output_scale, output_block_scale)
        from flag_gems.runtime.backend._iluvatar.fused.paged_gqa import paged_gqa

        if not getattr(self, "_leanscale_gqa_logged", False):
            logger.info("LeanScale GQA active: Q16/KV2/D128, segmented decode and mixed batch; decode_warps=4 decode_stages=1 segments=16 decode_layout=1 decode_rows=8 prefill_rows=64 prefill_warps=4 prefill_stages=1 prefill_tile=32; layer=%s",
                        getattr(layer, "layer_name", "unknown"))
            self._leanscale_gqa_logged = True
        meta = attn_metadata
        k_cache, v_cache = kv_cache.unbind(1)
        n = meta.num_actual_tokens
        paged_gqa(query[:n], k_cache, v_cache, output[:n],
                  meta.query_start_loc, meta.seq_lens, meta.block_table, self.scale,
                  meta.max_query_len, meta.softmax_segm_output,
                  meta.softmax_segm_max, meta.softmax_segm_expsum,
                  segments=16, decode_warps=4, decode_stages=1,
                  decode_layout=1, decode_rows=8,
                  prefill_rows=64, prefill_warps=4, prefill_stages=1, prefill_tile=32)
        return output
