# SPDX-License-Identifier: Apache-2.0
"""Opt-in load-time MLP weight layout for MiniCPM5-2B on BI-V150.

The logical [out, in] matrix and Parameter object are preserved. Only its
storage changes to stride (1, out); native linear and compiled SwiGLU remain
unchanged. Set VLLM_FL_MLP_WEIGHT_LAYOUT=1 before starting the engine.
"""
import functools
import os
import re

import torch
from vllm.logger import init_logger

logger = init_logger("vllm.leanscale.mlp_weight_layout")
ENV_NAME = "VLLM_FL_MLP_WEIGHT_LAYOUT"
_PREFIX = re.compile(r"model\.layers\.(\d+)\.mlp\.(gate_up_proj|down_proj)")
_SHAPES = {"gate_up_proj": (12288, 2048), "down_proj": (2048, 6144)}


def register_mlp_weight_layout_env():
    """Include the startup flag in vLLM's environment-based compile cache key."""
    from vllm import envs
    envs.environment_variables.setdefault(
        ENV_NAME, lambda: os.environ.get(ENV_NAME, "0") == "1"
    )


def _supported_config(config):
    if config is None:
        return False
    model = getattr(config, "model_config", None)
    hf = getattr(model, "hf_text_config", None)
    expected = {
        "model_type": "llama", "hidden_size": 2048,
        "intermediate_size": 6144, "num_hidden_layers": 42,
        "num_attention_heads": 16, "num_key_value_heads": 2,
        "head_dim": 128, "hidden_act": "silu", "vocab_size": 130560,
    }
    if any(getattr(hf, key, None) != value for key, value in expected.items()):
        return False
    if getattr(hf, "architectures", None) != ["LlamaForCausalLM"]:
        return False
    if getattr(model, "dtype", None) != torch.bfloat16:
        return False
    if getattr(model, "quantization", None) is not None:
        return False
    if getattr(model, "enable_sleep_mode", False):
        return False
    parallel = getattr(config, "parallel_config", None)
    if any(getattr(parallel, key, None) != 1 for key in (
        "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
        "decode_context_parallel_size", "prefill_context_parallel_size",
    )):
        return False
    if any(getattr(config, key, None) is not None for key in (
        "quant_config", "lora_config", "speculative_config",
        "weight_transfer_config",
    )):
        return False
    offload = getattr(config, "offload_config", None)
    if (getattr(getattr(offload, "uva", None), "cpu_offload_gb", None) != 0
            or getattr(getattr(offload, "prefetch", None),
                       "offload_group_size", None) != 0):
        return False
    return True


def _target_projection(layer):
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear, RowParallelLinear, UnquantizedLinearMethod,
    )
    match = _PREFIX.fullmatch(getattr(layer, "prefix", ""))
    if match is None or not 0 <= int(match[1]) < 42:
        return None
    projection = match[2]
    expected_type = (MergedColumnParallelLinear if projection == "gate_up_proj"
                     else RowParallelLinear)
    if type(layer) is not expected_type:
        return None
    if (type(getattr(layer, "quant_method", None)) is not UnquantizedLinearMethod
            or getattr(layer, "quant_config", None) is not None
            or getattr(layer, "tp_size", None) != 1
            or getattr(layer, "has_bias", True)
            or getattr(layer, "bias", None) is not None):
        return None
    weight = getattr(layer, "weight", None)
    if not isinstance(weight, torch.nn.Parameter):
        return None
    shape = _SHAPES[projection]
    if (tuple(weight.shape) != shape or weight.dtype != torch.bfloat16
            or weight.device.type != "cuda" or weight.requires_grad
            or getattr(weight, "_vllm_is_uva_offloaded", False)
            or getattr(layer, "params_dtype", None) != torch.bfloat16
            or getattr(layer, "input_size", None) != shape[1]
            or getattr(layer, "output_size", None) != shape[0]):
        return None
    # Do not repack views or unverified layouts. The target layout is allowed
    # so repeated post-load calls can be recognized without an extra copy.
    if (weight.storage_offset() != 0
            or weight.untyped_storage().nbytes() != weight.numel() * weight.element_size()
            or weight.stride() not in ((shape[1], 1), (1, shape[0]))):
        return None
    return projection


@torch.no_grad()
def _repack_weight(weight):
    """Replace only storage, never the Parameter or its loader/TP attributes."""
    if weight.stride() == (1, weight.shape[0]):
        return False
    packed = weight.t().contiguous().t()
    if not torch.equal(weight, packed):
        raise RuntimeError("MLP layout conversion changed weight values")
    weight.data = packed
    return True


def install_mlp_weight_layout():
    register_mlp_weight_layout_env()
    from vllm.platforms import current_platform
    if getattr(current_platform, "vendor_name", None) != "iluvatar":
        return
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    original = UnquantizedLinearMethod.process_weights_after_loading
    if getattr(original, "_leanscale_mlp_weight_layout", False):
        return

    @functools.wraps(original)
    def after_loading(self, layer):
        result = original(self, layer)
        if os.environ.get(ENV_NAME, "0") != "1":
            return result
        from vllm import envs
        from vllm.config import get_current_vllm_config_or_none
        if envs.VLLM_BATCH_INVARIANT:
            return result
        if not _supported_config(get_current_vllm_config_or_none()):
            return result
        if _target_projection(layer) is None:
            return result
        weight = layer.weight
        if torch.cuda.get_device_name(weight.device) != "Iluvatar BI-V150":
            return result
        old_stride = weight.stride()
        if _repack_weight(weight):
            logger.info(
                "LeanScale MLP layout active: prefix=%s shape=%s old_stride=%s "
                "new_stride=%s exact_values=True parameter_preserved=True bytes=%d",
                layer.prefix, tuple(weight.shape), old_stride, weight.stride(),
                weight.numel() * weight.element_size(),
            )
        return result

    after_loading._leanscale_mlp_weight_layout = True
    UnquantizedLinearMethod.process_weights_after_loading = after_loading
