"""Run with pytest; GPU checks require LEANSCALE_TEST_GPU=1."""
import copy
import os
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch
import vllm_fl  # registers the vendor torch compatibility shim
from vllm_fl.patches import mlp_weight_layout as layout


@pytest.fixture(autouse=True)
def single_rank_parameter_metadata(monkeypatch):
    # Isolated layer tests have no distributed process group. Constructors
    # still query TP metadata even when the layer uses disable_tp=True.
    # No collective is invoked; the real service will initialize its group.
    import vllm.distributed.parallel_state as parallel_state
    monkeypatch.setattr(parallel_state, "_TP", NS(rank_in_group=0, world_size=1))


def config():
    return NS(
        model_config=NS(dtype=torch.bfloat16, quantization=None,
                        hf_text_config=NS(
                            model_type="llama", architectures=["LlamaForCausalLM"],
                            hidden_size=2048, intermediate_size=6144,
                            num_hidden_layers=42, num_attention_heads=16,
                            num_key_value_heads=2, head_dim=128,
                            hidden_act="silu", vocab_size=130560)),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
                           data_parallel_size=1, decode_context_parallel_size=1,
                           prefill_context_parallel_size=1),
        offload_config=NS(uva=NS(cpu_offload_gb=0),
                          prefetch=NS(offload_group_size=0)),
    )


@pytest.mark.parametrize("path,value", [
    ("model_config.hf_text_config.hidden_size", 4096),
    ("model_config.hf_text_config.intermediate_size", 8192),
    ("model_config.hf_text_config.num_hidden_layers", 40),
    ("model_config.hf_text_config.num_key_value_heads", 4),
    ("model_config.hf_text_config.architectures", ["OtherModel"]),
    ("model_config.hf_text_config.hidden_act", "gelu"),
    ("model_config.dtype", torch.float16),
    ("model_config.quantization", "awq"),
    ("model_config.enable_sleep_mode", True),
    ("parallel_config.tensor_parallel_size", 2),
    ("parallel_config.pipeline_parallel_size", 2),
    ("parallel_config.data_parallel_size", 2),
    ("parallel_config.decode_context_parallel_size", 2),
    ("parallel_config.prefill_context_parallel_size", 2),
    ("lora_config", NS()), ("speculative_config", NS()),
    ("quant_config", NS()), ("weight_transfer_config", NS()),
    ("offload_config.uva.cpu_offload_gb", 1),
    ("offload_config.prefetch.offload_group_size", 2),
])
def test_unsupported_config(path, value):
    candidate = config()
    assert layout._supported_config(candidate)
    target = candidate
    parts = path.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)
    assert not layout._supported_config(candidate)


def test_missing_config():
    assert not layout._supported_config(None)
    assert not layout._supported_config(NS())


def test_parameter_and_reload_semantics():
    from vllm.model_executor.parameter import ModelWeightParameter
    loader = lambda *args: None
    parameter = ModelWeightParameter(
        data=torch.arange(35, dtype=torch.float32).reshape(7, 5),
        input_dim=1, output_dim=0, weight_loader=loader,
    )
    before = parameter.detach().clone()
    metadata = dict(parameter.__dict__)
    identity, pointer = id(parameter), parameter.data_ptr()
    assert layout._repack_weight(parameter)
    assert id(parameter) == identity and parameter.__dict__ == metadata
    assert torch.equal(parameter, before) and parameter.stride() == (1, 7)
    assert pointer != parameter.data_ptr()
    pointer = parameter.data_ptr()
    assert not layout._repack_weight(parameter)
    assert parameter.data_ptr() == pointer
    with torch.no_grad():
        parameter.copy_(before + 1)
    assert not layout._repack_weight(parameter)
    assert torch.equal(parameter, before + 1)
    # A reload that replaces storage with a contiguous tensor must repack again.
    parameter.data = before.contiguous()
    assert layout._repack_weight(parameter)
    assert parameter.__dict__ == metadata and torch.equal(parameter, before)


def test_original_first_default_off_exception_and_idempotent_install(monkeypatch):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    calls = []
    token = object()
    def original(self, layer):
        calls.append(layer)
        if layer == "error":
            raise ValueError("original failure")
        return token
    monkeypatch.setattr(UnquantizedLinearMethod, "process_weights_after_loading", original)
    monkeypatch.setattr("vllm.platforms.current_platform", NS(vendor_name="iluvatar"))
    monkeypatch.delenv(layout.ENV_NAME, raising=False)
    layout.install_mlp_weight_layout()
    installed = UnquantizedLinearMethod.process_weights_after_loading
    layout.install_mlp_weight_layout()
    assert UnquantizedLinearMethod.process_weights_after_loading is installed
    with patch.object(layout, "_supported_config", side_effect=AssertionError("off")):
        assert UnquantizedLinearMethod().process_weights_after_loading("layer") is token
    assert calls == ["layer"]
    monkeypatch.setenv(layout.ENV_NAME, "1")
    with pytest.raises(ValueError, match="original failure"):
        UnquantizedLinearMethod().process_weights_after_loading("error")
    assert calls == ["layer", "error"]


def test_compile_cache_flag(monkeypatch):
    from vllm import envs
    layout.register_mlp_weight_layout_env()
    # Compare the public cache factors in separate states. This environment
    # is a startup choice; hot switching a running engine is not supported.
    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()
    monkeypatch.setenv(layout.ENV_NAME, "0")
    before = envs.compile_factors()
    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()
    monkeypatch.setenv(layout.ENV_NAME, "1")
    after = envs.compile_factors()
    assert before[layout.ENV_NAME] is False and after[layout.ENV_NAME] is True
    assert {k: v for k, v in before.items() if k != layout.ENV_NAME} == {
        k: v for k, v in after.items() if k != layout.ENV_NAME}


@pytest.mark.skipif(os.environ.get("LEANSCALE_TEST_GPU") != "1", reason="GPU opt-in")
@pytest.mark.parametrize("projection", ["gate_up_proj", "down_proj"])
def test_real_gpu_layer_layout_memory_and_graph(projection, monkeypatch):
    # Standalone pytest resolves vLLM's placeholder platform during collection.
    # Exercise the actual vendor platform used by the serve entry point.
    from vllm_fl.platform import PlatformFL
    import vllm.model_executor.layers.linear as linear_module
    platform = PlatformFL()
    assert platform.vendor_name == "iluvatar"
    monkeypatch.setattr("vllm.platforms.current_platform", platform)
    monkeypatch.setattr(linear_module, "current_platform", platform)
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear, RowParallelLinear,
    )
    torch.manual_seed(202610281)
    prefix = "model.layers.0.mlp." + projection
    kwargs = dict(bias=False, params_dtype=torch.bfloat16,
                  prefix=prefix, disable_tp=True)
    with torch.device("cuda"):
        layer = (MergedColumnParallelLinear(2048, [6144, 6144], **kwargs)
                 if projection == "gate_up_proj"
                 else RowParallelLinear(6144, 2048, **kwargs))
    assert torch.cuda.get_device_name(layer.weight.device) == "Iluvatar BI-V150"
    with torch.no_grad():
        layer.weight.normal_(std=.02)
    old_cpu = layer.weight.detach().cpu()
    weight = layer.weight
    metadata = copy.copy(weight.__dict__)
    old_pointer = weight.data_ptr()
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    monkeypatch.setenv(layout.ENV_NAME, "1")
    layout.install_mlp_weight_layout()
    with patch("vllm.config.get_current_vllm_config_or_none", return_value=config()):
        assert layout._target_projection(layer) == projection
        layer.quant_method.process_weights_after_loading(layer)
        torch.cuda.synchronize()
        assert layer.weight is weight and weight.__dict__ == metadata
        assert torch.equal(weight.cpu(), old_cpu)
        assert weight.stride() == (1, weight.shape[0])
        assert old_pointer != weight.data_ptr()
        assert torch.cuda.memory_allocated() == allocated, "Old weight still allocated"
        pointer = weight.data_ptr()
        layer.quant_method.process_weights_after_loading(layer)
        assert weight.data_ptr() == pointer
        assert torch.cuda.memory_allocated() == allocated
    # Real guard rejects QKV/invalid layer numbers and CPU weights.
    for bad in ("model.layers.0.self_attn.qkv_proj", "model.layers.42.mlp."+projection):
        layer.prefix = bad
        assert layout._target_projection(layer) is None
    layer.prefix = prefix
    old = old_cpu.cuda()
    for m in (1, 8, 16, 512, 2048):
        x = torch.randn((m, weight.shape[1]), device="cuda", dtype=torch.bfloat16)
        expected = torch.nn.functional.linear(x, old)
        actual, _ = layer(x)
        torch.testing.assert_close(actual, expected, atol=.01, rtol=.02)
    x = torch.randn((16, weight.shape[1]), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            layer(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, _ = layer(x)
    with torch.no_grad():
        x.mul_(.97).add_(.01)
        weight.add_(.0001)
    graph.replay()
    expected, _ = layer(x)
    assert torch.equal(output, expected)
    assert weight.data_ptr() == pointer
    layer.weight.data = layer.weight.cpu()
    assert layout._target_projection(layer) is None
