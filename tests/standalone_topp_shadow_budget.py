"""Device regression of warmup admission, late B64, and shadow graph fallback."""
from pathlib import Path
import json
import os
import torch
import vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar
import vllm.v1.sample.ops.topk_topp_triton as base
from vllm_fl.patches import topp_row_parallel as patch


def main():
    out = Path(os.environ['LEAN_SCALE_VALIDATION_DIR'])
    arm = out / 'shadow.arm'
    assert not arm.exists()
    os.environ[patch.ENV_NAME] = '1'
    os.environ[patch.SHADOW_ENV] = '1'
    os.environ[patch.SHADOW_ARM_ENV] = str(arm)
    original = base.apply_top_k_top_p_triton
    assert patch.install_topp_row_parallel()
    candidate = base.apply_top_k_top_p_triton
    checks = []
    def record(name, ok):
        row = dict(name=name, passed=bool(ok));checks.append(row)
        with (out / 'checks.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        assert ok, row
    from vllm import envs
    record('arm_flag_registered', envs.environment_variables[patch.SHADOW_ARM_ENV]() == str(arm))
    torch.manual_seed(202610301)
    x = torch.randn(64, 130560, device='cuda')
    p = torch.full((64,), .95, device='cuda')
    for _ in range(3):
        candidate(x.clone(), None, p)
    record('warmup_B64_no_checks', patch._stats['shadow_checks'] == 0)
    arm.touch(exist_ok=False)
    for b in range(17, 33):
        for _ in range(2):
            candidate(x[:b].clone(), None, p[:b])
    record('small_batch_checks_32', patch._stats['shadow_checks'] == 32)
    for i in range(2):
        rng = torch.cuda.get_rng_state().clone()
        actual = candidate(x.clone(), None, p)
        reference = original(x.clone(), None, p)
        record('late_B64_exact_rng_' + str(i), torch.equal(actual, reference) and torch.equal(rng, torch.cuda.get_rng_state()) and patch._stats['shadow_checks'] == 33 + i)
    for _ in range(5):
        candidate(x.clone(), None, p)
    record('B64_checks_capped', patch._stats['shadow_checks'] == 34)
    before = dict(patch._stats)
    y = x.clone();graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        y.copy_(x)
        candidate(y, None, p)
    record('shadow_capture_falls_back', patch._stats == before)
    graph.replay();torch.cuda.synchronize()
    record('shadow_capture_exact', torch.equal(y, original(x.clone(), None, p)))
    x.mul_(1.5);p.fill_(.7)
    graph.replay();torch.cuda.synchronize()
    record('shadow_capture_updated_input_exact', torch.equal(y, original(x.clone(), None, p)))
    result = dict(status='passed', checks=len(checks), shadow_checks=patch._stats['shadow_checks'], model_accuracy='deferred_by_user')
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
