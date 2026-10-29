"""Actual-device integration checks; production hook and original JIT path."""
from pathlib import Path
import json,os
import torch
import vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar
import vllm.v1.sample.ops.topk_topp_triton as base
import vllm.v1.sample.ops.topk_topp_sampler as sampler
from vllm_fl.patches import topp_row_parallel as patch

def main():
    out=Path(os.environ['LEAN_SCALE_VALIDATION_DIR']);checks=[]
    def record(name,ok,**kw):
        r=dict(name=name,passed=bool(ok),**kw);checks.append(r)
        with (out/'checks.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
        assert ok,r
    from vllm.platforms import current_platform
    assert current_platform.vendor_name=='iluvatar'
    original=base.apply_top_k_top_p_triton
    os.environ[patch.ENV_NAME]='0'
    record('disabled_keeps_aliases',not patch.install_topp_row_parallel() and base.apply_top_k_top_p_triton is original and sampler.apply_top_k_top_p_triton is original)
    oldhash=patch.EXPECTED_SOURCE_SHA256;patch.EXPECTED_SOURCE_SHA256='0'*64;os.environ[patch.ENV_NAME]='1'
    record('source_drift_fallback',not patch.install_topp_row_parallel() and base.apply_top_k_top_p_triton is original)
    patch.EXPECTED_SOURCE_SHA256=oldhash
    record('install',patch.install_topp_row_parallel())
    candidate=base.apply_top_k_top_p_triton
    record('alias_and_idempotence',sampler.apply_top_k_top_p_triton is candidate and patch.install_topp_row_parallel() and base.apply_top_k_top_p_triton is candidate)
    patch.register_topp_row_parallel_env()
    from vllm import envs
    record('cache_environment_flags',patch.ENV_NAME in envs.environment_variables and patch.SHADOW_ENV in envs.environment_variables)
    torch.manual_seed(202610294);v=130560
    for b in [1,4,8,16,17,30,31,32,33,48,63,64,65,128]:
        x=torch.randn((b,v),device='cuda');p=torch.full((b,),.95,device='cuda')
        a=original(x.clone(),None,p);y=x.clone();before=patch._stats['candidate_calls'];z=candidate(y,None,p)
        record('batch_'+str(b),torch.equal(a,z) and z is y and patch._stats['candidate_calls']-before==int(17<=b<=64))
    # Unsupported vocabulary and top-k still use the original path.
    for label,vocab,kval in [('vocab',1024,None),('topk',v,50)]:
        x=torch.randn((32,vocab),device='cuda');p=torch.full((32,),.95,device='cuda')
        k=torch.full((32,),kval,dtype=torch.int32,device='cuda') if kval else None
        before=patch._stats['candidate_calls'];a=original(x.clone(),k,p);b=candidate(x.clone(),k,p)
        record('fallback_'+label,torch.equal(a,b) and patch._stats['candidate_calls']==before)
    x=torch.randn((32,v*2),device='cuda')[:,::2];p=torch.full((32,),.95,device='cuda');before=patch._stats['candidate_calls']
    record('column_stride_fallback',torch.equal(original(x.clone(),None,p),candidate(x,None,p)) and patch._stats['candidate_calls']==before)
    x=torch.randn((32,v+32),device='cuda')[:,:v];expected=original(x.clone(),None,p);before=patch._stats['candidate_calls']
    record('row_stride_candidate',torch.equal(candidate(x,None,p),expected) and patch._stats['candidate_calls']==before+1)
    # An existing baseline graph may retain its old16-row buffer after expansion.
    base._TRITON_BUFFER_CACHE.clear();base._TRITON_TABLE_CACHE.clear()
    x=torch.randn((64,v),device='cuda');p=torch.full((64,),.95,device='cuda');a=x.clone();b=x.clone()
    for _ in range(3):a.copy_(x);original(a,None,p)
    torch.cuda.synchronize();key=(x.device,x.dtype,v);old_buffer=base._TRITON_BUFFER_CACHE[key]
    ga=torch.cuda.CUDAGraph()
    with torch.cuda.graph(ga):a.copy_(x);original(a,None,p)
    for _ in range(3):b.copy_(x);candidate(b,None,p)
    new_buffer=base._TRITON_BUFFER_CACHE[key]
    record('cache_expands_single_entry',len(base._TRITON_BUFFER_CACHE)==1 and old_buffer.shape[0]==16 and new_buffer.shape[0]==64 and new_buffer.data_ptr()!=old_buffer.data_ptr())
    torch.cuda.synchronize();gb=torch.cuda.CUDAGraph()
    with torch.cuda.graph(gb):b.copy_(x);candidate(b,None,p)
    x.mul_(1.5);p.fill_(.8);ga.replay();gb.replay();torch.cuda.synchronize()
    record('old_and_new_graph_after_growth',torch.equal(a,b) and torch.equal(a,original(x.clone(),None,p)))
    del ga,gb,old_buffer,new_buffer,a,b
    # reset empties shared caches, then an eager call reinitializes safely.
    base.reset_buffer_cache();before=patch._stats['candidate_calls'];candidate(x,None,p)
    record('reset_reinitializes',patch._stats['candidate_calls']==before+1 and base._TRITON_BUFFER_CACHE[key].shape[0]==64)
    # The fallback remains responsible for CPU dtype/layout errors, preserving behavior.
    record('cpu_guard',not patch._eligible(torch.empty(32,v),None,torch.full((32,),.95),-float('inf')))
    rng=torch.cuda.get_rng_state().clone();y=torch.randn((32,v),device='cuda');rng=torch.cuda.get_rng_state().clone();p=torch.full((32,),.95,device='cuda')
    a=original(y.clone(),None,p);b=candidate(y.clone(),None,p)
    record('mask_rng_unchanged',torch.equal(rng,torch.cuda.get_rng_state()))
    for per_request in [False,True]:
        def generators():return {0:torch.Generator(device='cuda').manual_seed(99)} if per_request else {}
        saved=torch.cuda.get_rng_state().clone();g1=generators();g2=generators()
        ta=sampler.random_sample(a.softmax(-1),g1);after=torch.cuda.get_rng_state().clone()
        torch.cuda.set_rng_state(saved);tb=sampler.random_sample(b.softmax(-1),g2)
        record('tokens_rng_'+str(per_request),torch.equal(ta,tb) and torch.equal(after,torch.cuda.get_rng_state()) and all(torch.equal(g1[i].get_state(),g2[i].get_state()) for i in g1))
    torch.cuda.synchronize()
    (out/'result.json').write_text(json.dumps(dict(status='passed',checks=len(checks),candidate_calls=patch._stats['candidate_calls'],
        persistent_scratch_bytes=base._TRITON_BUFFER_CACHE[key].numel()*4,full_model_accuracy='deferred_by_user'),indent=2)+'\n')
    print(json.dumps(dict(status='passed',checks=len(checks))),flush=True)
if __name__=='__main__':main()
