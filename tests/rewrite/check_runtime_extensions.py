#!/usr/bin/env python3
"""Source-valid bounded profiler/header, host-allocation and callback matrix."""
from pathlib import Path
import argparse,hashlib,json,subprocess,tempfile,shutil,re
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True)
p.add_argument('--cuda-path',type=Path,required=True)
p.add_argument('--resource-dir',type=Path,required=True)
p.add_argument('--clang',type=Path)
p.add_argument('--evidence-dir',type=Path)
p.add_argument('--source-only',action='store_true')
a=p.parse_args()
a.binary=a.binary.resolve();cuda=a.cuda_path.resolve();resource=a.resource_dir.resolve()
if not (cuda/'include/cuda_runtime.h').is_file() or not (resource/'include/__clang_cuda_runtime_wrapper.h').is_file():
 p.error('supply real CUDA and complete Clang resources')
choices=[a.clang] if a.clang else [a.binary.parent/'clang++',resource.parents[2]/'bin/clang++',Path(shutil.which('clang++') or '/missing')]
clang=next((x.resolve() for x in choices if x.is_file()),None)
if clang is None or not a.binary.is_file():p.error('supply a real binary and matching --clang')
version=subprocess.check_output([str(clang),'--version'],text=True)
major=re.search(r'\bclang version (\d+)(?:\.|\s)',version)
if major is None or major.group(1)!=resource.name.split('.')[0]:p.error('clang version must match resources')
helpers=Path(__file__).resolve().parent/'fixtures/nvidia_samples/Common'
temporary=tempfile.TemporaryDirectory(prefix='ascify-runtime-extensions-')
a.output=a.evidence_dir.resolve() if a.evidence_dir else Path(temporary.name)/'evidence'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
prof='#include <cuda_runtime.h>\n#include <cuda_profiler_api.h>\n'
body='int main() { return cudaProfilerStart() == cudaSuccess ? cudaProfilerStop() : 1; }\n'
cases=[
 ('profiler_direct',prof+body,['ascify::cudaProfilerStart()', 'ascify::cudaProfilerStop()', '<acl/acl_prof.h>'],[],False),
 ('profiler_native_stdio',prof+'#include <stdio.h>\n'+body,['ascify::cudaProfilerStart()'],[],False),
 ('profiler_checked',prof+'#include <helper_cuda.h>\nint main(){checkCudaErrors(cudaProfilerStart()); checkCudaErrors(cudaProfilerStop());}\n',['ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS(ascify::cudaProfilerStart())','ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS(ascify::cudaProfilerStop())'],['#include <helper_cuda.h>'],False),
 ('event_query','#include <cuda_runtime.h>\nint main(){cudaEvent_t e;cudaEventCreate(&e);return cudaEventQuery(e)==cudaErrorNotReady;}\n',['ascify::cudaEventCreate(&e)','ascify::cudaEventQuery(e)==ACL_ERROR_RT_EVENT_NOT_COMPLETE'],[],False),
 ('event_checked','#include <cuda_runtime.h>\n#include <helper_cuda.h>\nint main(){cudaEvent_t e;checkCudaErrors(cudaEventQuery(e));}\n',['ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS(ascify::cudaEventQuery(e))'],['#include <helper_cuda.h>'],False),
 ('host_typed_flags','#include <cuda_runtime.h>\nint main(){float *p; return cudaMallocHost(&p,64,0);}\n',['ascify::cudaMallocHost(&p,64,0)'],[],False),
 ('host_void_flags','#include <cuda_runtime.h>\nint main(){void *p; return cudaMallocHost(&p,64,0);}\n',['ascify::cudaMallocHost(&p,64,0)'],[],False),
 ('host_flags_once','#include <cuda_runtime.h>\nint main(){int *p;unsigned flags=0; return cudaMallocHost(&p,64,flags++);}\n',['ascify::cudaMallocHost(&p,64,flags++)'],[],False),
 ('callback_direct','#include <cuda_runtime.h>\nvoid CUDART_CB done(void*){}\nint main(){cudaStream_t s;return cudaLaunchHostFunc(s,done,nullptr);}\n',['void  done(void*){}','ascify::cudaLaunchHostFunc(s,done,nullptr)'],['CUDART_CB'],False),
 ('callback_forward_decl','#include <cuda_runtime.h>\nvoid CUDART_CB done(void*);\nvoid CUDART_CB done(void*){}\nint main(){cudaHostFn_t cb=done;cudaStream_t s;return cudaLaunchHostFunc(s,cb,nullptr);}\n',['void  done(void*);','void  done(void*){}','ascify::cudaHostFn_t cb=done'],['CUDART_CB'],False),
 ('callback_user_empty','#include <cuda_runtime.h>\n#undef CUDART_CB\n#define CUDART_CB\nvoid CUDART_CB done(void*){}\n',['void CUDART_CB done(void*){}'],[],False),
 ('callback_user_attribute','#include <cuda_runtime.h>\n#undef CUDART_CB\n#define CUDART_CB __attribute__((noinline))\nvoid CUDART_CB done(void*){}\n',['void CUDART_CB done(void*){}','__attribute__((noinline))'],[],False),
 ('callback_macro_alias','#include <cuda_runtime.h>\n#define CALLING CUDART_CB\nvoid CALLING done(void*){}\n',['#define CALLING CUDART_CB','void CALLING done(void*){}'],[],False),
 ('profiler_conditional','#include <cuda_runtime.h>\n#if __has_include(<cuda_profiler_api.h>)\n#include <cuda_profiler_api.h>\n#endif\n'+body,[],[],True),
 ('profiler_guard_observation',prof+'#ifdef __CUDA_PROFILER_API_H__\nint observed=1;\n#endif\n'+body,[],[],True),
 ('profiler_guard_inactive',prof+'#if 0\nint __CUDA_PROFILER_API_H__;\n#endif\n'+body,[],[],True),
 ('profiler_inquiry_after',prof+'#if __has_include(<cuda_profiler_api.h>)\nint observed=1;\n#endif\n'+body,[],[],True),
 ('profiler_macro_include','#include <cuda_runtime.h>\n#define H <cuda_profiler_api.h>\n#include H\n'+body,[],[],True),
 ('profiler_pragma_guard',prof+'#pragma push_macro("__CUDA_PROFILER_API_H__")\n'+body,[],[],True),
 ('profiler_caller_new',prof+'#include <new>\n'+body,[],[],True),
 ('profiler_paste',prof+'#define JOIN(a,b) a##b\n'+body,[],[],True),
 ('profiler_helper_macro_shadow',prof+'#define ProfilerManager COLLISION\n'+body,[],[],True),
]
cap='cudaDevAttrComputeCapabilityMajor'
for name, code, expected, absent in [
 ('sm_major_selector','int main(){int n=1;return cudaDeviceGetAttribute(&n,'+cap+',0);}', ['::ascify::'+cap], []),
 ('sm_minor_selector','int main(){int n=1;return cudaDeviceGetAttribute(&n,cudaDevAttrComputeCapabilityMinor,0);}', ['::ascify::cudaDevAttrComputeCapabilityMinor'], []),
 ('sm_selector_sizeof','int main(){return sizeof('+cap+');}', ['sizeof('+cap+')'], ['::ascify::'+cap]),
 ('sm_selector_numeric','int main(){return static_cast<int>('+cap+');}', ['static_cast<int>('+cap+')'], ['::ascify::'+cap]),
 ('sm_selector_variable','int main(){int n=1;cudaDeviceAttr attr='+cap+';return cudaDeviceGetAttribute(&n,attr,0);}', ['attr='+cap], ['::ascify::'+cap]),
 ('sm_selector_macro','#define ATTR '+cap+'\nint main(){int n=1;return cudaDeviceGetAttribute(&n,ATTR,0);}', ['#define ATTR '+cap], ['::ascify::'+cap]),
 ('sm_selector_template','template<class T> int query(){int n=1;return cudaDeviceGetAttribute(&n,'+cap+',0);}', [cap], ['::ascify::'+cap]),
 ('sm_selector_mixed','int main(){int n=1;cudaDeviceGetAttribute(&n,'+cap+',0);return sizeof('+cap+');}', ['::ascify::'+cap,'sizeof('+cap+')'], []),
]:
 cases.append((name,'#include <cuda_runtime.h>\n'+code+'\n',expected,absent,False))
a.output.mkdir(parents=True,exist_ok=False)
summary={'source_only':a.source_only,'binary_sha256':sha(a.binary),'clang_sha256':sha(clang),'cases':[],'passed':False}
def save():(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
def run(d,stage,argv):
 (d/(stage+'.argv.json')).write_text(json.dumps(list(map(str,argv)),indent=2)+'\n')
 r=subprocess.run(list(map(str,argv)),capture_output=True,text=True,timeout=60)
 (d/(stage+'.stdout')).write_text(r.stdout);(d/(stage+'.stderr')).write_text(r.stderr)
 return r
save()
try:
 for name,source,expected,absent,refused in cases:
  d=a.output/name;d.mkdir();cu=d/'input.cu';cu.write_text(source);out=d/'output.cce';receipt=d/'receipt.json'
  row={'name':name,'input_sha256':sha(cu),'expected_publication_refusal':refused,'passed':False};summary['cases'].append(row)
  common=['-x','cuda','-std=c++17','-I'+str(helpers),'--cuda-path='+str(cuda),'-resource-dir='+str(resource)]
  r=run(d,'syntax',[clang,cu,*common,'--cuda-host-only','-nocudalib','-fsyntax-only','-Wno-unknown-cuda-version']);row['syntax_rc']=r.returncode;save();assert r.returncode==0,(name,r.stderr)
  if not a.source_only:
   r=run(d,'conversion',[a.binary,cu,'--target-policy=dav-c310-vec','--simt-math=fast','--target-recipe=none','--default-preprocessor','--cuda-path='+str(cuda),'--clang-resource-directory='+str(resource),'--migration-receipt='+str(receipt),'-o',out,'--',*common]);row['conversion_rc']=r.returncode
   if refused:
    assert r.returncode!=0 and not out.exists(),(name,r.stderr)
    assert 'profiler header boundary' in r.stderr or 'Ascify cannot publish CUDA compatibility output' in r.stderr,(name,r.stderr)
   else:
    assert r.returncode==0,(name,r.stderr)
    text=out.read_text(); assert all(x in text for x in expected),(name,text,expected)
    assert not any(x in text for x in absent),(name,text,absent)
    row['output_sha256']=sha(out)
   row['receipt_sha256']=sha(receipt)
  assert sha(cu)==row['input_sha256'];row['passed']=True;save();print(name,'PASS',flush=True)
 assert sha(a.binary)==summary['binary_sha256'] and sha(clang)==summary['clang_sha256'];summary.update(passed=True,total=len(cases));save()
except Exception as e:
 summary['error']=str(e);save();raise
