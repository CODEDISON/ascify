#!/usr/bin/env python3
"""Real SDK texture owner/descriptor boundaries; no CCEC or device execution."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

CASES = [
    ('float_fetch','__global__ void k(float*p,cudaTextureObject_t t){p[0]=tex2D<float>(t,.5f,.5f);}', ['ascify::tex2D<float>']),
    ('layered_fetch','__global__ void k(float*p,cudaTextureObject_t t){p[0]=tex2DLayered<float>(t,.5f,.5f,1);}', ['ascify::tex2DLayered<float>']),
    ('array_allocate','void f(){cudaArray*a=nullptr;auto c=cudaCreateChannelDesc<float>();auto s=cudaMallocArray(&a,&c,4,3);}', ['ascify::cudaMallocArray','ascify::cudaCreateChannelDesc<float>']),
    ('typed_pitch','void f(){float*p=nullptr;size_t pitch=0;auto s=cudaMallocPitch(&p,&pitch,16,3);}', ['ascify::cudaMallocPitch']),
    ('empty_descriptor','void f(){cudaResourceDesc r{};cudaTextureDesc d={};r.resType=cudaResourceTypeArray;d.filterMode=cudaFilterModeLinear;}', ['ascify::cudaResourceDesc','ascify::cudaTextureDesc']),
    ('exact_type_zeroing','void f(){cudaResourceDesc r;memset(&r,0,sizeof(cudaResourceDesc));}', ['memset(&r,0,sizeof(ascify::cudaResourceDesc))']),
    ('exact_object_zeroing','void f(){cudaTextureDesc d;memset(&d,0,sizeof(d));}', ['memset(&d,0,sizeof(d))']),
    ('copy_zero_and_units','void f(float*p,cudaArray*a){cudaMemcpy3DParms c={0};c.srcPtr=make_cudaPitchedPtr(p,16,4,3);c.dstArray=a;c.extent=make_cudaExtent(4,3,2);c.kind=cudaMemcpyHostToDevice;auto s=cudaMemcpy3D(&c);}', ['ascify::cudaMemcpy3DParms','ascify::make_cudaPitchedPtr','ascify::cudaMemcpy3D']),
    ('type_alias','using Desc=cudaTextureDesc;void f(){Desc d{};d.addressMode[0]=cudaAddressModeWrap;}', ['using Desc=ascify::cudaTextureDesc']),
    ('layered_flag','void f(){cudaArray*a=nullptr;auto c=cudaCreateChannelDesc<float>();auto s=cudaMalloc3DArray(&a,&c,make_cudaExtent(4,3,2),cudaArrayLayered);}', ['make_cudaExtent(4,3,2),0x01)']),
    ('layered_flag_int_type', 'auto flag=cudaArrayLayered;static_assert(__is_same(decltype(flag),int),"int prvalue");static_assert(__is_same(decltype((cudaArrayLayered)),int),"int expression");', ['auto flag=0x01', 'decltype((0x01))']),
    ('fixed_template_fetch', 'template<int i> __device__ float f(cudaTextureObject_t t){return tex2D<float>(t,i+.5f,.5f);}\n__global__ void k(float*p,cudaTextureObject_t t){p[0]=f<1>(t);}', ['ascify::tex2D<float>']),
    ('fixed_recursive_nttp', 'template<int n> __device__ float recursive_nttp(float x,float y,cudaTextureObject_t t){return tex2D<float>(t,x,y)+recursive_nttp<n-1>(x,y,t);}\ntemplate<> __device__ float recursive_nttp<-1>(float,float,cudaTextureObject_t){return 0;}\n__global__ void k(float*p,cudaTextureObject_t t){p[0]=recursive_nttp<2>(.5f,.5f,t);}', ['ascify::tex2D<float>', 'recursive_nttp<n-1>']),
    ('nttp_static_assert_message', 'template<int n>struct StaticAssertMessage{static_assert(n>0,"template diagnostic");};\nStaticAssertMessage<1> message;cudaTextureDesc descriptor{};', ['static_assert(n>0,"template diagnostic")', 'ascify::cudaTextureDesc descriptor']),
    ('nttp_dependent_return', 'template<int n>struct Result{using type=float;};template<int n>typename Result<n>::type recurse(float x){return recurse<n-1>(x);}\ncudaTextureDesc descriptor{};', None),
    ('nttp_record_argument', 'struct Argument{float x;};template<int n>float recurse(Argument x){return recurse<n-1>(x);}\ncudaTextureDesc descriptor{};', None),
    ('nttp_body_reflection', 'template<int n>float recurse(float x){return float(sizeof(cudaTextureDesc))+recurse<n-1>(x);}\ncudaTextureDesc descriptor{};', None),
    ('nttp_multiple_candidates', 'template<int n>float recurse(float x){return x;}template<int n>float recurse(double x){return float(x);}template<int n>float caller(float x){return recurse<n>(x);}\ncudaTextureDesc descriptor{};', None),
    ('nttp_function_address', 'template<int n>float recurse(float x){return x;}template<int n>void caller(){auto p=&recurse<n>; (void)p;}\ncudaTextureDesc descriptor{};', None),
    ('nttp_argument_reflection', 'template<int n>float recurse(float x){return recurse<n+sizeof(cudaTextureDesc)>(x);}\ncudaTextureDesc descriptor{};', None),
    ('nttp_parameter_pack', 'template<int...n>float recurse(float x){return x;}template<int n>float caller(float x){return recurse<n>(x);}\ncudaTextureDesc descriptor{};', None),
    ('frozen_helper_call', '#include <helper_cuda.h>\nvoid f(cudaTextureObject_t t){checkCudaErrors(cudaDestroyTextureObject(t));}', ['ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS(ascify::cudaDestroyTextureObject(t))']),
    ('frozen_helper_layered', '#include <helper_cuda.h>\nvoid f(){cudaArray*a=nullptr;auto c=cudaCreateChannelDesc<float>();checkCudaErrors(cudaMalloc3DArray(&a,&c,make_cudaExtent(4,3,2),cudaArrayLayered));}', ['ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS(ascify::cudaMalloc3DArray', 'make_cudaExtent(4,3,2),0x01)']),
    ('frozen_full_helpers', '#include <helper_functions.h>\n#include <helper_cuda.h>\nvoid f(cudaTextureObject_t t){checkCudaErrors(cudaDestroyTextureObject(t));}', ['ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS(ascify::cudaDestroyTextureObject(t))']),
    ('flag_stringify', '#define S0(x) #x\n#define S(x) S0(x)\nconst char*s=S(cudaArrayLayered);', None),
    ('flag_direct_stringify', '#define S(x) #x\nconst char*s=S(cudaArrayLayered);', None),
    ('flag_forward', '#define ID(x) (x)\nint flag=ID(cudaArrayLayered);', None),
    ('flag_conditional', '#if cudaArrayLayered\nint flag=1;\n#endif', ['#if 0x01', 'int flag=1;']),
    ('flag_ifdef', '#ifdef cudaArrayLayered\nint flag=1;\n#endif', None),
    ('flag_defined', '#if defined(cudaArrayLayered)\nint flag=1;\n#endif', None),
    ('fake_helper_observer', '#define checkCudaErrors(x) ((x), #x)\nconst char*f(){return checkCudaErrors(cudaDestroyTextureObject(0));}', None),
    ('frozen_helper_outer_observer', '#include <helper_cuda.h>\n#define S0(x) #x\n#define S(x) S0(x)\nconst char*s=S(checkCudaErrors(cudaDestroyTextureObject(0)));', None),
    ('sizeof_observation','size_t f(){return sizeof(cudaTextureDesc);}', None),
    ('alignof_observation','size_t f(){return alignof(cudaResourceDesc);}', None),
    ('offset_observation','size_t f(){return __builtin_offsetof(cudaResourceDesc,res);}', None),
    ('alias_sizeof','using D=cudaTextureDesc;size_t f(){return sizeof(D);}', None),
    ('typedef_alignof','typedef cudaTextureDesc D;size_t f(){return alignof(D);}', None),
    ('alias_offsetof','using R=cudaResourceDesc;size_t f(){return __builtin_offsetof(R,res);}', None),
    ('record_only','cudaTextureDesc descriptor;', ['ascify::cudaTextureDesc descriptor;']),
    ('record_pointer_reflection','size_t f(cudaTextureDesc*p){return sizeof(*p);}', None),
    ('record_type_trait','bool f(){return __is_trivial(cudaTextureDesc);}', None),
    ('record_template_observation','template<class T>struct S{};S<cudaTextureDesc> descriptor;', None),
    ('record_builtin_inquiry','size_t f(cudaTextureDesc*p){return __builtin_object_size(p,0);}', None),
    ('direct_exception_header','#include <exception.h>\ncudaTextureDesc descriptor{};', None),
    ('user_dependent_template','template<class T> T identity(T x){return x;} cudaTextureDesc descriptor{};', None),
    ('uninstantiated_record_reflection','template<class T>size_t size(){return sizeof(T);} cudaTextureDesc descriptor{};', None),
    ('instantiated_record_reflection','template<class T>size_t size(T*p){return sizeof(T);} size_t f(cudaTextureDesc*p){return size(p);}', None),
    ('explicit_record_template_argument','template<class T>size_t size(); size_t f(){return size<cudaTextureDesc>();}', None),
    ('enclosing_layout','struct S{cudaTextureDesc d;};size_t f(){return sizeof(S);}', None),
    ('nonzero_memset','void f(){cudaTextureDesc d;memset(&d,1,sizeof(d));}', None),
    ('wrong_object_memset','void f(){cudaTextureDesc d;cudaResourceDesc r;memset(&d,0,sizeof(r));}', None),
    ('truncated_memset','void f(){cudaTextureDesc d;memset(&d,0,sizeof(d)-1);}', None),
    ('nonempty_descriptor','void f(){cudaTextureDesc d={{cudaAddressModeWrap,cudaAddressModeClamp,cudaAddressModeWrap}};}', None),
    ('macro_initializer','#define INIT {}\nvoid f(){cudaTextureDesc d=INIT;}', None),
    ('macro_empty_initializer','#define NOTHING\nvoid f(){cudaTextureDesc d={NOTHING};}', None),
    ('macro_copy_zero','#define ZERO 0\nvoid f(){cudaMemcpy3DParms c={ZERO};}', None),
    ('nonfloat_fetch','__global__ void k(int*p,cudaTextureObject_t t){p[0]=tex2D<int>(t,.5f,.5f);}', None),
    ('qualified_fetch','__global__ void k(float*p,cudaTextureObject_t t){p[0]=::tex2D<float>(t,.5f,.5f);}', None),
    ('macro_fetch','#define FETCH(t) tex2D<float>(t,.5f,.5f)\n__global__ void k(float*p,cudaTextureObject_t t){p[0]=FETCH(t);}', None),
    ('dependent_fetch','template<class T> __device__ T f(cudaTextureObject_t t){return tex2D<T>(t,.5f,.5f);}', None),
    ('function_address','auto pointer=&cudaDestroyTextureObject;', None),
    ('input_namespace','namespace user{void cudaDestroyTextureObject(int);void f(){cudaDestroyTextureObject(1);}}', None),
    ('input_redeclaration','extern "C" cudaError_t cudaDestroyTextureObject(cudaTextureObject_t);void f(cudaTextureObject_t t){auto s=cudaDestroyTextureObject(t);}', None),
    ('unexpanded_macro','#define MAKE cudaTextureDesc\nvoid f(){cudaTextureDesc d{};}', None),
    ('compat_macro_collision','#define TextureRegistry user_registry\nvoid f(){cudaTextureDesc d{};}', None),
    ('namespace_collision','namespace ascify{}\nvoid f(){cudaTextureDesc d{};}', None),
]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser()
    for name in ('binary','cuda-path','resource-dir','evidence-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--clang',type=Path)
    p.add_argument('--profile-host-math',type=Path)
    p.add_argument('--frontend-compat',choices=('none','ascify-admitted-v1'),default='none')
    a=p.parse_args()
    a.binary,a.cuda_path,a.resource_dir,a.evidence_dir=(q.resolve() for q in (a.binary,a.cuda_path,a.resource_dir,a.evidence_dir))
    choices=[a.clang] if a.clang else [a.binary.parent/'clang++',a.resource_dir.parents[2]/'bin/clang++',Path(shutil.which('clang++') or '/missing')]
    a.clang=next((q.resolve() for q in choices if q.is_file()),None)
    if a.clang is None:p.error('matching real clang++ required')
    version=subprocess.check_output([str(a.clang),'--version'],text=True)
    major=re.search(r'\bclang version (\d+)(?:\.|\s)',version)
    if major is None or major.group(1)!=a.resource_dir.name.split('.')[0]:p.error('clang/resource major mismatch')
    helpers=Path(__file__).resolve().parent/'fixtures/nvidia_samples/Common'
    helper_identity={str(q):sha(q) for q in sorted(helpers.iterdir()) if q.is_file()}
    a.evidence_dir.mkdir(parents=True,exist_ok=False)
    summary={'passed':False,'binary_sha256':sha(a.binary),'clang_sha256':sha(a.clang),
             'runner_sha256':sha(Path(__file__)),'helper_identity':helper_identity,'frontend_compatibility':a.frontend_compat,'cases':[],'device_executed':False}
    def save():(a.evidence_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    def run(directory,role,argv):
        argv=list(map(str,argv));(directory/(role+'.argv.json')).write_text(json.dumps(argv,indent=2)+'\n')
        z=subprocess.run(argv,capture_output=True,text=True,timeout=90)
        (directory/(role+'.stdout.log')).write_text(z.stdout);(directory/(role+'.stderr.log')).write_text(z.stderr)
        return z
    save()
    cases=[(name,code,expected,[],{}) for name,code,expected in CASES]
    recursive_cases={}
    if a.frontend_compat=='ascify-admitted-v1':
        installed_host_math=a.binary.parent.parent/'libexec/ascify/frontend-compat/ascify-admitted-v1/host_math.h'
        profile_host_math=(a.profile_host_math or (installed_host_math if installed_host_math.is_file() else
                          Path(__file__).resolve().parents[2]/'frontend_compat/ascify-admitted-v1/host_math.h')).resolve(strict=True)
        summary['profile_host_math']={'path':str(profile_host_math),'sha256':sha(profile_host_math)}
        cases += [
            ('profile_fake_host_math','#include <fake/host_math.h>\ncudaTextureDesc descriptor{};',None,['-I{case}'],{'fake/host_math.h':'template<class T>T caller_identity(T x){return x;}\n'}),
            ('profile_macro_template_parameter','cudaTextureDesc descriptor{};',None,['-DA=CallerType'],{}),
            ('profile_macro_template_owner','cudaTextureDesc descriptor{};',None,['-Dhost_float_max_enabled=CallerEnabled'],{}),
            ('profile_empty_macro_expansion','cudaTextureDesc descriptor{};',None,['-Dinline='],{}),
            ('profile_preinclude_macro_then_undef','#undef A\ncudaTextureDesc descriptor{};',None,['-include','{case}/caller_macros.h'],{'caller_macros.h':'#define A CallerType\n'}),
            ('profile_late_unused_macro','#define host_float_max_enabled CallerEnabled\n#undef host_float_max_enabled\ncudaTextureDesc descriptor{};',['ascify::cudaTextureDesc descriptor'],[],{}),
        ]
        for altered in (False,True):
            code=('#undef ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_HOST_MATH_H_\n'
                  'namespace repeated_profile {\n'+('#define A CallerType\n' if altered else '')+
                  '#include <host_math.h>\n'+('#undef A\n' if altered else '')+
                  '}\ncudaTextureDesc descriptor{};\n'
                  'float f(){return repeated_profile::max(1.0f,2.0f);}')
            for recursive in (False,True):
                name=('profile_reinclude_macro_parameter' if altered else 'profile_reinclude_namespace')+('_recursive' if recursive else '_plain')
                cases.append((name,code,None,['-I'+str(profile_host_math.parent)],{}))
                recursive_cases[name]=recursive
    for name,code,expected,extra,headers in cases:
        directory=a.evidence_dir/name;directory.mkdir()
        for filename,contents in headers.items():
            path=directory/filename;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(contents)
        extra=[value.replace('{case}',str(directory)) for value in extra]
        source=directory/'input.cu';output=directory/'output.cce';receipt=directory/'receipt.json'
        source.write_text('#include <cuda_runtime.h>\n#include <string.h>\n'+code+'\n')
        row={'name':name,'expected_fragments':expected,'passed':False,'source_sha256':sha(source),
             'extra_arguments':extra,'local_header_identity':{filename:sha(directory/filename) for filename in headers},
             'recursive':recursive_cases.get(name,a.frontend_compat=='ascify-admitted-v1')}
        summary['cases'].append(row);save()
        try:
            z=run(directory,'syntax',[a.clang,source,'-x','cuda','-std=c++17','--cuda-host-only','-fsyntax-only','-nocudalib','--cuda-path='+str(a.cuda_path),'-resource-dir='+str(a.resource_dir),'-Wno-unknown-cuda-version','-I'+str(helpers),*extra])
            row['syntax_returncode']=z.returncode;save();assert z.returncode==0,(name,z.stderr)
            projection=['--local-headers-recursive'] if row['recursive'] else []
            z=run(directory,'convert',[a.binary,source,'--target-policy=dav-c310-vec','--simt-math=fast','--target-recipe=none','--default-preprocessor','--frontend-compat='+a.frontend_compat,'--cuda-path='+str(a.cuda_path),'--clang-resource-directory='+str(a.resource_dir),'--migration-receipt='+str(receipt),'-o',output,*projection,'--','-x','cuda','-std=c++17','-I'+str(helpers),*extra])
            row['conversion_returncode']=z.returncode;save()
            if expected is None:
                output_headers=directory/'output.cce.headers'
                assert z.returncode>0 and not output.exists() and not output.is_symlink() and not output_headers.exists() and not output_headers.is_symlink() and json.loads(receipt.read_text())['status']=='failed',(name,z.stderr)
                assert any(marker in z.stderr for marker in ('CUDA texture boundary','CUDA runtime-API header boundary','Ascify cannot publish CUDA compatibility output')),(name,z.stderr)
                assert 'fatal error:' not in z.stderr,(name,z.stderr)
            else:
                assert z.returncode==0 and json.loads(receipt.read_text())['status']=='succeeded',(name,z.stderr)
                text=output.read_text();assert '#include <ascify/texture_compat.hpp>' in text,(name,text)
                assert all(fragment in text for fragment in expected),(name,expected,text)
                row['output_sha256']=sha(output)
            assert sha(source)==row['source_sha256']
            assert all(sha(directory/q)==digest for q,digest in row['local_header_identity'].items())
            row['passed']=True
        except (AssertionError, subprocess.TimeoutExpired) as exc:
            row['error']={'type':type(exc).__name__,'message':str(exc)}
        save();print(name,'passed' if row['passed'] else 'FAILED',flush=True)
    assert sha(a.binary)==summary['binary_sha256'] and sha(a.clang)==summary['clang_sha256']
    assert all(sha(Path(q))==digest for q,digest in helper_identity.items())
    if 'profile_host_math' in summary:
        assert sha(profile_host_math)==summary['profile_host_math']['sha256']
    summary.update(passed=all(row['passed'] for row in summary['cases']),total=len(cases));save()
    assert summary['passed'], 'texture boundary matrix contains failures'
if __name__=='__main__':main()
