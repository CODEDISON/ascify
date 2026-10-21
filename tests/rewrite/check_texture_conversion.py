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
    ('unsigned_fetch','__global__ void k(unsigned int*p,cudaTextureObject_t t){p[0]=tex2D<unsigned int>(t,.5f,.5f);}', ['ascify::tex2D<unsigned int>']),
    ('unsigned_layered_fetch','__global__ void k(unsigned int*p,cudaTextureObject_t t){p[0]=tex2DLayered<unsigned int>(t,.5f,.5f,1);}', ['ascify::tex2DLayered<unsigned int>']),
    ('unsigned_alias_fetch','using Pixel=unsigned int;__global__ void k(Pixel*p,cudaTextureObject_t t){p[0]=tex2D<Pixel>(t,.5f,.5f);}', ['ascify::tex2D<Pixel>']),
    ('unsigned_array_allocate','void f(){cudaArray*a=nullptr;auto c=cudaCreateChannelDesc<unsigned int>();auto s=cudaMallocArray(&a,&c,4,3);}', ['ascify::cudaMallocArray','ascify::cudaCreateChannelDesc<unsigned int>']),
    ('unsigned_layered_allocate','void f(){cudaArray*a=nullptr;auto c=cudaCreateChannelDesc<unsigned int>();auto s=cudaMalloc3DArray(&a,&c,make_cudaExtent(4,3,2),cudaArrayLayered);}', ['ascify::cudaCreateChannelDesc<unsigned int>','make_cudaExtent(4,3,2),0x01)']),
    ('unsigned_macro_fetch','#define FETCH_UINT(t) tex2D<unsigned int>(t,.5f,.5f)\n__global__ void k(unsigned int*p,cudaTextureObject_t t){p[0]=FETCH_UINT(t);}', None),
    ('unsigned_pointer_fetch','__global__ void k(unsigned int*p,cudaTextureObject_t t){tex2D<unsigned int>(p,t,.5f,.5f);}', None),
    ('unsupported_small_channel','void f(){auto d=cudaCreateChannelDesc<unsigned short>();}', None),
    ('unsigned_sampler_macro','#define textureSampleUnsigned fake_sampler\n__global__ void k(unsigned int*p,cudaTextureObject_t t){p[0]=tex2D<unsigned int>(t,.5f,.5f);}', None),
    ('unsigned_tag_macro','#define unsigned_element invalid_tag\n__global__ void k(unsigned int*p,cudaTextureObject_t t){p[0]=tex2D<unsigned int>(t,.5f,.5f);}', None),
    ('layered_fetch','__global__ void k(float*p,cudaTextureObject_t t){p[0]=tex2DLayered<float>(t,.5f,.5f,1);}', ['ascify::tex2DLayered<float>']),
    ('fractional_fetch','__global__ void k(float*p,cudaTextureObject_t t){p[0]=tex2D<float>(t,.501f,.333f);}', ['ascify::tex2D<float>']),
    ('fast_name_macro','#define tex2DFast user_fetch\n__global__ void k(float*p,cudaTextureObject_t t){p[0]=tex2D<float>(t,.5f,.5f);}', None),
    ('fast_layered_name_macro','#define tex2DLayeredFast user_fetch\n__global__ void k(float*p,cudaTextureObject_t t){p[0]=tex2DLayered<float>(t,.5f,.5f,0);}', None),
    ('fast_policy_macro','#define Approximate false\n__global__ void k(float*p,cudaTextureObject_t t){p[0]=tex2D<float>(t,.5f,.5f);}', None),
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
    p.add_argument('--simt-math',choices=('precise','fast'),default='fast')
    p.add_argument('--target-policy',choices=('portable','dav-c310-vec'),default='dav-c310-vec')
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
             'runner_sha256':sha(Path(__file__)),'helper_identity':helper_identity,'frontend_compatibility':a.frontend_compat,'simt_math':a.simt_math,'target_policy':a.target_policy,'cases':[],'device_executed':False}
    def save():(a.evidence_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    def run(directory,role,argv):
        argv=list(map(str,argv));(directory/(role+'.argv.json')).write_text(json.dumps(argv,indent=2)+'\n')
        z=subprocess.run(argv,capture_output=True,text=True,timeout=90)
        (directory/(role+'.stdout.log')).write_text(z.stdout);(directory/(role+'.stderr.log')).write_text(z.stderr)
        return z
    save()
    cases=[(name,code,expected,[],{}) for name,code,expected in CASES]
    recursive_cases={}
    cg_entry_counts={}
    if a.frontend_compat=='ascify-admitted-v1':
        installed_host_math=a.binary.parent.parent/'libexec/ascify/frontend-compat/ascify-admitted-v1/host_math.h'
        profile_host_math=(a.profile_host_math or (installed_host_math if installed_host_math.is_file() else
                          Path(__file__).resolve().parents[2]/'frontend_compat/ascify-admitted-v1/host_math.h')).resolve(strict=True)
        summary['profile_host_math']={'path':str(profile_host_math),'sha256':sha(profile_host_math)}
        profile_cg = profile_host_math.parent/'cooperative_groups.h'
        profile_reduce = profile_host_math.parent/'cooperative_groups/reduce.h'
        summary['profile_cg_inputs'] = {str(q):sha(q) for q in (profile_cg,profile_reduce)}
        cg_include = '#include <cooperative_groups.h>\n'
        cg_descriptor = 'cudaTextureDesc descriptor{};'
        # These tests isolate the parser profile boundary: the syntax stage
        # reads the exact same product CG header, with the genuine CUDA SDK
        # attribute macros. Other texture cases retain their official SDK path.
        cg_extra = ['-I'+str(profile_host_math.parent)]
        cg_cases = [
            ('profile_cg_unused',cg_include+cg_descriptor,['ascify::cudaTextureDesc descriptor'],{}),
            ('profile_cg_block_fetch',cg_include+'__global__ void k(float*p,cudaTextureObject_t t){auto b=cooperative_groups::this_thread_block();b.sync();p[0]=tex2D<float>(t,.5f,.5f);}', ['ascify::tex2D<float>'],{}),
            ('profile_cg_tile_fetch',cg_include+'__global__ void k(float*p,cudaTextureObject_t t){auto b=cooperative_groups::this_thread_block();auto g=cooperative_groups::tiled_partition<32>(b);p[0]=g.shfl_up(tex2D<float>(t,.5f,.5f),1);}', ['ascify::tex2D<float>'],{}),
            ('profile_cg_reduce_unused','#include <cooperative_groups/reduce.h>\n'+cg_descriptor,['ascify::cudaTextureDesc descriptor'],{}),
            ('profile_cg_late_unused_macro',cg_include+'#define host_float_max_enabled CallerEnabled\n#undef host_float_max_enabled\n'+cg_descriptor,['ascify::cudaTextureDesc descriptor'],{}),
            ('profile_cg_local_header','#include "local.cuh"\n'+cg_descriptor,['ascify::cudaTextureDesc descriptor'],{'local.cuh':cg_include+'__device__ inline void sync_block(){cooperative_groups::this_thread_block().sync();}\n'}),
            ('profile_cg_user_template',cg_include+'template<class T>T identity(T v){return v;}\n'+cg_descriptor,None,{}),
            ('profile_cg_concrete_sdk_reflection',cg_include+'template<class T>size_t bytes(T*p){return sizeof(T);} size_t f(cudaTextureDesc*p){return bytes(p);}',None,{}),
            ('profile_cg_layout_observation',cg_include+cg_descriptor+'\nsize_t f(){return sizeof(cudaTextureDesc);}',None,{}),
            ('profile_cg_full_runtime','#include <cuda_runtime_api.h>\n'+cg_include+cg_descriptor,['ascify::cudaTextureDesc descriptor'],{}),
            # Full Runtime admits only authenticated, uninstantiated product
            # patterns. Caller and concrete-instance observations stay audited.
            ('runtime_cg_plain','#include <cuda_runtime_api.h>\n'+cg_include+'int retained;',['int retained;'],{}),
            ('runtime_cg_caller_observation','#include <cuda_runtime_api.h>\n'+cg_include+'__annotate__(unused) int retained;',None,{}),
            ('runtime_cg_inactive_observation','#include <cuda_runtime_api.h>\n'+cg_include+'#if 0\n#if defined(__CUDACC__)\nint retained;\n#endif\n#endif',None,{}),
            ('runtime_cg_late_control_restore','#include <cuda_runtime_api.h>\n'+cg_include+'#undef __CUDA__\n#define __CUDA__ 1\nint retained;',None,{}),
            ('runtime_cg_late_pragma_restore','#include <cuda_runtime_api.h>\n'+cg_include+'#pragma push_macro("__CUDA__")\n#undef __CUDA__\n#pragma pop_macro("__CUDA__")\nint retained;',None,{}),
            ('runtime_cg_changed_qualifier','#include <cuda_runtime_api.h>\n#undef __annotate__\n#define __annotate__(a) __attribute__((a))\n'+cg_include+'int retained;',None,{}),
            ('runtime_cg_copy','#include <cuda_runtime_api.h>\n#include "cooperative_groups.h"\nint retained;',None,{'cooperative_groups.h':profile_cg.read_text()}),
            ('runtime_cg_nested_namespace','#include <cuda_runtime_api.h>\n#include <type_traits>\nnamespace caller {\n'+cg_include+'}\nint retained;',None,{}),
            ('profile_cg_macro_empty','#define inline\n'+cg_include+'#undef inline\n'+cg_descriptor,None,{}),
            ('profile_cg_macro_namespace','#define cooperative_groups caller_groups\n'+cg_include+'#undef cooperative_groups\n'+cg_descriptor,None,{}),
            ('profile_cg_macro_method','#define shfl_up caller_shuffle\n'+cg_include+'#undef shfl_up\n'+cg_descriptor,None,{}),
            ('profile_cg_macro_parameter','#define T CallerT\n'+cg_include+'#undef T\n'+cg_descriptor,None,{}),
            ('profile_cg_foreign_qualifier','#define ASCIFY_FRONTEND_COMPAT_DEVICE_ __device__\n'+cg_include+cg_descriptor,None,{}),
            ('profile_cg_redefine_device','#undef __device__\n#define __device__ __attribute__((device))\n'+cg_include+cg_descriptor,None,{}),
            ('profile_cg_redefine_location','#undef __location__\n#define __location__(a) __annotate__(a)\n'+cg_include+cg_descriptor,None,{}),
            ('profile_cg_redefine_annotate','#undef __annotate__\n#define __annotate__(a) __attribute__((a))\n'+cg_include+cg_descriptor,None,{}),
            ('profile_cg_redefine_cudacc','#undef __CUDACC__\n#define __CUDACC__ 1\n'+cg_include+cg_descriptor,None,{}),
            ('profile_cg_redefine_cuda','#undef __CUDA__\n#define __CUDA__ 1\n'+cg_include+cg_descriptor,None,{}),
            ('profile_cg_cuda_pop','#pragma push_macro("__CUDA__")\n#undef __CUDA__\n#pragma pop_macro("__CUDA__")\n'+cg_include+cg_descriptor,None,{}),
            ('profile_cg_namespace','#include <type_traits>\nnamespace caller {\n'+cg_include+'}\n'+cg_descriptor,None,{}),
            ('profile_cg_copy','#include "cooperative_groups.h"\n'+cg_descriptor,None,{'cooperative_groups.h':profile_cg.read_text()}),
            ('profile_cg_changed_copy','#include "cooperative_groups.h"\n'+cg_descriptor,None,{'cooperative_groups.h':profile_cg.read_text().replace('shfl_up(', 'caller_shuffle(')}),
        ]
        # A clean second include is skipped by the immutable product guard.
        # Real reentry requires changing that guard and is deliberately refused,
        # even when the source restores CUDA control spelling before reentry.
        cg_reentry = (cg_include+
                      '#undef ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_COOPERATIVE_GROUPS_H_\n')
        cg_reentry_tail = 'namespace repeated_cg {\n'+cg_include+'}\n'
        cg_cases += [
            ('profile_cg_repeat_guarded',cg_include+cg_include+cg_descriptor,['ascify::cudaTextureDesc descriptor'],{}),
            ('runtime_cg_repeat_guarded','#include <cuda_runtime_api.h>\n'+cg_include+cg_include+'int retained;',['int retained;'],{}),
            ('profile_cg_reentry_guard',cg_reentry+cg_reentry_tail+cg_descriptor,None,{}),
            ('runtime_cg_reentry_guard','#include <cuda_runtime_api.h>\n'+cg_reentry+cg_reentry_tail+'int retained;',None,{}),
            ('runtime_cg_reentry_control_restore','#include <cuda_runtime_api.h>\n'+cg_reentry+'#undef __CUDA__\n#define __CUDA__ 1\n'+cg_reentry_tail+'int retained;',None,{}),
        ]
        cg_entry_counts = {'profile_cg_repeat_guarded':1,'runtime_cg_repeat_guarded':1,
                           'profile_cg_reentry_guard':2,'runtime_cg_reentry_guard':2,
                           'runtime_cg_reentry_control_restore':2}
        for name,code,expected,headers in cg_cases:
            # Root staging uses a temporary file. Preserve explicit quote
            # lookup so these two fixtures actually select the caller shadow.
            extra = cg_extra + (['-iquote{case}'] if name in
                                ('profile_cg_copy','profile_cg_changed_copy','runtime_cg_copy') else [])
            cases.append((name,code,expected,extra,headers))
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
        if expected is not None and a.simt_math=='fast' and a.target_policy=='dav-c310-vec':
            expected=[fragment.replace('ascify::tex2D<','ascify::tex2DFast<').replace('ascify::tex2DLayered<','ascify::tex2DLayeredFast<') for fragment in expected]
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
            z=run(directory,'syntax',[a.clang,source,'-x','cuda','-std=c++17','--cuda-host-only','-fsyntax-only','-nocudalib','--cuda-path='+str(a.cuda_path),'-resource-dir='+str(a.resource_dir),'-Wno-unknown-cuda-version','-I'+str(helpers),*extra,*(['-H'] if name in cg_entry_counts else [])])
            row['syntax_returncode']=z.returncode;save();assert z.returncode==0,(name,z.stderr)
            if name in cg_entry_counts:
                # -H reports entered headers, not textual includes skipped by
                # their guards. Match the exact authenticated physical path.
                entries=[match.group(1) for line in z.stderr.splitlines()
                         if (match:=re.fullmatch(r'\.+ (.+)',line))]
                row['syntax_cg_entry_count']=entries.count(str(profile_cg))
                row['expected_syntax_cg_entry_count']=cg_entry_counts[name]
                save()
                assert row['syntax_cg_entry_count']==cg_entry_counts[name],(name,z.stderr)
            projection=['--local-headers-recursive'] if row['recursive'] else []
            z=run(directory,'convert',[a.binary,source,'--target-policy='+a.target_policy,'--simt-math='+a.simt_math,'--target-recipe=none','--default-preprocessor','--frontend-compat='+a.frontend_compat,'--cuda-path='+str(a.cuda_path),'--clang-resource-directory='+str(a.resource_dir),'--migration-receipt='+str(receipt),'-o',output,*projection,'--','-x','cuda','-std=c++17','-I'+str(helpers),*extra])
            row['conversion_returncode']=z.returncode;save()
            if expected is None:
                output_headers=directory/'output.cce.headers'
                assert z.returncode>0 and not output.exists() and not output.is_symlink() and not output_headers.exists() and not output_headers.is_symlink() and json.loads(receipt.read_text())['status']=='failed',(name,z.stderr)
                assert any(marker in z.stderr for marker in ('CUDA texture boundary','CUDA runtime-API header boundary','Ascify cannot publish CUDA compatibility output','frontend compatibility')),(name,z.stderr)
                assert 'fatal error:' not in z.stderr,(name,z.stderr)
                if name in ('runtime_cg_reentry_guard','runtime_cg_reentry_control_restore'):
                    assert 'unproved product CG macro provenance' in z.stderr,(name,z.stderr)
                if name=='runtime_cg_caller_observation':
                    assert "observes removed SDK macro '__annotate__'" in z.stderr,(name,z.stderr)
                if name=='runtime_cg_inactive_observation':
                    assert "observes CUDA compiler condition '__CUDACC__'" in z.stderr or "observes removed SDK macro '__CUDACC__'" in z.stderr,(name,z.stderr)
            else:
                assert z.returncode==0 and json.loads(receipt.read_text())['status']=='succeeded',(name,z.stderr)
                text=output.read_text()
                required_header = ('#include <ascify/ascify_cuda_compat.hpp>' if name in
                                   ('runtime_cg_plain','runtime_cg_repeat_guarded') else
                                   '#include <ascify/texture_compat.hpp>')
                assert required_header in text,(name,text)
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
        assert all(sha(Path(q))==digest for q,digest in summary['profile_cg_inputs'].items())
    summary.update(passed=all(row['passed'] for row in summary['cases']),total=len(cases));save()
    assert summary['passed'], 'texture boundary matrix contains failures'
if __name__=='__main__':main()
