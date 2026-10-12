#!/usr/bin/env python3
"""Actual CUDA AST max-only attribute proof; no device or fake runtime oracle."""
import argparse,hashlib,json,re,subprocess,tempfile
from pathlib import Path

K='__global__ void kernel(unsigned *p) { p[threadIdx.x]=threadIdx.x; }\n'
QUERY='cudaFuncGetAttributes(&a, kernel)'
BODY='int host(){cudaFuncAttributes a; auto status='+QUERY+'; return a.maxThreadsPerBlock + int(status);}\n'
CASES=[
 ('default',K+BODY,1024),
 ('zero_initializer',K+BODY.replace('cudaFuncAttributes a;','cudaFuncAttributes a{};'),1024),
 ('empty_copy_list_initializer',K+BODY.replace('cudaFuncAttributes a;','cudaFuncAttributes a={};'),1024),
 ('macro_empty_initializer',K+'#define EMPTY {}\n'+BODY.replace('cudaFuncAttributes a;','cudaFuncAttributes a EMPTY;'),None),
 ('macro_empty_initializer_element',K+'#define EMPTY\n'+BODY.replace('cudaFuncAttributes a;','cudaFuncAttributes a{EMPTY};'),None),
 ('nonempty_initializer',K+BODY.replace('cudaFuncAttributes a;','cudaFuncAttributes a={0,7};'),None),
 ('nonempty_initializer_before_query',K+BODY.replace('cudaFuncAttributes a;','cudaFuncAttributes a={0,7}; int before=a.maxThreadsPerBlock;').replace('return a.maxThreadsPerBlock','return before+a.maxThreadsPerBlock'),None),
 ('zero_element_initializer',K+BODY.replace('cudaFuncAttributes a;','cudaFuncAttributes a{0};'),None),
 ('copy_initializer',K+BODY.replace('cudaFuncAttributes a;','cudaFuncAttributes seed{}; cudaFuncAttributes a=seed;'),None),
 ('temporary_initializer',K+BODY.replace('cudaFuncAttributes a;','cudaFuncAttributes a=cudaFuncAttributes();'),None),
 ('offsetof_only','int host(){return __builtin_offsetof(cudaFuncAttributes,maxThreadsPerBlock);}',None),
 ('offsetof_with_query',K+BODY.replace('return a.maxThreadsPerBlock','return __builtin_offsetof(cudaFuncAttributes,maxThreadsPerBlock)+a.maxThreadsPerBlock'),None),
 ('offsetof_alias','using Attr=cudaFuncAttributes;\nint host(){return __builtin_offsetof(Attr,maxThreadsPerBlock);}',None),
 ('offsetof_nested','struct Nested{cudaFuncAttributes attr;};\nint host(){return __builtin_offsetof(Nested,attr.maxThreadsPerBlock);}',None),
 ('offsetof_instantiation','template<class T> int offset(){return __builtin_offsetof(T,maxThreadsPerBlock);}\nint host(){return offset<cudaFuncAttributes>();}',None),
 ('offsetof_retained_header','#include <offset_observer.hpp>\n'+K+BODY,None),
 ('offsetof_unrelated_field','struct User{int maxThreadsPerBlock;};\n'+K+BODY.replace('return a.maxThreadsPerBlock','return __builtin_offsetof(User,maxThreadsPerBlock)+a.maxThreadsPerBlock'),1024),
 ('explicit128',K.replace('__global__','__global__ __launch_bounds__(128)')+BODY,128),
 ('explicit256',K.replace('__global__','__global__ __launch_bounds__(256)')+BODY,256),
 ('macro_argument',K+'#define CHECK(x) (x)\n'+BODY.replace(QUERY,'CHECK('+QUERY+')'),1024),
 ('kernel_address',K+BODY.replace('&a, kernel','&a, &kernel'),1024),
 ('namespace_kernel','namespace app { '+K+'}\n'+BODY.replace('&a, kernel','&a, app::kernel'),None),
 ('qualified_kernel_address',K+BODY.replace('&a, kernel','&a, (&::kernel)'),1024),
 ('namespace_alias_kernel','namespace app { '+K+'} namespace alias=app;\n'+BODY.replace('&a, kernel','&a, alias::kernel'),None),
 ('using_namespace_kernel','namespace app { '+K+'} using namespace app;\n'+BODY,None),
 ('macro_kernel_address',K+'#define ADDRESS &\n'+BODY.replace('&a, kernel','&a, ADDRESS kernel'),None),
 ('macro_kernel_parens',K+'#define LP (\n#define RP )\n'+BODY.replace('&a, kernel','&a, LP kernel RP'),None),
 ('macro_leading_kernel_token',K+'#define EMPTY\n'+BODY.replace('&a, kernel','&a, EMPTY kernel'),None),
 ('macro_trailing_kernel_token',K+'#define EMPTY\n'+BODY.replace('&a, kernel','&a, kernel EMPTY'),None),
 ('conditional_kernel_definition','#ifdef __CCE_AICORE__\n'+K.replace('__global__','__global__ __launch_bounds__(256)')+'#else\n'+K+'#endif\n'+BODY,None),
 ('conditional_kernel_redeclaration','#ifdef __CCE_AICORE__\n__global__ __launch_bounds__(256) void kernel(unsigned *p);\n#endif\n'+K+BODY,None),
 ('macro_empty_kernel_signature','#define BOUND\n'+K.replace('__global__','__global__ BOUND')+BODY,None),
 ('macro_empty_kernel_signature_redecl','#define BOUND\n__global__ BOUND void kernel(unsigned *p);\n'+K+BODY,None),
 ('macro_leading_kernel_signature','#define PREFIX\nPREFIX\n'+K+BODY,None),
 ('directive_kernel_signature',K.replace('__global__','__global__\n#define EMPTY\nEMPTY')+BODY,None),
 ('conditional_output_declaration','#include "kernel.cuh"\n'+BODY.replace('cudaFuncAttributes a;','\n#ifndef __CCE_AICORE__\ncudaFuncAttributes a{};\n#else\ncudaFuncAttributes a{0,7};\n#endif\n'),None),
 ('conditional_query','#include "kernel.cuh"\n'+BODY.replace('auto status='+QUERY+';','auto status=cudaSuccess;\n#ifndef __CCE_AICORE__\nstatus='+QUERY+';\n#endif\n'),None),
 ('macro_trailing_output',K+'#define EMPTY\n'+BODY.replace('cudaFuncAttributes a;','cudaFuncAttributes a EMPTY;'),None),
 ('macro_leading_output',K+'#define EMPTY\n'+BODY.replace('cudaFuncAttributes a;','EMPTY cudaFuncAttributes a;'),None),
 ('output_type_alias','using Attr=cudaFuncAttributes;\n'+K+BODY.replace('cudaFuncAttributes a;','Attr a;'),None),
 ('output_keyword_prefix_alias','using structcudaFuncAttributes=cudaFuncAttributes;\n'+K+BODY.replace('cudaFuncAttributes a;','structcudaFuncAttributes a;'),None),
 ('elaborated_output_type',K+BODY.replace('cudaFuncAttributes a;','struct cudaFuncAttributes a;'),1024),
 ('conditional_header_kernel','#include "conditional_kernel.cuh"\n'+BODY,None),
 ('macro_namespace_kernel','namespace app { '+K+'}\n#define NS app\n'+BODY.replace('&a, kernel','&a, NS::kernel'),None),
 ('macro_middle_namespace_kernel','namespace outer { namespace app { '+K+'}}\n#define NS app\n'+BODY.replace('&a, kernel','&a, outer::NS::kernel'),None),
 ('macro_empty_namespace_kernel','namespace app { '+K+'}\n#define EMPTY\n'+BODY.replace('&a, kernel','&a, app::EMPTY kernel'),None),
 ('macro_namespace_in_argument','namespace app { '+K+'}\n#define NS app\n#define CHECK(x) (x)\n'+BODY.replace(QUERY,'CHECK('+QUERY.replace('&a, kernel','&a, NS::kernel')+')'),None),
 ('read_twice',K+BODY.replace(' + int(status)',' + a.maxThreadsPerBlock + int(status)'),1024),
 ('parenthesized_read',K+BODY.replace('a.maxThreadsPerBlock','((a.maxThreadsPerBlock))'),1024),
 ('min_lvalue_read',K+'#define MIN(a,b) ((a)<(b)?(a):(b))\n'+BODY.replace('auto status=', 'int other=512; auto status=').replace('return a.maxThreadsPerBlock','return MIN(other,a.maxThreadsPerBlock)'),1024),
 ('conditional_read',K+BODY.replace('auto status=', 'int other=512; auto status=').replace('return a.maxThreadsPerBlock','return (other ? a.maxThreadsPerBlock : other)'),1024),
 ('conditional_reference_escape',K+BODY.replace('auto status=', 'int other=512; auto status=').replace('return a.maxThreadsPerBlock','int &alias=(other ? a.maxThreadsPerBlock : other); return a.maxThreadsPerBlock'),None),
 ('conditional_address_escape',K+BODY.replace('auto status=', 'int other=512; auto status=').replace('return a.maxThreadsPerBlock','int *alias=&(other ? a.maxThreadsPerBlock : other); return a.maxThreadsPerBlock'),None),
 ('conditional_write',K+BODY.replace('auto status=', 'int other=512; auto status=').replace('return a.maxThreadsPerBlock','(other ? a.maxThreadsPerBlock : other)=7; return a.maxThreadsPerBlock'),None),
 ('header_kernel','#include "kernel.cuh"\n'+BODY,1024),
 ('unrelated_completed_condition','#include "kernel.cuh"\n#if defined(__CCE_AICORE__)\nstatic int irrelevant=1;\n#else\nstatic int irrelevant=0;\n#endif\n'+BODY,1024),
 ('indirect_kernel',K+BODY.replace('auto status=', 'auto k=kernel; auto status=').replace('&a, kernel','&a, k'),None),
 ('extern_only','__global__ void kernel(unsigned *p);\n'+BODY,None),
 ('forward_declared_kernel','__global__ void kernel(unsigned *p);\n'+K+BODY,None),
 ('template_kernel',K.replace('__global__','template<class T> __global__')+BODY.replace('&a, kernel','&a, kernel<unsigned>'),None),
 ('shared_read',K+BODY.replace('a.maxThreadsPerBlock','a.maxThreadsPerBlock + int(a.sharedSizeBytes)'),None),
 ('whole_copy',K+BODY.replace('return a.maxThreadsPerBlock','cudaFuncAttributes b=a; return a.maxThreadsPerBlock'),None),
 ('escape_address',K+'void escape(cudaFuncAttributes*);\n'+BODY.replace('return a.maxThreadsPerBlock','escape(&a); return a.maxThreadsPerBlock'),None),
 ('reference_alias',K+BODY.replace('return a.maxThreadsPerBlock','auto &other=a; return a.maxThreadsPerBlock'),None),
 ('lambda_capture',K+BODY.replace('return a.maxThreadsPerBlock','auto f=[a]{return a.maxThreadsPerBlock;}; return a.maxThreadsPerBlock'),None),
 ('field_write',K+BODY.replace('return a.maxThreadsPerBlock','a.maxThreadsPerBlock=3; return a.maxThreadsPerBlock'),None),
 ('two_queries',K+BODY.replace('return a.maxThreadsPerBlock',QUERY+'; return a.maxThreadsPerBlock'),None),
 ('static_output',K+BODY.replace('cudaFuncAttributes a;','static cudaFuncAttributes a;'),None),
 ('parameter_output',K+BODY.replace('int host(){cudaFuncAttributes a;','int host(cudaFuncAttributes &a){'),None),
 ('global_output',K+'cudaFuncAttributes a;\n'+BODY.replace('cudaFuncAttributes a;',''),None),
 ('macro_callee',K+'#define QUERY(a,k) cudaFuncGetAttributes(a,k)\n'+BODY.replace(QUERY,'QUERY(&a,kernel)'),None),
 ('macro_kernel','#define GLOBAL_KERNEL '+K+'\nGLOBAL_KERNEL\n'+BODY,None),
 ('macro_kernel_reference',K+'#define PICK kernel\n'+BODY.replace('&a, kernel','&a, PICK'),None),
 ('macro_kernel_reference_in_argument',K+'#define PICK kernel\n#define CHECK(x) (x)\n'+BODY.replace(QUERY,'CHECK('+QUERY.replace('&a, kernel','&a, PICK')+')'),None),
 ('expression_bound','constexpr int bound=128;\n'+K.replace('__global__','__global__ __launch_bounds__(bound)')+BODY,None),
 ('residency_bound',K.replace('__global__','__global__ __launch_bounds__(128,2)')+BODY,None),
 ('no_query',K+BODY.replace('auto status='+QUERY+';', 'auto status=0;'),None),
 ('retained_header_observation','#include <observer.hpp>\n'+K+BODY,None),
 ('nested_header_observation','#include "outer.hpp"\n'+K+BODY,None),
 ('system_header_observation','#include <observer.hpp>\n'+K+BODY,None),
 ('forced_header_observation',K+BODY,None),
 ('member_pointer',K+'int host(){cudaFuncAttributes a; cudaFuncGetAttributes(&a,kernel); auto field=&cudaFuncAttributes::maxThreadsPerBlock; return a.*field;}\n',None),
 ('static_query_initializer',K+BODY.replace('auto status=', 'static auto status='),None),
 ('lambda_query_initializer',K+BODY.replace(QUERY,'[&]{return '+QUERY+';}()'),None),
 ('legacy_shared_only',K+BODY.replace('a.maxThreadsPerBlock','int(a.sharedSizeBytes)'),0),
]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--binary',required=True,type=Path);p.add_argument('--cuda-path',required=True,type=Path);p.add_argument('--resource-dir',required=True,type=Path);p.add_argument('--clang',type=Path);p.add_argument('--evidence-dir',type=Path);a=p.parse_args();temp=tempfile.TemporaryDirectory(prefix='ascify-kernel-attributes-') if a.evidence_dir is None else None;out=a.evidence_dir.resolve() if a.evidence_dir else Path(temp.name)/'evidence';out.mkdir(parents=True,exist_ok=False)
 summary={'passed':False,'binary_sha256':sha(a.binary),'cases':[],'scope':'CUDA source validity + semantic conversion; no target execution'}
 def save():(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 def run(d,role,argv):
  (d/(role+'.argv.json')).write_text(json.dumps(list(map(str,argv)),indent=2)+'\n');z=subprocess.run(list(map(str,argv)),capture_output=True,text=True,timeout=60);(d/(role+'.stdout.log')).write_text(z.stdout);(d/(role+'.stderr.log')).write_text(z.stderr);return z
 save()
 for name,code,expected in CASES:
  d=out/name;d.mkdir();src=d/'input.cu';dst=d/'output.cpp';receipt=d/'receipt.json';src.write_text('#include <cuda_runtime.h>\n'+code);(d/'kernel.cuh').write_text(K);(d/'conditional_kernel.cuh').write_text('#if 1\n'+K+'#endif\n');(d/'observer.hpp').write_text('#include <cuda_runtime.h>\ninline int observed(cudaFuncAttributes &a){return a.maxThreadsPerBlock;}\n');(d/'outer.hpp').write_text('#include "observer.hpp"\n');(d/'offset_observer.hpp').write_text('#include <cuda_runtime.h>\ninline int observed_offset(){return __builtin_offsetof(cudaFuncAttributes,maxThreadsPerBlock); }\n')
  row={'name':name,'expected_bound':expected,'passed':False,'source_sha256':sha(src)};summary['cases'].append(row);save()
  flags=['-x','cuda','-std=c++17','-I'+str(d)]
  if name=='system_header_observation':flags += ['-isystem',str(d)]
  if name=='forced_header_observation':flags += ['-include',str(d/'observer.hpp')]
  if a.clang:
   z=run(d,'syntax',[a.clang,src,*flags,'-fsyntax-only','--cuda-host-only','-nocudalib','--cuda-path='+str(a.cuda_path),'-resource-dir='+str(a.resource_dir),'-Wno-unknown-cuda-version']);row['syntax_returncode']=z.returncode;save();assert z.returncode==0,(name,z.stderr)
  z=run(d,'convert',[a.binary,src,'--target-policy=dav-c310-vec','--simt-math=fast','--target-recipe=none','--default-preprocessor','--local-headers-recursive','--cuda-path='+str(a.cuda_path),'--clang-resource-directory='+str(a.resource_dir),'--migration-receipt='+str(receipt),'-o',dst,'--',*flags]);row['conversion_returncode']=z.returncode;save()
  if expected is None:
   assert z.returncode>0 and not dst.exists() and json.loads(receipt.read_text())['status']=='failed' and 'unproved per-kernel maxThreadsPerBlock' in z.stderr,(name,z.stderr)
  else:
   assert z.returncode==0 and json.loads(receipt.read_text())['status']=='succeeded',(name,z.stderr)
   actual=re.findall(r'::ascify::cudaFuncGetAttributesMaxThreads<(\d+)>',dst.read_text());assert actual==([str(expected)] if expected else []),(name,actual,expected,dst.read_text())
  assert sha(src)==row['source_sha256'];row['passed']=True;save();print(name,'passed',flush=True)
 assert sha(a.binary)==summary['binary_sha256'];summary.update(passed=True,total=len(CASES));save()
 if temp:temp.cleanup()
if __name__=='__main__':main()
