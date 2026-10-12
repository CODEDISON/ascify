#!/usr/bin/env python3
"""Real CUDA helper atomicAdd ownership, type and expression boundaries (no NPU)."""
import argparse,hashlib,json,re,shutil,subprocess,tempfile
from pathlib import Path

CASES = [
 ('signed_parameter','__device__ int f(int*p){return atomicAdd(p,1);}',1),
 ('unsigned_parameter','__device__ unsigned f(unsigned*p){return atomicAdd(p,1);}',1),
 ('subscript','__device__ unsigned f(unsigned*p,int i){return atomicAdd(&p[i],1U);}',1),
 ('offset','__device__ int f(int*p,int i){return atomicAdd(p+i,1);}',1),
 ('nested_helpers','__device__ int f(int*p){return atomicAdd(p,1);}\n__device__ int g(int*p){return f(p);}',1),
 ('float','__device__ float f(float*p){return atomicAdd(p,1.f);}',0),
 ('double','__device__ double f(double*p){return atomicAdd(p,1.);}',0),
 ('wide_integer','__device__ unsigned long long f(unsigned long long*p){return atomicAdd(p,1ULL);}',0),
 ('other_family','__device__ int f(int*p){return atomicSub(p,1);}',0),
 ('local_alias','__device__ int f(int*p){int*q=p;return atomicAdd(q,1);}',0),
 ('local_storage','__device__ int f(int*p){int x=0;return atomicAdd(&x,1);}',0),
 ('shared_storage','__device__ int f(int*p){__shared__ int x;return atomicAdd(&x,1);}',0),
 ('template','template<class T> __device__ int f(T*p){return atomicAdd(p,1);}\n__device__ int g(int*p){return f(p);}',0),
 ('qualified_call','__device__ int f(int*p){return ::atomicAdd(p,1);}',0),
 ('macro_call','#define ADD(p) atomicAdd(p,1)\n__device__ int f(int*p){return ADD(p);}',0),
 ('project_redeclaration','__device__ int atomicAdd(int*,int);\n__device__ int f(int*p){return atomicAdd(p,1);}',0),
 ('project_namespace','namespace n{__device__ int atomicAdd(int*,int);\n__device__ int f(int*p){return atomicAdd(p,1);}}',0),
 ('kernel_keeps_global_mapping','__global__ void f(int*p){atomicAdd(p,1);}',0),
 ('macro_collision','#define atomic_add_generic user_macro\n__device__ int f(int*p){return atomicAdd(p,1);}',None),
 ('namespace_collision','namespace ascify{}\n__device__ int f(int*p){return atomicAdd(p,1);}',None),
]

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--binary',required=True,type=Path);p.add_argument('--cuda-path',required=True,type=Path);p.add_argument('--resource-dir',required=True,type=Path);p.add_argument('--clang',type=Path);p.add_argument('--evidence-dir',required=True,type=Path);a=p.parse_args()
 a.binary=a.binary.resolve();a.cuda_path=a.cuda_path.resolve();a.resource_dir=a.resource_dir.resolve()
 choices=[a.clang] if a.clang else [a.binary.parent/'clang++',a.resource_dir.parents[2]/'bin/clang++',Path(shutil.which('clang++') or '/missing')]
 a.clang=next((x.resolve() for x in choices if x.is_file()),None)
 if a.clang is None or not a.binary.is_file():p.error('supply a real binary and matching --clang')
 version=subprocess.check_output([str(a.clang),'--version'],text=True)
 major=re.search(r'\bclang version (\d+)(?:\.|\s)',version)
 if major is None or major.group(1)!=a.resource_dir.name.split('.')[0]:p.error('clang version must match resource directory')
 out=a.evidence_dir.resolve();out.mkdir(parents=True,exist_ok=False)
 summary={'passed':False,'binary_sha256':sha(a.binary),'clang_sha256':sha(a.clang),'cases':[],'scope':'Real CUDA source validation and converter output; no target execution'}
 def save(): (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 def run(d,role,argv):
  (d/(role+'.argv.json')).write_text(json.dumps(list(map(str,argv)),indent=2)+'\n')
  z=subprocess.run(list(map(str,argv)),capture_output=True,text=True,timeout=60)
  (d/(role+'.stdout.log')).write_text(z.stdout);(d/(role+'.stderr.log')).write_text(z.stderr);return z
 save()
 for name,code,expected in CASES:
  d=out/name;d.mkdir();src=d/'input.cu';dst=d/'output.cpp';receipt=d/'receipt.json';src.write_text('#include <cuda_runtime.h>\n'+code+'\n')
  row={'name':name,'expected_rewrites':expected,'passed':False,'source_sha256':sha(src)};summary['cases'].append(row)
  z=run(d,'syntax',[a.clang,src,'-x','cuda','-std=c++17','-fsyntax-only','--cuda-host-only','-nocudalib','--cuda-path='+str(a.cuda_path),'-resource-dir='+str(a.resource_dir),'-Wno-unknown-cuda-version']);row['syntax_returncode']=z.returncode;save();assert z.returncode==0,(name,z.stderr)
  z=run(d,'convert',[a.binary,src,'--target-policy=dav-c310-vec','--simt-math=fast','--target-recipe=none','--default-preprocessor','--cuda-path='+str(a.cuda_path),'--clang-resource-directory='+str(a.resource_dir),'--migration-receipt='+str(receipt),'-o',dst,'--','-x','cuda','-std=c++17']);row['conversion_returncode']=z.returncode;save()
  if expected is None:
   assert z.returncode>0 and json.loads(receipt.read_text())['status']=='failed' and not dst.exists() and 'Ascify cannot publish CUDA compatibility output' in z.stderr,(name,z.stderr)
  else:
   assert z.returncode==0 and json.loads(receipt.read_text())['status']=='succeeded',(name,z.stderr)
   text=dst.read_text();actual=text.count('::ascify::atomic_add_generic(')
   assert actual==expected,(name,actual,expected,text)
   if expected == 0:
    if name == 'kernel_keeps_global_mapping':
     assert 'ascify::atomic_add_global(p,1)' in text,(name,text)
    else:
     for operation in ('atomicAdd(', 'atomicSub('):
      assert text.count(operation)==code.count(operation),(name,operation,text)
   row['output_sha256']=sha(dst)
  assert sha(src)==row['source_sha256'];row['passed']=True;save();print(name,'passed',flush=True)
 assert sha(a.binary)==summary['binary_sha256'] and sha(a.clang)==summary['clang_sha256']
 summary.update(passed=True,total=len(CASES));save()
if __name__=='__main__':main()
