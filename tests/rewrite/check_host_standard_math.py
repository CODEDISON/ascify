#!/usr/bin/env python3
"""Source-valid host standard-math preservation and device-map boundary matrix."""
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
temporary=tempfile.TemporaryDirectory(prefix='ascify-host-standard-math-')
a.output=a.evidence_dir.resolve() if a.evidence_dir else Path(temporary.name)/'evidence'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
prefix='#include <cuda_runtime.h>\n#include <math.h>\n'
cases=[]
def case(name,body,expected,absent=()):
 cases.append((name,prefix+body+'\n',list(expected),list(absent),False))
for name in ('exp','log','sqrt'):
 case('host_double_'+name,'double host(double x){return '+name+'(x);}', ['return '+name+'(x);'], ['return '+name+'f(x);'])
 case('host_float_'+name,'float host(float x){return '+name+'(x);}', ['return '+name+'(x);'], ['return '+name+'f(x);'])
case('oracle_double_literal','float gold(float d){return (float)exp(-0.5*d*d);}', ['(float)exp(-0.5*d*d)'], ['expf('])
case('oracle_double_initializer','double gold(double delta,double sum){double L2norm=sqrt(delta/sum);return L2norm;}', ['double L2norm=sqrt(delta/sum);'], ['sqrtf('])
case('nested_initializer','double host(double x){double a=exp(log(sqrt(x)));return a;}', ['a=exp(log(sqrt(x)))'], ['expf(','logf(','sqrtf('])
case('host_explicit','__host__ double host(double x){return sqrt(x);}', ['return sqrt(x);'], ['sqrtf('])
case('argument_side_effect','double host(double x){double a=exp(x++);double b=sqrt(++x);return a+b;}', ['double a=exp(x++);double b=sqrt(++x);return a+b;'], ['expf(','sqrtf('])
case('operator_macro','#define ADD +\ndouble host(double x){return sqrt(x ADD 1.0);}', ['sqrt(x ADD 1.0)'], ['sqrtf('])
case('mixed_host_device','double host(double x){return sqrt(x); }\n__device__ float dev(float x){return sqrt(x)+exp(x)+log(x)+rsqrt(x);}', ['double host(double x){return sqrt(x); }','return sqrtf(x)+expf(x)+logf(x)+rsqrtf(x);'])
case('device_global','__global__ void kernel(float *p){p[0]=sqrt(p[0])+exp(p[0])+log(p[0]);}', ['sqrtf(p[0])+expf(p[0])+logf(p[0])'])
case('dual_context','__host__ __device__ float both(float x){return sqrt(x);}', ['return sqrtf(x);'])
case('qualified','#include <cmath>\ndouble host(double x){return std::sqrt(x)+::exp(x);}', ['std::sqrt(x)+::exp(x)'], ['std::sqrtf','::expf'])
# These cases pin the existing mapping outside the bounded host-only proof;
# their target executability is not claimed by this test.
case('macro_callee','#define ROOT sqrt\ndouble host(double x){return ROOT(x);}', ['#define ROOT sqrtf','return ROOT(x);'])
case('macro_shared_body','#define ROOT(x) sqrt(x)\ndouble host(double x){return ROOT(x); }\n__device__ float dev(float x){return ROOT(x);}', ['#define ROOT(x) sqrtf(x)'])
case('host_template','template<class T> T host(T x){return sqrt(x);}', ['return sqrtf(x);'])
case('lambda_boundary','double host(double x){auto f=[](double y){return sqrt(y);};return f(x);}', ['return sqrtf(y);'])
case('static_initializer','double host(double x){static double y=sqrt(x);return y;}', ['static double y=sqrtf(x);'])
case('user_overload','long sqrt(long x){return x;}\nlong host(long x){return sqrt(x);}', ['long sqrtf(long x)','return sqrtf(x);'])
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
  common=['-x','cuda','-std=c++17','--cuda-path='+str(cuda),'-resource-dir='+str(resource)]
  r=run(d,'syntax',[clang,cu,*common,'--cuda-host-only','-nocudalib','-fsyntax-only','-Wno-unknown-cuda-version']);row['syntax_rc']=r.returncode;save();assert r.returncode==0,(name,r.stderr)
  if not a.source_only:
   r=run(d,'conversion',[a.binary,cu,'--target-policy=dav-c310-vec','--simt-math=fast','--target-recipe=none','--default-preprocessor','--cuda-path='+str(cuda),'--clang-resource-directory='+str(resource),'--migration-receipt='+str(receipt),'-o',out,'--',*common]);row['conversion_rc']=r.returncode
   if refused:
    assert r.returncode>0 and not out.exists(),(name,r.stderr)
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
