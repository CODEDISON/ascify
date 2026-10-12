#!/usr/bin/env python3
"""Real CUDA parser and generated same-TU symbol resolver boundaries (no NPU)."""
import argparse,hashlib,json,re,shutil,subprocess,tempfile
from pathlib import Path

CASES = [
 ('output_name','__constant__ int output[8];',['output']),
 ('namespace_shadow','namespace n { constexpr int ACL_SUCCESS = 31; constexpr int ACL_MEMCPY_DEVICE_TO_HOST = 33; __constant__ int output[8]; }',['n::output']),
 ('address_overload','struct Entry { int value; __host__ __device__ Entry *operator&() { return nullptr; } }; __constant__ Entry scalar;',['scalar']),
 ('anonymous_namespace','namespace { __constant__ int values[8]; }',[]),
 ('array','__constant__ float values[8];',['values']),
 ('initialized','__constant__ int values[3] = {1,2,3};',['values']),
 ('scalar','__device__ unsigned value;',['value']),
 ('static_table','static __constant__ unsigned table[3][31];',['table']),
 ('namespace','namespace example { __constant__ int values[8]; }',['example::values']),
 ('struct','struct Entry { float x; unsigned y; }; __constant__ Entry table[8];',['table']),
 ('pointer_value','__device__ int *pointer;',['pointer']),
 ('two_symbols','__constant__ int a[2]; __device__ float b[3];',['a','b']),
 ('extern_then_definition','extern __constant__ int values[8]; __constant__ int values[8];',['values']),
 ('extern_only','extern __constant__ int values[8];',[]),
 ('host_storage','int values[8];',[]),
 ('const_storage','__constant__ const int values[2] = {1,2};',[]),
 ('volatile_storage','__device__ volatile int values[2];',[]),
 ('const_member','struct Entry { const int x; }; __constant__ Entry table[2] = {{1},{2}};',[]),
 ('union_storage','union Entry { int x; float y; }; __constant__ Entry table[2];',[]),
 ('template_variable','template<class T> __device__ T value;',[]),
 ('inline_variable','inline __device__ int value;',[]),
 ('static_member','struct Entry { static __device__ int value; }; __device__ int Entry::value;',[]),
 ('macro_definition','#define DEFINE_SYMBOL __constant__ int value;\nDEFINE_SYMBOL',[]),
 ('symbol_macro_collision','#define ASCIFY_REGISTER_DEVICE_SYMBOL(a,b) ignored\n__constant__ int value;',None),
 ('namespace_collision','namespace ascify {}\n__constant__ int value;',None),
 ('address_builtin_macro','#define __builtin_addressof(x) nullptr\n__constant__ int value;',None),
]

PREFIX = '#include <cuda_runtime.h>\n'
DECL = '__constant__ int value;\n'
OFFSET = (PREFIX+DECL).index('value')
CASES.append(('generated_declaration_collision', DECL+'int ascify_generated_symbol_'+str(OFFSET)+'_address;', None))
macro_offset=0
for _ in range(5):
    body='#define ascify_generated_symbol_'+str(macro_offset)+'_address collision\n'+DECL
    macro_offset=(PREFIX+body).index('value')
CASES.append(('generated_macro_collision', body, None))

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
  row={'name':name,'expected_symbols':expected,'passed':False,'source_sha256':sha(src)};summary['cases'].append(row)
  z=run(d,'syntax',[a.clang,src,'-x','cuda','-std=c++17','-fsyntax-only','--cuda-host-only','-nocudalib','--cuda-path='+str(a.cuda_path),'-resource-dir='+str(a.resource_dir),'-Wno-unknown-cuda-version']);row['syntax_returncode']=z.returncode;save();assert z.returncode==0,(name,z.stderr)
  z=run(d,'convert',[a.binary,src,'--target-policy=dav-c310-vec','--simt-math=fast','--target-recipe=none','--default-preprocessor','--cuda-path='+str(a.cuda_path),'--clang-resource-directory='+str(a.resource_dir),'--migration-receipt='+str(receipt),'-o',dst,'--','-x','cuda','-std=c++17']);row['conversion_returncode']=z.returncode;save()
  if expected is None:
   assert z.returncode>0 and not dst.exists() and 'Ascify cannot publish CUDA compatibility output' in z.stderr,(name,z.stderr)
  else:
   assert z.returncode==0 and json.loads(receipt.read_text())['status']=='succeeded',(name,z.stderr)
   text=dst.read_text();actual=re.findall(r'^ASCIFY_REGISTER_DEVICE_SYMBOL\(::([\w:]+), ascify_generated_symbol_\d+\);$',text,re.M)
   assert actual==expected,(name,actual,expected,text)
   row['output_sha256']=sha(dst)
  assert sha(src)==row['source_sha256'];row['passed']=True;save();print(name,'passed',flush=True)
 assert sha(a.binary)==summary['binary_sha256'] and sha(a.clang)==summary['clang_sha256']
 summary.update(passed=True,total=len(CASES));save()
if __name__=='__main__':main()
