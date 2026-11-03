#!/usr/bin/env python3
"""Syntax-first kernel-bound/erased-helper CLI matrix; never a device test."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PREFIX = '#include <cuda_runtime.h>\n'
KERNEL = '__global__ void k(int*p){int i=threadIdx.x;if(i==0)*p=1;}\n'
POTENTIAL = 'int main(){int g,b;return cudaOccupancyMaxPotentialBlockSize(&g,&b,(void*)k,0,0);}\n'
HELPER = 'static int report(void* kernel,int block){int n;return cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,kernel,block,0);}\n'

def fixtures():
    def row(name, body, accepted, markers=(), auxiliary=None, frontend_compat=False):
        return dict(name=name, source=PREFIX+body, accepted=accepted, markers=list(markers),
                    auxiliary=auxiliary or {}, frontend_compat=frontend_compat)
    attributes='int attributes(){cudaFuncAttributes a;auto status=cudaFuncGetAttributes(&a,k);return a.maxThreadsPerBlock+int(status); }\n'
    yield row('default_explicit_policy',KERNEL+attributes+POTENTIAL,True,
              ['ASCIFY_GLOBAL __launch_bounds__(1024)', 'cudaOccupancyMaxPotentialBlockSizeMaxThreads<1024>',
               'cudaFuncGetAttributesMaxThreads<1024>'])
    bounded=KERNEL.replace('__global__','__global__ __launch_bounds__(512)')
    yield row('source_bound512',bounded+POTENTIAL,True,
              ['__launch_bounds__(512)', 'cudaOccupancyMaxPotentialBlockSizeMaxThreads<512>'])
    yield row('source_bound2048',KERNEL.replace('__global__','__global__ __launch_bounds__(2048)')+
              'int main(){int n;return cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,(void*)k,32,0);}\n',True,
              ['__launch_bounds__(2048)','cudaOccupancyMaxActiveBlocksPerMultiprocessorMaxThreads<2048>'])
    yield row('original_limit128',KERNEL+POTENTIAL.replace(',0,0)',',0,128)'),True,
              ['MaxPotentialBlockSizeMaxThreads<1024>(&g,&b,(void*)k,0,128)'])
    yield row('erased_helper_default',KERNEL+HELPER+'int main(){return report((void*)k,32);}\n',True,
              ['cudaOccupancyMaxActiveBlocksPerMultiprocessorMaxThreads<1024>', 'report((void*)k,32)'])
    yield row('helper_same_bound_domain',bounded+bounded.replace(' k(',' other(')+HELPER+
              'int main(){return report((void*)k,32)+report((void*)other,32);}\n',True,
              ['cudaOccupancyMaxActiveBlocksPerMultiprocessorMaxThreads<512>'])
    yield row('manual32_unchanged',KERNEL+HELPER+
              'int main(){int block=32;return report((void*)k,block);}\n',True,
              ['int block=32','report((void*)k,block)'])
    yield row('argument_once',KERNEL+
              'int main(){int g,b,limit=128;return cudaOccupancyMaxPotentialBlockSize(&g,&b,(void*)k,0,limit++);}\n',True,
              ['MaxPotentialBlockSizeMaxThreads<1024>(&g,&b,(void*)k,0,limit++)'])
    yield row('macro_leading_kernel_signature','#define PREFIX\nPREFIX\n'+KERNEL+POTENTIAL,False)
    yield row('undefined_kernel','__global__ void k(int*);\n'+POTENTIAL,False)
    yield row('macro_bound','#define B 512\n'+KERNEL.replace('__global__','__global__ __launch_bounds__(B)')+POTENTIAL,False)
    yield row('conditional_definition','#if 1\n'+KERNEL+'#endif\n'+POTENTIAL,False)
    yield row('kernel_pointer_variable',KERNEL+
              'int main(){int g,b;void* f=(void*)k;return cudaOccupancyMaxPotentialBlockSize(&g,&b,f,0,0);}\n',False)
    yield row('helper_external',KERNEL+HELPER.replace('static int','int')+
              'int main(){return report((void*)k,32);}\n',False)
    yield row('helper_address_escape',KERNEL+HELPER+
              'int main(){auto f=&report;return f((void*)k,32);}\n',False)
    yield row('helper_parameter_escape',KERNEL+'static void* escaped;\n'+
              HELPER.replace('int n;','int n;escaped=kernel;')+'int main(){return report((void*)k,32);}\n',False)
    yield row('helper_different_bounds',bounded+KERNEL.replace(' k(',' other(')+HELPER+
              'int main(){return report((void*)k,32)+report((void*)other,32);}\n',False)
    yield row('helper_unknown_actual',KERNEL+HELPER+
              'int route(void*p){return report(p,32);}int main(){return route((void*)k);}\n',False)
    yield row('helper_overloaded',KERNEL+HELPER+
              'static int report(int,int){return 0;}int main(){return report((void*)k,32);}\n',False)
    yield row('helper_inquiry',KERNEL+HELPER+
              'int q=__is_identifier(report);int main(){return report((void*)k,32);}\n',False)
    yield row('helper_assembly_caller',KERNEL+HELPER+
              'void hidden(){asm volatile("bl _ZL6reportPvi");}int main(){return report((void*)k,32); }\n',False)
    yield row('helper_unreviewed_include',KERNEL+HELPER+
              '#include \"hidden.h\"\nint main(){return report((void*)k,32); }\n',False,
              auxiliary={'hidden.h':'void hidden(){asm volatile(\"bl _ZL6reportPvi\");}\n'})
    yield row('foreign_API',KERNEL+
              'namespace foreign{int cudaOccupancyMaxPotentialBlockSize(int*,int*,void*,size_t,int){return 0;}}\n'
              'int main(){int g,b;return foreign::cudaOccupancyMaxPotentialBlockSize(&g,&b,(void*)k,0,0);}\n',False)
    yield row('API_address_escape',KERNEL+
              'int main(){int n;cudaError_t (*f)(int*,const void*,int,size_t)=&cudaOccupancyMaxActiveBlocksPerMultiprocessor;return f(&n,(void*)k,32,0);}\n',False)
    yield row('forged_kernel_pointer',KERNEL+
              'int main(){int g,b;return cudaOccupancyMaxPotentialBlockSize(&g,&b,(void*)1,0,0);}\n',False)
    yield row('helper_verified_frontend_profile',KERNEL+HELPER+'int main(){return report((void*)k,32);}\n',True,
              ['cudaOccupancyMaxActiveBlocksPerMultiprocessorMaxThreads<1024>'],frontend_compat=True)
    # Reuse the exact existing profile bytes as a caller copy. A matching name
    # and byte payload cannot acquire the product provider's physical identity.
    host_math=(Path(__file__).resolve().parents[2]/'frontend_compat/ascify-admitted-v1/host_math.h').read_text()
    yield row('helper_caller_host_math_copy',KERNEL+HELPER+
              '#include "host_math.h"\nint main(){return report((void*)k,32);}\n',False,
              auxiliary={'host_math.h':host_math},frontend_compat=True)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['binary','clang','cuda-path','resource-dir','evidence-dir']:
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.evidence_dir.mkdir(parents=True,exist_ok=False)
    cases=list(fixtures());assert len(cases)==27 and sum(c['accepted'] for c in cases)==9
    tools={str(x.resolve()):sha(x) for x in [a.binary,a.clang,Path(__file__)]}
    result=dict(passed=False,cases=[],CCEC_invocations=0,device_executed=False,original_sample_successes=0)
    def save():(a.evidence_dir/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    def run(folder,name,argv):
        (folder/(name+'.argv.json')).write_text(json.dumps(list(map(str,argv)),indent=2)+'\n')
        r=subprocess.run(list(map(str,argv)),capture_output=True,timeout=120)
        (folder/(name+'.stdout')).write_bytes(r.stdout);(folder/(name+'.stderr')).write_bytes(r.stderr)
        (folder/(name+'.process.json')).write_text(json.dumps({'returncode':r.returncode})+'\n')
        return r
    try:
        for case in cases:
            folder=a.evidence_dir/case['name'];folder.mkdir();source=folder/'input.cu';source.write_text(case['source'])
            auxiliary={}
            for name,contents in case['auxiliary'].items():
                path=folder/name;path.write_text(contents);auxiliary[str(path)]=sha(path)
            row=dict(name=case['name'],accepted=case['accepted'],passed=False,input_sha256=sha(source),auxiliary_sha256=auxiliary);result['cases'].append(row);save()
            common=['-x','cuda','-std=c++17','--cuda-path='+str(a.cuda_path)]
            r=run(folder,'original_syntax',[a.clang,source,*common,'--cuda-host-only','-nocudalib',
                  '-resource-dir='+str(a.resource_dir),'-fsyntax-only','-Wno-unknown-cuda-version'])
            assert r.returncode==0,(case['name'],r.stderr.decode())
            for mode in ['fresh'] if case['accepted'] else ['fresh','existing']:
                output=folder/(mode+'.cce');receipt=folder/(mode+'.receipt.json');sentinel=b'ORIGINAL OUTPUT SENTINEL\n'
                if mode=='existing':output.write_bytes(sentinel)
                r=run(folder,mode,[a.binary,source,'--target-policy=dav-c310-vec','--target-recipe=none',
                      '--simt-math=fast','--default-preprocessor','--cuda-path='+str(a.cuda_path),
                      '--clang-resource-directory='+str(a.resource_dir),'--migration-receipt='+str(receipt),'-o',output,
                      *(['--frontend-compat=ascify-admitted-v1'] if case['frontend_compat'] else []),'--',*common])
                if case['accepted']:
                    assert r.returncode==0,(case['name'],r.stderr.decode())
                    text=output.read_text();assert all(s in text for s in case['markers']),(case['name'],text)
                    if case['name']=='argument_once':assert text.count('limit++')==1
                else:
                    assert r.returncode!=0 and (output.read_bytes()==sentinel if mode=='existing' else not output.exists())
                    assert 'occupancy kernel-bound boundary' in r.stderr.decode(),(case['name'],r.stderr.decode())
            assert sha(source)==row['input_sha256'] and all(sha(Path(p))==v for p,v in auxiliary.items());row['passed']=True;save()
        assert all(sha(Path(p))==v for p,v in tools.items())
        result.update(passed=True,case_count=27,positive=9,negative=18);save()
    except Exception as error:
        result['error']=str(error);save();raise
if __name__=='__main__':main()
