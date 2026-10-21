#!/usr/bin/env python3
"""Bounded array rectangle copies: physical SDK, atomic publication and CCEC; no NPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


CASES = {
    'array_copy2d_direct': (True, 'void f(cudaArray*a,const void*p){auto e=cudaMemcpy2DToArray(a,4,1,p,32,16,3,cudaMemcpyHostToDevice);}'),
    'array_copy2d_helper': (True, '#include <helper_cuda.h>\nvoid f(cudaArray*a,const void*p){checkCudaErrors(cudaMemcpy2DToArray(a,4,1,p,32,16,3,cudaMemcpyDeviceToDevice));}'),
    'array_copy2d_address': (False, 'auto p=&cudaMemcpy2DToArray;'),
    'array_copy2d_qualified': (False, 'void f(cudaArray*a,const void*p){::cudaMemcpy2DToArray(a,0,0,p,32,16,3,cudaMemcpyHostToDevice);}'),
    'array_copy2d_user_namespace': (False, 'namespace user{int cudaMemcpy2DToArray(int);void f(){cudaMemcpy2DToArray(1);}}'),
    'array_copy2d_macro': (False, '#define COPY(a,p) cudaMemcpy2DToArray(a,0,0,p,32,16,3,cudaMemcpyHostToDevice)\nvoid f(cudaArray*a,const void*p){COPY(a,p);}'),
    'array_copy2d_redeclaration': (False, 'extern "C" cudaError_t cudaMemcpy2DToArray(cudaArray_t,size_t,size_t,const void*,size_t,size_t,size_t,cudaMemcpyKind);void f(cudaArray*a,const void*p){cudaMemcpy2DToArray(a,0,0,p,32,16,3,cudaMemcpyHostToDevice);}'),
    'array_copy2d_name_observation': (False, '#define S0(x) #x\n#define S(x) S0(x)\nconst char*p=S(cudaMemcpy2DToArray);cudaTextureDesc d{};'),
    'rgba_array_h2d': (True, 'void f(const uchar4*p){cudaArray*a=nullptr;auto c=cudaCreateChannelDesc<uchar4>();cudaMallocArray(&a,&c,8,5);auto e=cudaMemcpy2DToArray(a,4,1,p,32,16,3,cudaMemcpyHostToDevice);}'),
    'rgba_array_helper_d2d': (True, '#include <helper_cuda.h>\nvoid f(const uchar4*p){cudaArray*a=nullptr;auto c=cudaCreateChannelDesc<uchar4>();checkCudaErrors(cudaMallocArray(&a,&c,8,5));checkCudaErrors(cudaMemcpy2DToArray(a,4,1,p,32,16,3,cudaMemcpyDeviceToDevice));}'),
}

EXPECTED = {
    'array_copy2d_direct': ['ascify::cudaMemcpy2DToArray(a,4,1,p,32,16,3,ACL_MEMCPY_HOST_TO_DEVICE)'],
    'array_copy2d_helper': ['ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS(ascify::cudaMemcpy2DToArray'],
    'rgba_array_h2d': ['ascify::cudaCreateChannelDesc<uchar4>', 'ascify::cudaMemcpy2DToArray'],
    'rgba_array_helper_d2d': ['ascify::cudaCreateChannelDesc<uchar4>', 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS(ascify::cudaMemcpy2DToArray'],
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','clang','cuda-path','resource-dir','ccec','evidence-dir'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--target-include',action='append',default=[])
    args=parser.parse_args()
    helpers=Path(__file__).resolve().parent/'fixtures/nvidia_samples/Common'
    require(helpers.is_dir(),'frozen Samples helpers required')
    helper_identity={str(p):sha(p) for p in sorted(helpers.rglob('*')) if p.is_file()}
    require(not args.evidence_dir.exists(),'fresh evidence directory required')
    require(args.target_include,'real installed helper and CANN include paths required')
    sdk={'vector_types.h':'ded087a2c3ca89fd3a5150589175aca0aa4714357016209ea3e7a51f94845e75',
         'channel_descriptor.h':'b78717961fe4aa0c3529fd5e6260331746249bd59b5fad1eebfdc7bc12ba9d1a',
         'texture_indirect_functions.h':'29f1850893f326de3c1b7048a63cab04e847b56570c7f4b35850e69dec727c03'}
    for relative,digest in sdk.items():require(sha(args.cuda_path/'include'/relative)==digest,'SDK identity: '+relative)
    sdk['cuda_runtime_api.h']='2c6732345e9662d91733f5e45fa9c76c955640ef02c4bf17ceb7f1bcc209d3de'
    require(sha(args.cuda_path/'include/cuda_runtime_api.h')==sdk['cuda_runtime_api.h'],'physical Runtime API source')
    identities={str(p.resolve()):sha(p.resolve()) for p in (args.binary,args.clang,args.ccec,Path(__file__))}
    args.evidence_dir.mkdir(parents=True)
    environment=os.environ.copy()
    for key in ('CPATH','CPLUS_INCLUDE_PATH','C_INCLUDE_PATH','OBJC_INCLUDE_PATH'):
        environment.pop(key,None)
    summary={'passed':False,'case_count':len(CASES),'tool_sha256':identities,'sdk_sha256':sdk,
             'helper_sha256':helper_identity,'rows':[],'device_executed':False,'sample_count_delta':0}
    def save():
        (args.evidence_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    def run(argv,directory):
        directory.mkdir();argv=list(map(str,argv))
        (directory/'argv.json').write_text(json.dumps(argv,indent=2)+'\n')
        result=subprocess.run(argv,capture_output=True,env=environment,timeout=180)
        (directory/'stdout.log').write_bytes(result.stdout);(directory/'stderr.log').write_bytes(result.stderr)
        (directory/'process.json').write_text(json.dumps({'returncode':result.returncode})+'\n')
        return result
    save()
    for name,(accepted,body) in CASES.items():
        directory=args.evidence_dir/name;directory.mkdir()
        source=directory/'input.cu';output=directory/'output.cce';receipt=directory/'receipt.json'
        source.write_text('#include <cuda_runtime.h>\n'+body+'\n')
        extra=['-I',str(directory),'-I',str(helpers)]
        inputs=[source]
        originals={str(p):sha(p) for p in inputs}
        row={'case':name,'expected_conversion_success':accepted,'passed':False,'source_sha256':originals}
        summary['rows'].append(row);save()
        syntax=run([args.clang,source,'-x','cuda','-std=c++17','--cuda-host-only','-fsyntax-only','-nocudalib',
                    '--cuda-path='+str(args.cuda_path),'-resource-dir='+str(args.resource_dir),
                    '-Wno-unknown-cuda-version',*extra],directory/'source_validation')
        require(syntax.returncode==0,(name,'invalid original fixture',syntax.stderr.decode(errors='replace')))
        bundle=directory/'output.cce.headers'
        if not accepted:
            output.write_text('preserve previous output\n');bundle.mkdir()
            (bundle/'prior.txt').write_text('preserve previous bundle sentinel\n')
        prior={str(p):sha(p) for p in (output,bundle/'prior.txt')} if not accepted else {}
        argv=[args.binary,source,'--frontend-compat=ascify-admitted-v1','--target-policy=dav-c310-vec',
              '--simt-math=fast','--default-preprocessor','--cuda-path='+str(args.cuda_path),
              '--clang-resource-directory='+str(args.resource_dir),'-o',output,'--migration-receipt='+str(receipt)]
        argv+=['--local-headers-recursive']
        argv+=['--','-x','cuda','-std=c++17',*extra]
        argv=list(map(str,argv));result=run(argv,directory/'convert');parsed=json.loads(receipt.read_text())
        require((result.returncode==0)==accepted,(name,result.stderr.decode(errors='replace')))
        if accepted:
            require(parsed['status']=='succeeded' and '#include <ascify/texture_compat.hpp>' in output.read_text(),name)
            require(all(fragment in output.read_text() for fragment in EXPECTED[name]),name+' mapped call missing')
            obj=directory/'target.o'
            compile_argv=[args.ccec,'-O2','-std=c++17','-funsigned-char','-x','dpp','--cce-aicore-arch=dav-c310-vec',
                          '-c',output,'-o',obj]
            for include in args.target_include:compile_argv+=['-I',include]
            compiled=run(compile_argv,directory/'target_compile')
            require(compiled.returncode==0 and obj.read_bytes().startswith(b'\x7fELF'),
                    (name,compiled.stderr.decode(errors='replace')))
            row.update(output_sha256=sha(output),target_object_sha256=sha(obj),target_compiled=True)
        else:
            def rejected(result,parsed):
                stderr=result.stderr.decode(errors='replace')
                return result.returncode!=0 and parsed['status']=='failed' and 'fatal error:' not in stderr and any(
                    marker in stderr for marker in ('CUDA texture boundary','CUDA runtime-API header boundary',
                                                    'Ascify cannot publish CUDA compatibility output'))
            require(rejected(result,parsed),(name,'wrong rejection',result.stderr.decode(errors='replace')))
            require(all(sha(Path(p))==h for p,h in prior.items()),name+' existing output changed')
            fresh=directory/'fresh_output.cce';fresh_receipt=directory/'fresh_receipt.json'
            fresh_argv=list(argv);fresh_argv[fresh_argv.index('-o')+1]=str(fresh)
            fresh_argv[fresh_argv.index('--migration-receipt='+str(receipt))]='--migration-receipt='+str(fresh_receipt)
            result=run(fresh_argv,directory/'convert_fresh')
            require(rejected(result,json.loads(fresh_receipt.read_text())),name+' fresh rejection')
            require(not fresh.exists() and not fresh.is_symlink() and
                    not fresh.with_name(fresh.name+'.headers').exists() and
                    not fresh.with_name(fresh.name+'.headers').is_symlink(),name+' fresh output published')
            require(all(sha(Path(p))==h for p,h in prior.items()),name+' earlier output changed')
            row['publication_checks']='existing_output_and_sentinel_preserved;fresh_absent'
        require(all(sha(Path(p))==h for p,h in originals.items()),name+' original changed')
        row['passed']=True;save();print(name,'PASS',flush=True)
    require(all(sha(Path(p))==h for p,h in identities.items()),'tool identity changed')
    for relative,digest in sdk.items():require(sha(args.cuda_path/'include'/relative)==digest,'SDK changed')
    require(all(sha(Path(path))==digest for path,digest in helper_identity.items()),'helper source changed')
    summary['passed']=True;save()


if __name__=='__main__':
    main()
