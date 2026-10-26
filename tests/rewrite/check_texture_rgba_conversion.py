#!/usr/bin/env python3
"""Authenticated RGBA factory/fetch, atomic publication, real target compile; no NPU."""
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
    'uchar_factory': (True, 'void f(){auto c=cudaCreateChannelDesc<uchar4>();}'),
    'uchar_alias_factory': (True, 'using Pixel=uchar4; void f(){auto c=cudaCreateChannelDesc<Pixel>();}'),
    'explicit_factory': (True, 'void f(){auto c=cudaCreateChannelDesc(8,8,8,8,cudaChannelFormatKindUnsigned);}'),
    'float_fetch': (True, '__global__ void k(float4*p,cudaTextureObject_t t){p[0]=tex2D<float4>(t,.5f,.5f);}'),
    'uchar_fetch': (True, '__global__ void k(uchar4*p,cudaTextureObject_t t){p[0]=tex2D<uchar4>(t,.5f,.5f);}'),
    'float_layered': (True, '__global__ void k(float4*p,cudaTextureObject_t t){p[0]=tex2DLayered<float4>(t,.5f,.5f,1);}'),
    'uchar_layered': (True, '__global__ void k(uchar4*p,cudaTextureObject_t t){p[0]=tex2DLayered<uchar4>(t,.5f,.5f,1);}'),
    'float_alias': (True, 'using Pixel=float4; __global__ void k(Pixel*p,cudaTextureObject_t t){p[0]=tex2D<Pixel>(t,.5f,.5f);}'),
    'all_channels': (True, '__global__ void k(float*p,cudaTextureObject_t t){auto v=tex2D<float4>(t,.5f,.5f);p[0]=v.x;p[1]=v.y;p[2]=v.z;p[3]=v.w;}'),
    'layout_and_return': (True, 'static_assert(sizeof(uchar4)==4 && alignof(uchar4)==4 && sizeof(float4)==16 && alignof(float4)==16,"vector ABI"); __global__ void k(uchar4*p,cudaTextureObject_t t){auto v=tex2D<uchar4>(t,.5f,.5f);p[0]=v;}'),
    'quoted_context': (True, '#include "local_rgba.h"\n__global__ void k(float4*p,cudaTextureObject_t t){p[0]=local_fetch(t);}'),
    'float4_channel_rejected': (False, 'void f(){auto c=cudaCreateChannelDesc<float4>();}'),
    'scalar_u8_rejected': (False, '__global__ void k(unsigned char*p,cudaTextureObject_t t){p[0]=tex2D<unsigned char>(t,.5f,.5f);}'),
    'signed_char4_rejected': (False, '__global__ void k(char4*p,cudaTextureObject_t t){p[0]=tex2D<char4>(t,.5f,.5f);}'),
    'ushort4_channel_rejected': (False, 'void f(){auto c=cudaCreateChannelDesc<ushort4>();}'),
    'const_uchar_channel_rejected': (False, 'void f(){auto c=cudaCreateChannelDesc<const uchar4>();}'),
    'caller_forward_rejected': (False, 'struct uchar4; void f(){auto c=cudaCreateChannelDesc<uchar4>();}'),
    'copied_provider_rejected': (False, 'void f(){auto c=cudaCreateChannelDesc<uchar4>();}'),
    'factory_redeclaration_rejected': (False, 'template<> __host__ inline cudaChannelFormatDesc cudaCreateChannelDesc<uchar4>(); void f(){auto c=cudaCreateChannelDesc<uchar4>();}'),
    'namespace_fake_rejected': (False, 'namespace caller{struct alignas(4) uchar4{unsigned char x,y,z,w;};} void f(){auto c=cudaCreateChannelDesc<caller::uchar4>();}'),
    'fake_size_rejected': (False, 'struct alignas(4) Pixel{unsigned char x,y,z,w,extra;};void f(){auto c=cudaCreateChannelDesc<Pixel>();}'),
    'fake_fields_rejected': (False, 'struct alignas(4) Pixel{unsigned short x,y;};void f(){auto c=cudaCreateChannelDesc<Pixel>();}'),
    'function_address_rejected': (False, 'auto factory=&cudaCreateChannelDesc<uchar4>;'),
    'qualified_fetch_rejected': (False, '__global__ void k(uchar4*p,cudaTextureObject_t t){p[0]=::tex2D<uchar4>(t,.5f,.5f);}'),
    'pointer_overload_rejected': (False, '__global__ void k(uchar4*p,cudaTextureObject_t t){tex2D<uchar4>(p,t,.5f,.5f);}'),
    'dependent_fetch_rejected': (False, 'template<class T> __device__ T f(cudaTextureObject_t t){return tex2D<T>(t,.5f,.5f);}'),
    'macro_fetch_rejected': (False, '#define FETCH(t) tex2D<float4>(t,.5f,.5f)\n__global__ void k(float4*p,cudaTextureObject_t t){p[0]=FETCH(t);}'),
    'old_tag_macro_rejected': (False, '#define unsigned_element other_tag\nvoid f(){auto c=cudaCreateChannelDesc<uchar4>();}'),
    'new_kind_macro_rejected': (False, '#define TextureSampleRGBANormalized other_tag\nvoid f(){auto c=cudaCreateChannelDesc<uchar4>();}'),
    'version_macro_rejected': (False, '#define abi_version ignored\nvoid f(){auto c=cudaCreateChannelDesc<uchar4>();}'),
    'sampler_macro_rejected': (False, '#define textureSampleRGBANormalized user_sampler\nvoid f(){auto c=cudaCreateChannelDesc<uchar4>();}'),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','clang','cuda-path','resource-dir','ccec','evidence-dir'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--target-include',action='append',default=[])
    args=parser.parse_args()
    require(not args.evidence_dir.exists(),'fresh evidence directory required')
    require(args.target_include,'real installed helper and CANN include paths required')
    sdk={'vector_types.h':'ded087a2c3ca89fd3a5150589175aca0aa4714357016209ea3e7a51f94845e75',
         'channel_descriptor.h':'b78717961fe4aa0c3529fd5e6260331746249bd59b5fad1eebfdc7bc12ba9d1a',
         'texture_indirect_functions.h':'29f1850893f326de3c1b7048a63cab04e847b56570c7f4b35850e69dec727c03'}
    for relative,digest in sdk.items():require(sha(args.cuda_path/'include'/relative)==digest,'SDK identity: '+relative)
    identities={str(p.resolve()):sha(p.resolve()) for p in (args.binary,args.clang,args.ccec,Path(__file__))}
    args.evidence_dir.mkdir(parents=True)
    environment=os.environ.copy()
    for key in ('CPATH','CPLUS_INCLUDE_PATH','C_INCLUDE_PATH','OBJC_INCLUDE_PATH'):
        environment.pop(key,None)
    summary={'passed':False,'case_count':len(CASES),'tool_sha256':identities,'sdk_sha256':sdk,
             'rows':[],'device_executed':False,'sample_count_delta':0}
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
        header=directory/'local_rgba.h'
        header.write_text('#ifndef LOCAL_RGBA_H\n#define LOCAL_RGBA_H\n__device__ inline float4 local_fetch(cudaTextureObject_t t){return tex2D<float4>(t,.5f,.5f);}\n#endif\n')
        source.write_text('#include <cuda_runtime.h>\n'+body+'\n')
        extra=['-I',str(directory)]
        inputs=[source,header]
        if name=='copied_provider_rejected':
            shim=directory/'shim';shim.mkdir()
            (shim/'cuda_runtime.h').write_text('#include "copied_vector_types.h"\n#include <channel_descriptor.h>\n')
            (shim/'copied_vector_types.h').write_bytes((args.cuda_path/'include/vector_types.h').read_bytes())
            inputs+=sorted(shim.glob('*.h'))
            extra=['-nocudainc','-nocudalib','-I',str(shim),'-I',str(args.cuda_path/'include')]
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
        if name=='quoted_context':argv+=['--local-headers-recursive']
        argv+=['--','-x','cuda','-std=c++17',*extra]
        argv=list(map(str,argv));result=run(argv,directory/'convert');parsed=json.loads(receipt.read_text())
        require((result.returncode==0)==accepted,(name,result.stderr.decode(errors='replace')))
        if accepted:
            require(parsed['status']=='succeeded' and '#include <ascify/texture_compat.hpp>' in output.read_text(),name)
            obj=directory/'target.o'
            compile_argv=[args.ccec,'-O2','-std=c++17','-x','dpp','--cce-aicore-arch=dav-c310-vec',
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
    summary['passed']=True;save()


if __name__=='__main__':
    main()
