#!/usr/bin/env python3
"""Real-SDK dormant/active line boundaries; no target programs or NPU run."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
for key in ('binary','clang','cuda-path','resource-dir','evidence-dir'):
    parser.add_argument('--'+key,type=Path,required=True)
args=parser.parse_args()
for key in ('binary','clang','cuda_path','resource_dir'):
    setattr(args,key,getattr(args,key).resolve(strict=True))
args.evidence_dir=args.evidence_dir.resolve()
args.evidence_dir.mkdir(parents=True,exist_ok=False)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
runtime='#include <cuda_runtime.h>\n'
probe=('#if defined(__has_include) && __has_include(<cuda_profiler_api.h>)\n'
       '#include <cuda_profiler_api.h>\n#define USER_CAPTURE_REGION 1\n#endif\n')
calls=('int main(){\n#ifdef USER_CAPTURE_REGION\n'
       'cudaError_t a=cudaProfilerStart();if(a!=cudaSuccess)return 1;\n'
       'cudaError_t b=cudaProfilerStop();return b==cudaSuccess?0:2;\n'
       '#else\nreturn 3;\n#endif\n}\n')
definition=('#ifndef ARBITRARY_CAPTURE_H\n#define ARBITRARY_CAPTURE_H\n'
            '#define DORMANT_MARK __LINE__\n#define DORMANT_ALIAS DORMANT_MARK\n#endif\n')
header='#include "arbitrary_capture.h"\n'
cases=[
    ('dormant_before',header+runtime+probe+calls,{},False,False),
    ('dormant_after',runtime+probe+header+calls,{},False,False),
    ('dormant_transitive_recursive','#include "wrapper.h"\n'+runtime+probe+calls,
     {'wrapper.h':header},False,True),
    ('explicit_main',header+runtime+probe+'int observed=__LINE__;\n'+calls,{},True,False),
    ('inactive_main',header+runtime+probe+'#if 0\nint observed=__LINE__;\n#endif\n'+calls,{},True,False),
    ('active_main_before',header+'int observed=DORMANT_MARK;\n'+runtime+probe+calls,{},True,False),
    ('active_main_after',header+runtime+probe+'int observed=DORMANT_MARK;\n'+calls,{},True,False),
    ('active_alias_main',header+runtime+probe+'int observed=DORMANT_ALIAS;\n'+calls,{},True,False),
    ('active_other_header_before',header+'#include "consumer.h"\n'+runtime+probe+calls,
     {'consumer.h':'int observed=DORMANT_MARK;\n'},True,False),
    ('active_alias_other_header_after',header+runtime+probe+'#include "consumer.h"\n'+calls,
     {'consumer.h':'int observed=DORMANT_ALIAS;\n'},True,False),
    ('retained_header_line_directive',header+runtime+probe+calls,
     {'arbitrary_capture.h':'#line 90\n'+definition},True,False),
    ('retained_header_inquiry',header+runtime+probe+calls,
     {'arbitrary_capture.h':definition+'#if __has_include(<stdio.h>)\nint available;\n#endif\n'},True,False),
]
summary={'schema':'ascify.profiler-dormant-line-frontend.v1','passed':False,
         'binary_sha256':sha(args.binary),'clang_sha256':sha(args.clang),'cases':[],
         'programs_executed':0,'npu_executed':False,'backend_status':'unverified'}
def save():write(args.evidence_dir/'summary.json',summary)
def run(directory,stage,argv):
    argv=list(map(str,argv));write(directory/(stage+'.argv.json'),argv)
    result=subprocess.run(argv,stdin=subprocess.DEVNULL,capture_output=True,timeout=90)
    (directory/(stage+'.stdout.log')).write_bytes(result.stdout)
    (directory/(stage+'.stderr.log')).write_bytes(result.stderr)
    write(directory/(stage+'.process.json'),{'returncode':result.returncode,
          'argv_sha256':sha(directory/(stage+'.argv.json')),
          'stdout_sha256':sha(directory/(stage+'.stdout.log')),
          'stderr_sha256':sha(directory/(stage+'.stderr.log'))})
    return result
save()
try:
    for name,source,extra_headers,refused,recursive in cases:
        directory=args.evidence_dir/name;directory.mkdir()
        original=directory/'input.cu';original.write_text(source)
        for filename,text in {'arbitrary_capture.h':definition,**extra_headers}.items():
            (directory/filename).write_text(text)
        inputs={str(p):sha(p) for p in directory.iterdir() if p.is_file()}
        output=directory/'output.cce';receipt=directory/'receipt.json'
        row={'name':name,'expected_refusal':refused,'recursive':recursive,
             'input_sha256':sha(original),'input_files':inputs,'passed':False}
        summary['cases'].append(row);save()
        common=['-x','cuda','-std=c++17','-I'+str(args.cuda_path/'include'),
                '--cuda-path='+str(args.cuda_path),'-resource-dir='+str(args.resource_dir),'-fgpu-rdc']
        result=run(directory,'original_syntax',[args.clang,original,*common,
                   '--cuda-host-only','-nocudalib','-fsyntax-only','-Wno-unknown-cuda-version'])
        row['syntax_rc']=result.returncode;save()
        assert result.returncode==0,(name,result.stderr)
        argv=[args.binary,original,'--target-policy=dav-c310-vec','--simt-math=fast','--target-recipe=none',
              '--default-preprocessor','--frontend-compat=ascify-admitted-v1',
              '--cuda-path='+str(args.cuda_path),'--clang-resource-directory='+str(args.resource_dir),
              '--migration-receipt='+str(receipt),'-o',output]
        if recursive:argv.append('--local-headers-recursive')
        result=run(directory,'convert',[*argv,'--',*common]);row['conversion_rc']=result.returncode
        record=json.loads(receipt.read_text());row['receipt_sha256']=sha(receipt)
        if refused:
            assert result.returncode!=0 and record['status']=='failed',(name,result.stderr)
            assert not output.exists() and not Path(str(output)+'.headers').exists(),name
            assert not any('cuda_profiler_availability' in item for item in record['inputs']),record
        else:
            assert result.returncode==0 and record['status']=='succeeded',(name,result.stderr)
            text=output.read_text()
            assert '__has_include(<acl/acl_prof.h>)' in text and '__has_include(<cuda_profiler_api.h>)' not in text
            assert '#error "Ascify admitted profiler branch requires the target ACL profiler header"' in text
            assert '#define USER_CAPTURE_REGION 1' in text and text.count('#ifdef USER_CAPTURE_REGION')==1
            assert 'ascify::cudaProfilerStart()' in text and 'ascify::cudaProfilerStop()' in text
            proof=record['inputs'][0]['cuda_profiler_availability']
            assert proof['rule']=='positive-physical-sdk-probe-v1' and proof['backend_status']=='unverified'
            assert proof['target_header']=='acl/acl_prof.h' and len(proof['evidence'])==1
            if recursive:
                bundle=Path(str(output)+'.headers');assert bundle.is_dir()
                assert any('#define DORMANT_MARK __LINE__' in p.read_text()
                           for p in bundle.rglob('*') if p.is_file()),name
            else:assert header in text,name
            row['output_sha256']=sha(output)
        assert all(sha(Path(path))==expected for path,expected in inputs.items()),name
        row['passed']=True;save();print(name,'PASS',flush=True)
    assert sha(args.binary)==summary['binary_sha256'] and sha(args.clang)==summary['clang_sha256']
    summary.update(passed=True,total=len(cases));save()
except Exception as error:
    summary['error']=type(error).__name__+': '+str(error);save();raise
