#!/usr/bin/env python3
"""OpenMP helper-transaction boundaries using actual CUDA and compiler omp.h."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HELPERS = ROOT / 'tests/rewrite/fixtures/nvidia_samples/Common'
HEADER_SHA = '4066c7a384a8c9a8a241287db0e2625216746da831c61083124d11ae203ab66f'
BODY = '#pragma omp parallel\n  { checkCudaErrors(cudaFree(p)); }\n'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--cuda-path', type=Path, required=True)
    ap.add_argument('--resource-dir', type=Path, required=True)
    ap.add_argument('--clang', required=True)
    ap.add_argument('--openmp-include', type=Path)
    ap.add_argument('--evidence-dir', type=Path, required=True)
    ap.add_argument('--syntax-only', action='store_true')
    args = ap.parse_args()
    clang = Path(shutil.which(args.clang) or args.clang).resolve(strict=True)
    omp_include = (args.openmp_include or clang.parent.parent / 'include').resolve(strict=True)
    header = omp_include / 'omp.h'
    assert sha(header) == HEADER_SHA, 'this contract requires the reviewed physical LLVM23 omp.h'
    out = args.evidence_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = [
        ('plain', '', BODY, '', True, True, 'ordinary'),
        ('header', '#include <omp.h>\n', BODY, '', True, True, 'ordinary'),
        ('line_comment', '', BODY.replace('parallel\n','parallel // exact directive\n'), '', True, True, 'ordinary'),
        ('two_parallel', '', BODY + BODY, '', True, True, 'ordinary'),
        ('no_openmp', '#include <omp.h>\n', BODY, '', False, False, 'ordinary'),
        ('num_threads_clause', '', BODY.replace('parallel','parallel num_threads(2)'), '', True, False, 'ordinary'),
        ('default_clause', '', BODY.replace('parallel','parallel default(shared)'), '', True, False, 'ordinary'),
        ('task', '', BODY.replace('parallel','task'), '', True, False, 'ordinary'),
        ('macro_pragma', '', BODY.replace('#pragma omp parallel','_Pragma("omp parallel")'), '', True, False, 'ordinary'),
        ('spliced_pragma', '', BODY.replace('omp parallel','omp \\\nparallel'), '', True, False, 'ordinary'),
        ('extra_taskwait', '#include <omp.h>\n', BODY + '#pragma omp taskwait\n', '', True, False, 'ordinary'),
        ('copied_header', '#include <omp.h>\n', BODY, '', True, False, 'copy'),
        ('mutated_header', '#include <omp.h>\n', BODY, '', True, False, 'mutated_copy'),
        ('system_copy_header', '#include <omp.h>\n', BODY, '', True, False, 'system_copy'),
        ('quoted_header', '#include "omp.h"\n', BODY, '', True, False, 'ordinary'),
        ('macro_include', '#define OMP_HEADER <omp.h>\n#include OMP_HEADER\n', BODY, '', True, False, 'ordinary'),
        ('transitive_include', '#include "wrapper.h"\n', BODY, '', True, False, 'ordinary'),
        ('lambda', '', 'auto f=[&]() {\n' + BODY + '}; f();\n', '', True, False, 'ordinary'),
        ('template', '', BODY, 'void instantiate(){ host<void>(nullptr); }\n', True, False, 'template'),
        ('class_template_method', '', BODY, 'void instantiate(){ Owner<void>{}.host(nullptr); }\n', True, False, 'class_template'),
        ('unsafe_status', 'cudaError_t customStatus();\n', BODY + 'checkCudaErrors(customStatus());\n', '', True, False, 'ordinary'),
        ('observable_pragma', '', BODY + '#pragma omp parallel if(EXIT_WAIVED)\n{}\n', '', True, False, 'ordinary'),
    ]
    summary = {'schema':'ascify.openmp-helper-boundaries.v1','passed':False,'syntax_only':args.syntax_only,
               'binary_sha256':sha(args.binary),'test_sha256':sha(Path(__file__)),
               'clang_sha256':sha(clang),'omp_header':str(header),'omp_header_sha256':sha(header),
               'device_executed':False,'new_original_device_successes':0,'cases':[]}
    def save():
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    def run(d,phase,argv):
        argv=list(map(str,argv));(d/(phase+'.argv.json')).write_text(json.dumps(argv,indent=2)+'\n')
        r=subprocess.run(argv,capture_output=True,text=True,timeout=60)
        (d/(phase+'.stdout.log')).write_text(r.stdout);(d/(phase+'.stderr.log')).write_text(r.stderr)
        return r
    save()
    for name,prelude,body,suffix,enabled,removed,mode in cases:
        d=out/name;d.mkdir();source=d/'input.cu';output=d/'output.cpp';receipt=d/'receipt.json'
        (d/'wrapper.h').write_text('#include <omp.h>\n')
        selected_include=omp_include
        if mode in ('copy','mutated_copy','system_copy'):
            selected_include=d/'copied';selected_include.mkdir()
            (selected_include/'omp.h').write_bytes(header.read_bytes() + (b'\n' if mode=='mutated_copy' else b''))
        prefix='template<class T>\n' if mode=='template' else ''
        if mode=='class_template': prefix='template<class T> struct Owner {\n'
        tail='};\n' if mode=='class_template' else ''
        source.write_text('#include <helper_cuda.h>\n'+prelude+prefix+'void host(void *p) {\n'+body+'}\n'+tail+suffix)
        flags=['-x','cuda','-std=c++17','-I'+str(HELPERS),'-I'+str(d)]
        flags += ['-isystem',str(selected_include)] if mode=='system_copy' else ['-I'+str(selected_include)]
        if enabled:flags.append('-fopenmp')
        r=run(d,'syntax',[clang,source,*flags,'-fsyntax-only','--cuda-host-only','-nocudalib',
            '--cuda-path='+str(args.cuda_path),'-resource-dir='+str(args.resource_dir),'-Wno-unknown-cuda-version'])
        row={'name':name,'source_sha256':sha(source),'expected_helper_removed':removed,'syntax_returncode':r.returncode,'passed':False}
        summary['cases'].append(row);save();assert r.returncode==0,(name,r.stderr)
        if not args.syntax_only:
            r=run(d,'convert',[args.binary,source,'--target-policy=dav-c310-vec','--simt-math=fast','--target-recipe=none',
                '--default-preprocessor','--local-headers-recursive','--cuda-path='+str(args.cuda_path),
                '--clang-resource-directory='+str(args.resource_dir),'--migration-receipt='+str(receipt),'-o',output,'--',*flags])
            row['conversion_returncode']=r.returncode;save()
            assert r.returncode==0,(name,r.stderr)
            assert json.loads(receipt.read_text())['status']=='succeeded'
            text=output.read_text()
            assert ('#include <helper_cuda.h>' not in text)==removed,(name,text,r.stderr)
            assert text.count('#pragma omp')==source.read_text().count('#pragma omp'),(name,text)
            assert re.findall(r'^\s*#pragma omp[^\n]*',text,re.M)==re.findall(r'^\s*#pragma omp[^\n]*',source.read_text(),re.M),(name,text)
            if removed:
                assert 'checkCudaErrors(' not in text and 'unsupported_macro=0' in r.stderr,(name,text,r.stderr)
            else:
                assert text.count('checkCudaErrors(')==source.read_text().count('checkCudaErrors('),(name,text)
                assert 'ASCIFY_CUDA_CHECK' not in text,(name,text)
        row['passed']=True;save()
    assert sha(header)==summary['omp_header_sha256'] and sha(args.binary)==summary['binary_sha256']
    summary['passed']=True;save()
    print(json.dumps({'passed':True,'cases':len(cases),'syntax_only':args.syntax_only,'device_executed':False}))


if __name__=='__main__':
    main()
