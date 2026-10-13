#!/usr/bin/env python3
"""Keep frozen quoted helper includes under their own atomic transaction."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HELPERS = ROOT / 'tests/rewrite/fixtures/nvidia_samples/Common'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--syntax-only', action='store_true')
    args = parser.parse_args()
    binary = os.environ.get('ASCIFY_BINARY')
    cuda = os.environ['ASCIFY_CUDA_PATH']
    resource = os.environ['ASCIFY_CLANG_RESOURCE_DIRECTORY']
    clang = os.environ.get('CLANG', 'clang')
    call = 'void release(void* p) { checkCudaErrors(cudaFree(p)); }\n'
    cases = [
        ('quoted', '#include "helper_cuda.h"\n'+call, {}, 'converted'),
        ('angled_control', '#include <helper_cuda.h>\n'+call, {}, 'converted'),
        ('ordinary_sibling', '#include "helper_cuda.h"\n#include "value.h"\n'+call+
         'int use() { return value(); }\n', {'value.h':'inline int value() { return 7; }\n'}, 'converted'),
        ('inherited_helper', '#include "helper_cuda.h"\n#include "release.cuh"\n',
         {'release.cuh':call}, 'converted'),
        ('observed_macro', '#include "helper_cuda.h"\n#ifdef checkCudaErrors\n'
         'constexpr int observed = 1;\n#endif\n'+call, {}, 'retained'),
        ('unknown_status', '#include "helper_cuda.h"\ncudaError_t custom_status();\n'
         'void use() { checkCudaErrors(custom_status()); }\n', {}, 'retained'),
        ('duplicate_helper', '#include "helper_cuda.h"\n#include "helper_cuda.h"\n'+call, {}, 'retained'),
        ('stringized_helper', '#include "helper_cuda.h"\n#define TEXT_IMPL(x) #x\n'
         '#define TEXT(x) TEXT_IMPL(x)\nconst char* observed = TEXT(checkCudaErrors(cudaFree(nullptr)));\n',
         {}, 'retained'),
        ('user_same_name', '#include "helper_cuda.h"\nint use() { return project_value(); }\n',
         {'helper_cuda.h':'inline int project_value() { return 17; }\n'}, 'ordinary'),
        ('macro_include', '#define HELPER "helper_cuda.h"\n#include HELPER\n'+call, {}, 'rejected'),
    ]
    fixture_before = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in HELPERS.iterdir() if p.is_file()}
    with tempfile.TemporaryDirectory(prefix='ascify-quoted-helper-') as temp:
        work = Path(temp)
        for label, source, headers, kind in cases:
            directory = work/label
            directory.mkdir()
            input_root = directory/'input'
            input_root.mkdir()
            unit = input_root/'source.cu'
            unit.write_text(source)
            for name, contents in headers.items():
                (input_root/name).write_text(contents)
            inputs = [p for p in input_root.rglob('*') if p.is_file()]
            before = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
            syntax = subprocess.run([clang,'-x','cuda','--cuda-host-only','-nocudalib',
                '--cuda-path='+cuda,'-Wno-unknown-cuda-version','-std=c++17','-fsyntax-only',
                '-I'+str(HELPERS),str(unit)],text=True,capture_output=True)
            assert syntax.returncode == 0, label+': invalid original CUDA source\n'+syntax.stderr
            if args.syntax_only:
                print(label+': original CUDA syntax valid; conversion not executed')
                continue
            assert binary, 'ASCIFY_BINARY must select the converter under test'
            output = directory/'source.cce'
            result = subprocess.run([binary,str(unit),'--target-policy=dav-c310-vec',
                '--default-preprocessor','--local-headers-recursive','--frontend-compat=ascify-admitted-v1',
                '--cuda-path='+cuda,'--clang-resource-directory='+resource,'-I'+str(HELPERS),
                '-o',str(output),'--','-x','cuda','-std=c++17','-I'+str(HELPERS)],text=True,capture_output=True)
            assert {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs} == before
            assert not list(directory.glob('.ascify-local-closure*')), label+': staging leaked'
            bundle = Path(str(output)+'.headers')
            if kind == 'rejected':
                assert result.returncode > 0 and not output.exists() and not bundle.exists(), result.stderr
                assert 'macro-produced quoted include' in result.stderr, result.stderr
            else:
                assert result.returncode == 0, label+'\n'+result.stderr
                text = output.read_text()
                if kind == 'converted':
                    assert 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' in text, text
                    assert 'helper_cuda.h' not in text, text
                    assert 'include removed' in result.stderr, result.stderr
                    assert not list(bundle.rglob('helper_cuda.h.dpp')), label+': helper separately selected'
                    if label == 'ordinary_sibling':
                        assert len(list(bundle.rglob('value.h.dpp'))) == 1
                    if label == 'inherited_helper':
                        assert 'context_inline=' in (bundle/'.ascify-local-closure').read_text()
                elif kind == 'retained':
                    assert text.count('#include "helper_cuda.h"') == source.count('#include "helper_cuda.h"'), text
                    assert 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' not in text, text
                    assert 'include kept' in result.stderr, result.stderr
                    assert not list(bundle.rglob('helper_cuda.h.dpp'))
                else:
                    assert len(list(bundle.rglob('helper_cuda.h.dpp'))) == 1
                    assert 'project_value()' in text and 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' not in text
                    assert '#include "helper_cuda.h"' not in text
            print(label+': '+kind+'; original CUDA syntax valid')
    assert fixture_before == {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in HELPERS.iterdir() if p.is_file()}
    print('quoted helper source syntax: 10 cases passed; conversion not executed' if args.syntax_only else
          'quoted helper gate: 4 converted, 4 retained, 1 ordinary user header, 1 macro rejection')


if __name__ == '__main__':
    main()
