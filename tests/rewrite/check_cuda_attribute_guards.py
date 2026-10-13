#!/usr/bin/env python3
"""Specialize only proved host/device attribute guards; all inputs parse as CUDA."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

MARKER = '#if 1 /* Ascify: proven CUDA host/device attributes */'
GUARD = '#ifdef __CUDACC__\n__host__ __device__\n#endif\n'
FUNCTION = 'inline float function(float x) { return x < 0.f ? -1.f : 1.f; }\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--syntax-only', action='store_true')
    args = ap.parse_args()
    cuda = os.environ['ASCIFY_CUDA_PATH']
    resource = os.environ['ASCIFY_CLANG_RESOURCE_DIRECTORY']
    clang = os.environ.get('CLANG', 'clang')
    binary = os.environ.get('ASCIFY_BINARY')
    body = GUARD + FUNCTION
    cases = [
        ('ordinary', body, 1, [], False),
        ('multiline_attributes', body.replace('__host__ __device__', '__host__\n    __device__'), 1, [], False),
        ('body_comments', body.replace('__host__ __device__', '/* left */ __host__ /* middle */ __device__ // right'), 1, [], False),
        ('primary_template', 'template<class T>\n'+GUARD+'inline T function(T x) { return x < T(0) ? T(-1) : T(1); }\n', 1, [], False),
        ('two_guards', body+body.replace('function', 'second'), 2, [], False),
        ('local_header', body, 1, [], True),
        ('extra_body_keyword', body.replace('__host__ __device__', 'inline __host__ __device__').replace('inline float', 'float'), 0, [], False),
        ('extra_attribute', body.replace('__host__ __device__', '__host__ __device__ __noinline__'), 0, [], False),
        ('device_only', body.replace('__host__ ', ''), 0, [], False),
        ('reversed', body.replace('__host__ __device__', '__device__ __host__'), 0, [], False),
        ('else_branch', body.replace('#endif', '#else\n__host__\n#endif'), 0, [], False),
        ('elif_branch', body.replace('#endif', '#elif 1\n__host__\n#endif'), 0, [], False),
        ('nested_branch', body.replace('__host__ __device__', '#if 1\n__host__ __device__\n#endif'), 0, [], False),
        ('body_alias', '#define ATTR __host__ __device__\n'+body.replace('__host__ __device__', 'ATTR'), 0, [], False),
        ('condition_alias', '#define ACTIVE __CUDACC__\n'+body.replace('#ifdef __CUDACC__', '#if ACTIVE'), 0, [], False),
        ('condition_expression', body.replace('#ifdef __CUDACC__', '#if defined(__CUDACC__)'), 0, [], False),
        ('source_control', '#undef __CUDACC__\n#define __CUDACC__ 1\n'+body, 0, [], False),
        ('inactive', '#undef __CUDACC__\n'+body, 0, [], False),
        ('later_control', body+'#undef __CUDACC__\n', 0, [], False),
        ('command_line_control', body, 0, ['-D__CUDACC__=1'], False),
        ('user_host', '#undef __host__\n#define __host__ __attribute__((host))\n'+body, 0, [], False),
        ('user_device', '#undef __device__\n#define __device__ __attribute__((device))\n'+body, 0, [], False),
        ('user_attribute_dependency', '#undef __annotate__\n#define __annotate__(a) __attribute__((a))\n'+body, 0, [], False),
        ('spliced_directive', body.replace('#ifdef __CUDACC__', '#ifdef __CUDACC__ \\\n'), 0, [], False),
        ('commented_directive', body.replace('#ifdef', '/* prefix */ #ifdef'), 0, [], False),
        ('declaration_only', GUARD+'float function(float);\n', 0, [], False),
        ('class_method', 'struct S {\n'+body+'};\n', 0, [], False),
        ('class_template_friend', 'template<class T> struct S { friend\n'+body+'};\n', 0, [], False),
    ]
    with tempfile.TemporaryDirectory(prefix='ascify-attribute-guards-') as temp:
        work = Path(temp)
        for label, text, count, flags, in_header in cases:
            directory = work / label
            directory.mkdir()
            source = directory / 'source.cu'
            source.write_text('#include <cuda_runtime.h>\n' + ('#include "value.cuh"\n' if in_header else text))
            if in_header:
                (directory / 'value.cuh').write_text(text)
            inputs = [p for p in directory.iterdir() if p.is_file()]
            before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
            syntax = subprocess.run([clang, '-x', 'cuda', '--cuda-host-only', '-nocudalib',
                '--cuda-path='+cuda, '-Wno-unknown-cuda-version', '-std=c++17', '-fsyntax-only',
                *flags, str(source)], text=True, capture_output=True)
            assert syntax.returncode == 0, label+' original CUDA invalid\n'+syntax.stderr
            if args.syntax_only:
                print(label+': source syntax valid; conversion not run')
                continue
            assert binary, 'ASCIFY_BINARY required'
            output = directory / 'source.cce'
            result = subprocess.run([binary, str(source), '--target-policy=dav-c310-vec',
                '--default-preprocessor', '--local-headers-recursive', '--cuda-path='+cuda,
                '--clang-resource-directory='+resource, '-o', str(output), '--', '-x', 'cuda',
                '-std=c++17', *flags], text=True, capture_output=True)
            assert result.returncode == 0, label+'\n'+result.stderr
            generated_files = [output]
            if in_header:
                generated_files = list(Path(str(output)+'.headers').rglob('value.cuh.dpp'))
                assert len(generated_files) == 1, label
            generated = generated_files[0].read_text()
            assert generated.count(MARKER) == count, label+'\n'+generated+'\n'+result.stderr
            expected = text.replace('__device__', '__aicore__')
            if count:
                expected = expected.replace('#ifdef __CUDACC__', MARKER)
            assert expected in generated, label+': body or unrelated directive changed\n'+generated
            assert {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs} == before
            print(label+': attribute guards='+str(count))
    print('CUDA attribute guards: 28 source-valid cases; '+('source syntax only' if args.syntax_only else '6 admitted, 22 retained'))


if __name__ == '__main__':
    main()
