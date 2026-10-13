#!/usr/bin/env python3
"""Resolved SDK int/uint generic atomicExch calls only; every fixture must first parse as CUDA."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HELPER = '::ascify::atomic_exch_generic'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--syntax-only', action='store_true')
    args = parser.parse_args()
    cuda = os.environ['ASCIFY_CUDA_PATH']
    resource = os.environ['ASCIFY_CLANG_RESOURCE_DIRECTORY']
    clang = os.environ.get('CLANG', 'clang')
    binary = os.environ.get('ASCIFY_BINARY')
    cases = [
        ('signed_pointer', '__device__ int f(int* p) { return atomicExch(p, 9); }', 1),
        ('unsigned_pointer', '__device__ unsigned f(unsigned* p) { return atomicExch(p, 9); }', 1),
        ('signed_reference', '__device__ int f(int& p) { return atomicExch(&p, 9); }', 1),
        ('unsigned_reference', '__device__ unsigned f(unsigned& p) { return atomicExch(&p, 9); }', 1),
        ('shared_scalar', '__global__ void f(int* output) { __shared__ unsigned p; output[0] = atomicExch(&p, 9); }', 1),
        ('shared_helper', '__device__ int f() { __shared__ int p; return atomicExch(&p, 9); }', 1),
        ('fixed_template_reference', 'template<class T> __device__ unsigned f(unsigned& p) { return atomicExch(&p, 9); }\n__device__ unsigned g(unsigned& p) { return f<int>(p); }', 1),
        ('fixed_template_pointer', 'template<class T> __device__ int f(int* p) { return atomicExch(p, 9); }\n__device__ int g(int* p) { return f<float>(p); }', 1),
        ('one_evaluation', '__device__ int f(int* p, int value) { return atomicExch(p, value++); }', 1),
        ('local_initializer', '__device__ int f(int& p) { int old = atomicExch(&p, 9); return old; }', 1),
        ('two_calls', '__device__ int f(int& p) { int a = atomicExch(&p, 9); int b = atomicExch(&p, 10); return a+b; }', 2),
        ('float_type', '__device__ float f(float* p) { return atomicExch(p, 9.f); }', 0),
        ('wide_type', '__device__ unsigned long long f(unsigned long long* p) { return atomicExch(p, 9ULL); }', 0),
        ('pointer_alias', '__device__ int f(int* p) { int* q=p; return atomicExch(q, 9); }', 0),
        ('reference_alias', '__device__ int f(int& p) { int& q=p; return atomicExch(&q, 9); }', 0),
        ('local_storage', '__device__ int f() { int p=0; return atomicExch(&p, 9); }', 0),
        ('field', 'struct S { int p; }; __device__ int f(S* p) { return atomicExch(&p->p, 9); }', 0),
        ('explicit_cast', '__device__ int f(unsigned* p) { return atomicExch(reinterpret_cast<int*>(p), 9); }', 0),
        ('parameter_offset', '__device__ int f(int* p) { return atomicExch(p+1, 9); }', 0),
        ('dependent_reference', 'template<class T> __device__ int f(T& p) { return atomicExch(&p, 9); }\n__device__ int g(int& p) { return f(p); }', 0),
        ('dependent_value', 'template<int N> __device__ int f(int& p) { return atomicExch(&p, N); }\n__device__ int g(int& p) { return f<9>(p); }', 0),
        ('class_template', 'template<class T> struct S { __device__ int f(int& p) { return atomicExch(&p, 9); } };\n__device__ int g(int& p) { S<int> s; return s.f(p); }', 0),
        ('qualified', '__device__ int f(int* p) { return ::atomicExch(p, 9); }', 0),
        ('macro_argument', '#define ARG p\n__device__ int f(int* p) { return atomicExch(ARG, 9); }', 0),
        ('macro_callee', '#define EXCH atomicExch\n__device__ int f(int* p) { return EXCH(p, 9); }', 0),
        ('outer_macro', '#define KEEP(x) x\n__device__ int f(int* p) { return KEEP(atomicExch(p, 9)); }', 0),
        ('project_redeclaration', '__device__ int atomicExch(int*, int);\n__device__ int f(int* p) { return atomicExch(p, 9); }', 0),
        ('user_namespace', 'namespace custom { __device__ int atomicExch(int*,int); __device__ int f(int* p) { return atomicExch(p,9); } }', 0),
        ('helper_macro', '#define atomic_exch_generic user_macro\n__device__ int f(int* p) { return atomicExch(p, 9); }', -1),
    ]
    with tempfile.TemporaryDirectory(prefix='ascify-generic-exch-') as temporary:
        work = Path(temporary)
        for label, text, count in cases:
            source = work/(label+'.cu')
            source.write_text('#include <cuda_runtime.h>\n'+text)
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            syntax = subprocess.run([clang,'-x','cuda','--cuda-host-only','-nocudalib',
                '--cuda-path='+cuda,'-Wno-unknown-cuda-version','-std=c++17','-fsyntax-only',
                '-fgpu-rdc',str(source)],text=True,capture_output=True)
            assert syntax.returncode == 0, label+' original syntax\n'+syntax.stderr
            if args.syntax_only:
                print(label+': source syntax valid; conversion not run')
                continue
            assert binary, 'ASCIFY_BINARY required'
            output = work/(label+'.cce')
            command = [binary,str(source),'--target-policy=dav-c310-vec','--default-preprocessor',
                '--cuda-path='+cuda,'--clang-resource-directory='+resource,'-o',str(output),
                '--','-x','cuda','-std=c++17','-fgpu-rdc']
            result = subprocess.run(command,text=True,capture_output=True)
            if count < 0:
                assert result.returncode > 0 and not output.exists(), label+'\n'+result.stderr
                assert 'collides with the frozen compat header' in result.stderr, result.stderr
            else:
                assert result.returncode == 0, label+'\n'+result.stderr
                generated = output.read_text()
                assert generated.count(HELPER) == count, label+'\n'+generated+'\n'+result.stderr
                expected_body = text.replace('__device__', '__aicore__').replace('__global__', 'ASCIFY_GLOBAL').replace('__shared__', '__ubuf__')
                if count:
                    assert '<ascify/ascify_cuda_compat.hpp>' in generated
                    expected_body = expected_body.replace('atomicExch(', HELPER+'(')
                    if label == 'one_evaluation':
                        assert HELPER+'(p, value++)' in generated
                assert expected_body in generated, label+': unexpected body edit\n'+generated
            assert hashlib.sha256(source.read_bytes()).hexdigest() == before
            print(label+': '+('rejected without publication' if count < 0 else 'rewrites='+str(count)))
    print('generic atomicExch: 29 source-valid cases; '+('syntax only, conversion not run' if args.syntax_only else
          '11 admitted, 17 retained, 1 macro publication refusal'))


if __name__ == '__main__':
    main()
