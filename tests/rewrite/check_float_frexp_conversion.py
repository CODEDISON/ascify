#!/usr/bin/env python3
"""Resolved SDK float frexp calls only; every fixture must first parse as CUDA."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HELPER = '::ascify::cuda_frexp_float'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--syntax-only', action='store_true')
    args = parser.parse_args()
    cuda = os.environ['ASCIFY_CUDA_PATH']
    resource = os.environ['ASCIFY_CLANG_RESOURCE_DIRECTORY']
    clang = os.environ.get('CLANG', 'clang')
    binary = os.environ.get('ASCIFY_BINARY')
    body = '__device__ float function(float x, int* exponent) { return frexp(x, exponent); }\n'
    cases = [
        ('device_float', body, 1),
        ('kernel_float', '__global__ void kernel(float* out, float x) { int exponent; out[0] = frexp(x, &exponent); }\n', 1),
        ('local_initializer', '__device__ float function(float x) { int exponent; float value = frexp(x, &exponent); return value; }\n', 1),
        ('float_alias', 'using Value = float;\n__device__ Value function(Value x, int* exponent) { return frexp(x, exponent); }\n', 1),
        ('one_evaluation', '__device__ float function(float x, int* exponent) { return frexp(x++, exponent++); }\n', 1),
        ('two_calls', '__device__ float function(float x, int* exponent) { float a = frexp(x, exponent); float b = frexp(x+1.0f, exponent+1); return a+b; }\n', 2),
        ('host_float', body.replace('__device__ ', ''), 0),
        ('host_device', '__host__ '+body, 0),
        ('device_double', body.replace('float', 'double'), 0),
        ('host_double', body.replace('__device__ ', '').replace('float', 'double'), 0),
        ('qualified_global', body.replace('return frexp', 'return ::frexp'), 0),
        ('function_template', 'template<class T> '+body, 0),
        ('class_template', 'template<class T> struct Object { '+body+' };\n', 0),
        ('device_lambda', '__device__ float function(float x, int* exponent) { auto f = [](float y, int* e) { return frexp(y,e); }; return f(x,exponent); }\n', 0),
        ('macro_callee', '#define F frexp\n'+body.replace('return frexp', 'return F'), 0),
        ('macro_argument', '#define ARG x\n'+body.replace('frexp(x,', 'frexp(ARG,'), 0),
        ('outer_macro', '#define KEEP(x) x\n'+body.replace('return frexp(x, exponent)', 'return KEEP(frexp(x, exponent))'), 0),
        ('user_namespace', 'namespace project { __device__ float frexp(float x, int* e) { *e=7; return x; }\n'+body+'}\n', 0),
        ('user_overload', '__device__ float frexp(float x, long* e) { *e=7; return x; }\n'+body.replace('int*', 'long*'), 0),
        ('indirect_call', '__device__ float function(float x, int* exponent) { auto p = static_cast<float(*)(float,int*)>(&frexp); return p(x,exponent); }\n', 0),
        ('native_name_macro', '#define frexpf replacement\n'+body, -1),
        ('helper_name_macro', '#define cuda_frexp_float replacement\n'+body, -1),
    ]
    with tempfile.TemporaryDirectory(prefix='ascify-float-frexp-') as temporary:
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
                expected_body = text.replace('__device__', '__aicore__').replace('__global__', 'ASCIFY_GLOBAL')
                if count:
                    assert '<ascify/ascify_cuda_compat.hpp>' in generated
                    expected_body = expected_body.replace('frexp(', HELPER+'(')
                    if label == 'one_evaluation':
                        assert HELPER+'(x++, exponent++)' in generated
                assert expected_body in generated, label+': unexpected body edit\n'+generated
            assert hashlib.sha256(source.read_bytes()).hexdigest() == before
            print(label+': '+('rejected without publication' if count < 0 else 'rewrites='+str(count)))
    print('float frexp: 22 source-valid cases; '+('syntax only, conversion not run' if args.syntax_only else
          '6 admitted, 14 retained, 2 macro publication refusals'))


if __name__ == '__main__':
    main()
