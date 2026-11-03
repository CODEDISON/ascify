#!/usr/bin/env python3
"""Syntax-first scalar64 AST/CPU regressions. No CCEC, ACL calls or NPU use."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
CUDA = '#include <cuda_runtime.h>\n'
ADD = '::ascify::detail::scalar_add_rne<'
ASSIGN = '::ascify::detail::scalar_add_assign_rne<'


def fixtures():
    def case(name, body, adds=0, assigns=0, cpu=None, accepted=True, prefix=CUDA,
             flags=(), standard='c++17', guard=False, retained=()):
        return dict(name=name, source=prefix + body, accepted=accepted, adds=adds,
                    assigns=assigns, cpu=cpu, flags=list(flags), standard=standard,
                    guard=guard, retained=list(retained))

    yield case('plain_add', '__device__ double pair(double a,double b){return a+b;}\n',
               adds=1, cpu='scalar64_check_add(pair);')
    yield case('reference_assign', '__device__ void step(double& a,double b){a+=b;}\n',
               assigns=1, cpu='scalar64_check_assign(step);')
    yield case('array_assign', '__global__ void step(double* p,const double* q,unsigned i){p[i]+=q[i];}\n',
               assigns=1, cpu='''
for(const auto& t:scalar64_cases){
  double p[3]={scalar64_from_raw<double>(0x0123456789abcdef),scalar64_from_raw<double>(t.left),
               scalar64_from_raw<double>(0xfedcba9876543210)};
  double q[3]={0,scalar64_from_raw<double>(t.right),0};
  step(p,q,1); scalar64_expect(scalar64_raw(p[1]),t.expected);
  scalar64_expect(scalar64_raw(p[0]),0x0123456789abcdef);
  scalar64_expect(scalar64_raw(p[2]),0xfedcba9876543210);
  scalar64_expect(scalar64_raw(q[1]),t.right);
}''')
    yield case('nested_add', '__device__ double tree(double a,double b,double c){return (a+b)+c;}\n',
               adds=2, cpu='''
scalar64_expect(scalar64_raw(tree(scalar64_from_raw<double>(0x3ff0000000000001),
  scalar64_from_raw<double>(0x3ca0000000000000),scalar64_from_raw<double>(0xbff0000000000000))),
  0x3cc0000000000000);''')
    yield case('assign_shuffle_once', '''
__device__ void step(double& a,double b){a += __shfl_down_sync(0xffffffffu,b,1,32);}
''', assigns=1, cpu='''
scalar64_check_assign([](double& a,double b){
  scalar64_test::reset(); step(a,b);
  if(scalar64_test::calls!=1) std::abort();
  scalar64_expect(scalar64_test::input,scalar64_raw(b));
});''')
    yield case('closed_add_template', '''
template<class T> __device__ T tree(T a,T b,T c){return (a+b)+c;}
__device__ double use_d(double a,double b,double c){return tree(a,b,c);}
__device__ float use_f(float a,float b,float c){return tree(a,b,c);}
__device__ int use_i(int a,int b,int c){return tree(a,b,c);}
''', adds=2, guard=True, cpu='''
if(use_i(27,14,-33)!=8 || use_f(16777216.f,1.f,-16777216.f)!=0.f) std::abort();
scalar64_expect(scalar64_raw(use_d(scalar64_from_raw<double>(0x3ff0000000000001),
  scalar64_from_raw<double>(0x3ca0000000000000),scalar64_from_raw<double>(0xbff0000000000000))),
  0x3cc0000000000000);''')
    yield case('closed_assign_template', '''
template<class T> __device__ void step(T& a,T b){a+=b;}
__device__ void use_d(double& a,double b){step(a,b);}
__device__ void use_f(float& a,float b){step(a,b);}
__device__ void use_i(int& a,int b){step(a,b);}
''', assigns=1, guard=True, cpu='''
scalar64_check_assign(use_d);
float f=16777216.f; use_f(f,1.f); if(f!=16777216.f) std::abort();
int i=27;use_i(i,14);if(i!=41)std::abort();''')
    yield case('raw_down_double', '''
__device__ double move(double value,unsigned delta){return __shfl_down_sync(0xffffffffu,value,delta,32);}
''', cpu='''
for(uint64_t bits:{UINT64_C(1),UINT64_C(0x8000000000000000),UINT64_C(0x7ff0000000000042),
                  UINT64_C(0x3ff0000000000001)}){
  scalar64_test::reset(); scalar64_test::supply_output=true;
  scalar64_test::output=bits^UINT64_C(0x123456789abcdef0);
  scalar64_expect(scalar64_raw(move(scalar64_from_raw<double>(bits),1)),scalar64_test::output);
  scalar64_expect(scalar64_test::input,bits);
  if(scalar64_test::calls!=1)std::abort();
}''')
    yield case('raw_index_double', '''
__device__ double move(double value){return __shfl_sync(0xffffffffu,value,31,32);}
''', cpu='''
scalar64_test::reset();scalar64_test::supply_output=true;scalar64_test::output=0x8000000000000001;
scalar64_expect(scalar64_raw(move(scalar64_from_raw<double>(0x7ff0000000000042))),0x8000000000000001);
scalar64_expect(scalar64_test::input,0x7ff0000000000042);
if(scalar64_test::calls!=1||scalar64_test::source_lane!=31)std::abort();''')
    for label, typename, function in [('signed64', 'long long', '__shfl_sync'),
                                      ('unsigned64', 'unsigned long long', '__shfl_down_sync')]:
        yield case('raw_' + label,
                   '__device__ ' + typename + ' move(' + typename + ' value){return ' +
                   function + '(0xffffffffu,value,1,32);}\n', cpu='''
scalar64_test::reset();
scalar64_expect(scalar64_raw(move(scalar64_from_raw<''' + typename + '''>(0xfedcba9876543210))),
  0xfedcba9876543210);
scalar64_expect(scalar64_test::input,0xfedcba9876543210);
if(scalar64_test::calls!=1)std::abort();''')
    yield case('closed_raw_transport_template', '''
template<class T> __device__ T move(T value){return __shfl_down_sync(0xffffffffu,value,1,32);}
__device__ double use_d(double v){return move(v);}
__device__ float use_f(float v){return move(v);}
__device__ int use_i(int v){return move(v);}
''', guard=True, cpu='''
scalar64_test::reset();
scalar64_expect(scalar64_raw(use_d(scalar64_from_raw<double>(0x3ff0000000000001))),0x3ff0000000000001);
if(use_f(1.25f)!=1.25f||use_i(12345)!=12345)std::abort();
if(scalar64_test::calls!=1)std::abort();''')
    cg = '''
__device__ double move(double value){
  const auto tile=cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
  return tile.shfl_down(value,1);
}
__device__ double index_move(double value){
  const auto tile=cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
  return tile.shfl(value,31);
}
'''
    for label, prefix in [('cuda_first', CUDA + '#include <cooperative_groups.h>\n'),
                          ('cg_first', '#include <cooperative_groups.h>\n' + CUDA)]:
        yield case('cg_' + label, cg, prefix=prefix, cpu='''
scalar64_test::reset();
scalar64_expect(scalar64_raw(move(scalar64_from_raw<double>(0x7ff0000000000042))),0x7ff0000000000042);
scalar64_expect(scalar64_raw(index_move(scalar64_from_raw<double>(0x8000000000000000))),0x8000000000000000);
if(scalar64_test::calls!=2)std::abort();''')
    host = 'double host_pair(double a,double b){return a+b;}\n'
    yield case('host_retained', host, retained=[host.strip()], cpu='scalar64_check_add(host_pair);')
    # The native host NaN policy is not the product's specified NaN policy.
    # Use only the finite/zero table for this retained-source neutral check.
    yield case('int_float_retained', '''
__device__ int add_i(int a,int b){return a+b;}
__device__ float add_f(float a,float b){return a+b;}
''', retained=['return a+b;'], cpu='''
if(add_i(27,14)!=41 || add_f(16777216.f,1.f)!=16777216.f)std::abort();''')

    negatives = [
        ('global_double_abi', '__global__ void kernel(double* out,double a,double b){out[0]=a+b;}\n'),
        ('global_template_abi', '''
template<class T> __global__ void kernel(T* out,T a,T b){out[0]=a+b;}
template __global__ void kernel<double>(double*,double,double);
'''),
        ('global_calls_exact_helper', '''
__device__ double pair(double a,double b){return a+b;}
__global__ void kernel(double* out,double a,double b){out[0]=pair(a,b);}
'''),
        ('lhs_index_effect', '__device__ void step(double* p,unsigned i,double b){p[i++]+=b;}\n'),
        ('lhs_pointer_effect', '__device__ void step(double* p,double b){*p+++=b;}\n'),
        ('rhs_mutates_lhs', '__device__ void step(double& a){a+=(a=1.0);}\n'),
        ('rhs_unknown_call', '__device__ double unknown(double&);\n__device__ void step(double& a){a+=unknown(a);}\n'),
        ('value_used_assign', '__device__ double step(double& a,double b){return a+=b;}\n'),
        ('volatile_assign', '__device__ void step(volatile double& a,double b){a+=b;}\n'),
        ('atomic_assign', '__device__ void step(_Atomic(double)* a,double b){*a+=b;}\n'),
        ('mixed_runtime_int', '__device__ double pair(double a,int b){return a+b;}\n'),
        ('mixed_runtime_float', '__device__ double pair(double a,float b){return a+b;}\n'),
        ('multiply_in_route', '__device__ double pair(double a,double b){return (a+b)*b;}\n'),
        ('compare_in_route', '__device__ double pair(double a,double b){double c=a+b;return c>b?a:b;}\n'),
        ('macro_operator', '#define PLUS(a,b) ((a)+(b))\n__device__ double pair(double a,double b){return PLUS(a,b);}\n'),
        ('macro_operand', '#define VALUE a\n__device__ double pair(double a,double b){return VALUE+b;}\n'),
        ('helper_macro', '#define scalar_add_rne project_value\n__device__ double pair(double a,double b){return a+b;}\n'),
        ('transport_sizeof_macro', '''
#define sizeof(T) 8
template<class T> __device__ T move(T value){return __shfl_down_sync(0xffffffffu,value,1,32);}
__device__ double use_d(double value){return move(value);}
'''),
        ('constexpr_function', '__device__ constexpr double pair(double a,double b){return a+b;}\n'),
        ('dual_host_device', '__host__ __device__ double pair(double a,double b){return a+b;}\n'),
        ('noexcept_function', '__device__ double pair(double a,double b) noexcept {return a+b;}\n'),
        ('sizeof_operator', '__device__ double pair(double a,double b){(void)sizeof(a+b);return a;}\n'),
        ('decltype_operator', '__device__ double pair(double a,double b){using R=decltype(a+b);return a;}\n'),
        ('noexcept_operator', '__device__ bool pair(double a,double b){return noexcept(a+b);}\n'),
        ('type_trait_operator', '__device__ double pair(double a,double b){static_assert(__is_same(decltype(a+b),double));return a;}\n'),
        ('constexpr_initializer', '__device__ double pair(double a){constexpr double c=1.0+0x1p-53;return a+c;}\n'),
        ('conditional_body', '__device__ double pair(double a,double b){\n#if 1\nreturn a+b;\n#else\nreturn a*b;\n#endif\n}\n'),
        ('body_pragma', '__device__ double pair(double a,double b){_Pragma("GCC diagnostic push") return a+b;}\n'),
        ('unproved_conversion', '''
struct Number{__device__ operator double() const{return 1.0;}};
__device__ double pair(Number a,double b){return a+b;}
'''),
        ('class_template_instance', '''
struct Number{double value;};
__device__ Number operator+(Number a,Number){return a;}
template<class T> __device__ T pair(T a,T b){return a+b;}
__device__ double use_d(double a,double b){return pair(a,b);}
__device__ Number use_n(Number a,Number b){return pair(a,b);}
'''),
        ('unsigned_math_template_instance', '''
template<class T> __device__ T pair(T a,T b){return a+b;}
__device__ double use_d(double a,double b){return pair(a,b);}
__device__ unsigned use_u(unsigned a,unsigned b){return pair(a,b);}
'''),
        ('function_address_escape', '''
template<class T> __device__ T pair(T a,T b){return a+b;}
__device__ double use_d(double a,double b){auto pointer=&pair<double>;return pointer(a,b);}
'''),
        ('explicit_specialization', '''
template<class T> __device__ T pair(T a,T b){return a+b;}
template<> __device__ double pair<double>(double a,double b){return a+b;}
__device__ double use_d(double a,double b){return pair(a,b);}
'''),
        ('narrow_wide_shuffle', '__device__ double move(double a){return __shfl_down_sync(0xffffffffu,a,1,16);}\n'),
        ('partial_wide_index', '__device__ double move(double a){return __shfl_sync(0xffffu,a,1,32);}\n'),
        ('out_of_range_wide_index', '__device__ double move(double a){return __shfl_sync(0xffffffffu,a,32,32);}\n'),
        ('fake_shuffle_provider', '''
namespace project { __device__ double __shfl_down_sync(unsigned,double v,unsigned,int){return v;} }
__device__ double move(double a){return project::__shfl_down_sync(0xffffffffu,a,1,32);}
'''),
    ]
    for name, body in negatives:
        yield case(name, body, accepted=False)
    yield case('global_abi_keep_flag', negatives[0][1], accepted=False,
               flags=['--no-lower-device-double-params'])
    yield case('consteval_function', '__device__ consteval double pair(double a,double b){return a+b;}\n',
               accepted=False, standard='c++20')
    yield case('cg_double_plus_exact', '''
__device__ double move(double a){
  auto tile=cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
  return cooperative_groups::reduce(tile,a,cooperative_groups::plus<double>());
}
''', prefix=CUDA + '#include <cooperative_groups/reduce.h>\n', cpu='''
scalar64_test::reset();
scalar64_expect(scalar64_raw(move(1.25)),scalar64_raw(40.0));
if(scalar64_test::calls!=5)std::abort();
''')
    yield case('cg_double_up_still_refused', '''
__device__ double move(double a){
  auto tile=cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
  return tile.shfl_up(a,1);
}
''', accepted=False, prefix=CUDA + '#include <cooperative_groups.h>\n')

    # Keep the original57 fixture domains above; cg_double_plus_exact is the
    # intentional newly admitted exact reduction boundary. These cases
    # exercise the builtin wrapped-double refusal and its nonactivation scope.
    wrapped_refusals = [
        ('atomic_typedef_assign', 'using D=double; using A=_Atomic(D);\n__device__ void step(A* a,D b){*a+=b;}\n'),
        ('atomic_ref_paren_assign', 'using A=_Atomic(double);\n__device__ void step(A& a,double b){(a)+=b;}\n'),
        ('atomic_array_assign', 'using A=_Atomic(double);\n__device__ void step(A* a,unsigned i,double b){a[i]+=b;}\n'),
        ('atomic_volatile_assign', '__device__ void step(volatile _Atomic(double)* a,double b){*a+=b;}\n'),
        ('atomic_binary_add', '__device__ double pair(_Atomic(double)* a,double b){return *a+b;}\n'),
        ('complex_double_add', 'using V=_Complex double;\n__device__ V pair(V a,V b){return a+b;}\n'),
        ('complex_double_assign', 'using V=_Complex double;\n__device__ void step(V& a,V b){a+=b;}\n'),
        ('vector_double_add', 'typedef double V __attribute__((ext_vector_type(2)));\n__device__ V pair(V a,V b){return a+b;}\n'),
        ('vector_double_assign', 'typedef double V __attribute__((ext_vector_type(2)));\n__device__ void step(V& a,V b){a+=b;}\n'),
    ]
    for name, body in wrapped_refusals:
        yield case(name, body, accepted=False)
    yield case('constinit_initializer', '''
__device__ double pair(double a){static constinit double c=1.0+2.0;return a+c;}
''', accepted=False, standard='c++20')
    yield case('host_complex_retained', '''
using V=_Complex double;
V pair(V a,V b){return a+b;}
''', retained=['return a+b;'], cpu='''
V a,b; __real__ a=1.0; __imag__ a=2.0; __real__ b=3.0; __imag__ b=4.0;
V result=pair(a,b); if(__real__ result!=4.0||__imag__ result!=6.0)std::abort();''')
    yield case('host_vector_retained', '''
typedef double V __attribute__((ext_vector_type(2)));
V pair(V a,V b){return a+b;}
''', retained=['return a+b;'], cpu='''
V result=pair(V{1.0,2.0},V{3.0,4.0});
if(result[0]!=4.0||result[1]!=6.0)std::abort();''')
    yield case('device_complex_copy_no_add', '''
using V=_Complex double;
__device__ void copy(V* out,const V* in){out[0]=in[0];}
''', retained=['out[0]=in[0];'], cpu='''
V in,out; __real__ in=scalar64_from_raw<double>(0x7ff0000000000042);
__imag__ in=scalar64_from_raw<double>(0x8000000000000000); copy(&out,&in);
scalar64_expect(scalar64_raw(__real__ out),0x7ff0000000000042);
scalar64_expect(scalar64_raw(__imag__ out),0x8000000000000000);''')
    for name, dtype, initial, increment, expected in [
        ('atomic_int_retained', 'int', '1', '2', '3'),
        ('atomic_float_retained', 'float', '1.0f', '0.5f', '1.5f'),
    ]:
        # A double declaration alone must not activate any refusal. These
        # preserve existing frontend behavior; no NPU atomic support is implied.
        yield case(name, 'using UnusedDouble=double;\n__device__ void step(_Atomic(' +
                   dtype + ')* a,' + dtype + ' b){*a+=b;}\n', retained=['*a+=b;'],
                   cpu='_Atomic(' + dtype + ') a; a=' + initial + '; step(&a,' +
                   increment + '); if(a!=' + expected + ')std::abort();')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--cuda-path', required=True)
    parser.add_argument('--resource-dir', required=True)
    parser.add_argument('--clang', default=os.environ.get('CLANG', 'clang++'))
    parser.add_argument('--cxx', default=os.environ.get('CXX', 'c++'))
    parser.add_argument('--evidence-dir')
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError('Assertions must remain enabled')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    summary = dict(schema='ascify.scalar64.cpu-regression.v1', device_execution=False,
                   binary_sha256=hashlib.sha256(Path(args.binary).read_bytes()).hexdigest(), cases=[])
    context = tempfile.TemporaryDirectory(prefix='ascify-scalar64-') if not args.evidence_dir else None
    work = Path(args.evidence_dir if args.evidence_dir else context.name).resolve()
    work.mkdir(parents=True, exist_ok=True)
    # CUDA13 keeps libcu++ under the selected SDK's include/cccl directory.
    # Match its ordinary SDK search root on BOTH source-validity and converter
    # invocations; do not change CUDA headers or disable their dependencies.
    cccl = Path(args.cuda_path).resolve() / 'include/cccl'
    cccl_header = cccl / 'cuda/std/type_traits'
    sdk_flags = []
    summary['external_sdk_includes'] = []
    if cccl_header.is_file():
        sdk_flags = ['-I' + str(cccl.resolve())]
        summary['external_sdk_includes'].append(dict(
            include_dir=str(cccl.resolve()), key_header=str(cccl_header.resolve()),
            key_header_sha256=hashlib.sha256(cccl_header.read_bytes()).hexdigest()))
    try:
        def command(argv, name, expect=0):
            result = subprocess.run(argv, text=True, capture_output=True, timeout=90)
            (work / (name + '.stdout')).write_text(result.stdout)
            (work / (name + '.stderr')).write_text(result.stderr)
            (work / (name + '.command.json')).write_text(json.dumps(argv, indent=2) + '\n')
            assert result.returncode == expect, (name, result.returncode, result.stderr[-6000:])
            return result

        host_flags = [args.cxx, '-std=c++17', '-O2', '-fno-fast-math',
                      '-fsanitize=undefined', '-fno-sanitize-recover=undefined',
                      '-D__aicore__=', '-D__global__=', '-D__device__=', '-D__ubuf__=',
                      '-D__NPU_ARCH__=3510', '-DASCENDC_CPU_DEBUG=1',
                      '-I' + str(ROOT / 'tests/rewrite/scalar64_stubs'),
                      '-I' + str(ROOT / 'tests/rewrite/stubs'),
                      '-I' + str(ROOT / 'include'), '-I' + str(ROOT / 'tests/rewrite')]
        for case in fixtures():
            label = case['name']
            source, output = work / (label + '.cu'), work / (label + '.cce')
            source.write_text(case['source'])
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            syntax = [args.clang, '-x', 'cuda', '--cuda-host-only', '-nocudalib',
                      '--cuda-path=' + args.cuda_path, '-Wno-unknown-cuda-version',
                      '-std=' + case['standard'], '-fsyntax-only', '-fgpu-rdc', *sdk_flags, str(source)]
            command(syntax, label + '.original')
            convert = [args.binary, str(source), '--target-policy=dav-c310-vec',
                       '--simt-math=precise', '--default-preprocessor',
                       '--frontend-compat=ascify-admitted-v1', '--cuda-path=' + args.cuda_path,
                       '--clang-resource-directory=' + args.resource_dir, *case['flags'],
                       '-o', str(output), '--', '-x', 'cuda', '-std=' + case['standard'], '-fgpu-rdc', *sdk_flags]
            result = subprocess.run(convert, text=True, capture_output=True, timeout=90)
            (work / (label + '.convert.stdout')).write_text(result.stdout)
            (work / (label + '.convert.stderr')).write_text(result.stderr)
            (work / (label + '.convert.command.json')).write_text(json.dumps(convert, indent=2) + '\n')
            assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, label + ': input mutated'
            if not case['accepted']:
                assert result.returncode != 0 and not output.exists(), (label, result.returncode,
                    result.stderr[-6000:], output.read_text() if output.exists() else '')
            else:
                assert result.returncode == 0 and output.exists(), (label, result.stderr[-6000:])
                text = output.read_text()
                assert text.count(ADD) == case['adds'], (label, 'addition count', text)
                assert text.count(ASSIGN) == case['assigns'], (label, 'assignment count', text)
                if case['guard']:
                    assert 'Ascify scalar64 requires the proven closed builtin dtype domain' in text, label
                for retained in case['retained']:
                    assert retained in text, (label, retained, text)
                assert 'double' in text or label.startswith('raw_') or label == 'int_float_retained', label
                if case['cpu']:
                    driver, executable = work / (label + '.host.cpp'), work / (label + '.host')
                    body = case['cpu']
                    if label == 'host_retained':
                        # Hardware host NaN selection is outside the retained-source check.
                        body = '''for(const auto& t:scalar64_cases){
if((t.left&UINT64_C(0x7ff0000000000000))==UINT64_C(0x7ff0000000000000)||
   (t.right&UINT64_C(0x7ff0000000000000))==UINT64_C(0x7ff0000000000000))continue;
scalar64_expect(scalar64_raw(host_pair(scalar64_from_raw<double>(t.left),scalar64_from_raw<double>(t.right))),t.expected);
}'''
                    driver.write_text('#include <initializer_list>\n#include "scalar64_oracle.hpp"\n'
                                      '#include "' + output.name + '"\nint main(){\n' + body + '\n}\n')
                    command(host_flags + [str(driver), '-o', str(executable)], label + '.host-build')
                    command([str(executable)], label + '.host-run')
            summary['cases'].append(dict(name=label, source_sha256=digest,
                                         accepted=case['accepted'], cpu_checked=bool(case['cpu'])))
            print(label + ': ' + ('accepted and CPU checked' if case['accepted'] else 'refused without publication'))

        executable = work / 'scalar64_compat_host_test'
        command(host_flags + [str(ROOT / 'tests/rewrite/scalar64_compat_host_test.cpp'),
                             '-o', str(executable)], 'compat.host-build')
        command([str(executable)], 'compat.host-run')
        for label in ['invalid-width', 'mask-mismatch', 'index-partial', 'index-range', 'inactive']:
            result = subprocess.run([str(executable), label], text=True, capture_output=True, timeout=30)
            assert result.returncode in {-signal.SIGILL, -signal.SIGTRAP}, (label, result.returncode)
        # Compile-only, negative target contracts use the same product header,
        # not text matching or a substitute implementation.
        for label, body in [
            ('pod', 'struct P{uint64_t x;};void f(){(void)ascify::detail::scalar64_to_bits(P{});}'),
            ('plus', 'void f(){(void)ascify_cg::plus<long double>();}'),
            ('up', 'void f(){auto t=ascify_cg::tiled_partition<32>(ascify_cg::this_thread_block());(void)t.shfl_up(1.0,1);}'),
            ('xor', 'void f(){auto t=ascify_cg::tiled_partition<32>(ascify_cg::this_thread_block());(void)t.shfl_xor(1.0,1);}'),
            ('unverified_target', 'void f(){(void)ascify::detail::scalar_add_rne<double>(1.0,1.0);}'),
            ('unverified_raw_down', 'void f(){(void)ascify::shfl_down_sync(~0U,1.0,1,32);}'),
            ('unverified_raw_index', 'void f(){(void)ascify::shfl_sync(~0U,1.0,1,32);}'),
        ]:
            source = work / ('reject_' + label + '.cpp')
            source.write_text('#include <ascify/ascify_cuda_compat.hpp>\n'
                              '#include <ascify/cooperative_groups_compat.hpp>\n' + body + '\n')
            flags = [flag for flag in host_flags if not label.startswith('unverified_') or flag != '-D__NPU_ARCH__=3510']
            result = subprocess.run(flags + ['-fsyntax-only', str(source)], text=True,
                                    capture_output=True, timeout=60)
            (work / (label + '.target-refusal.stderr')).write_text(result.stderr)
            assert result.returncode != 0, label + ': unsupported contract compiled'
        summary['passed'] = True
        summary['accepted'] = sum(c['accepted'] for c in summary['cases'])
        summary['refused'] = len(summary['cases']) - summary['accepted']
        summary['target_compile_refusals'] = 7
        summary['host_collective_refusals'] = 5
        print('scalar64: ' + str(summary['accepted']) + ' admitted/retained, ' +
              str(summary['refused']) + ' source-valid refusals; host checks only')
    finally:
        summary['external_sdk_inputs_unchanged'] = all(
            Path(item['key_header']).is_file() and
            hashlib.sha256(Path(item['key_header']).read_bytes()).hexdigest() == item['key_header_sha256']
            for item in summary['external_sdk_includes'])
        if not summary['external_sdk_inputs_unchanged']:
            summary['passed'] = False
        (work / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        if context:
            context.cleanup()
        assert summary['external_sdk_inputs_unchanged'], 'Selected SDK CCCL input changed during regression'


if __name__ == '__main__':
    main()
