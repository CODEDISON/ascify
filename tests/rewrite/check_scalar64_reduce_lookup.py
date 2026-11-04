"""Closed static/scratch CG forwarding and scalar64 lookup regressions; CPU only."""
from pathlib import Path
import importlib.util,sys
base=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('scalar_suite',base/'tests/rewrite/check_scalar64_lowering.py');suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite)
prefix='#include <cuda_runtime.h>\n#include <cooperative_groups/reduce.h>\n'
forward='''
template<class T,class Portion> __device__ T bridge(T value, Portion& portion){
 return cooperative_groups::reduce(portion,value,cooperative_groups::plus<T>());
}
__device__ double use_d(double value){
 auto tile=cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
 return bridge(value,tile);
}
__device__ float use_f(float value){
 auto tile=cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
 return bridge(value,tile);
}
__device__ int use_i(int value){
 auto tile=cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
 return bridge(value,tile);
}
'''
def case(name,body,accepted=True,guard=False,cpu=None):
 return dict(name=name,source=prefix+body,accepted=accepted,adds=0,assigns=0,cpu=cpu,flags=[],standard='c++17',guard=guard,retained=[])
def fixtures():
 yield case('closed_reduce_three_dtypes',forward,guard=True,cpu='''
 scalar64_test::reset();
 scalar64_expect(scalar64_raw(use_d(1.25)),scalar64_raw(40.0));
 if(scalar64_test::calls!=6||use_f(1.25f)!=40.f||use_i(2)!=64)std::abort();
 ''')
 yield case('closed_reduce_const_group',forward.replace('auto tile=','const auto tile='),guard=True,cpu='''
 scalar64_test::reset();scalar64_expect(scalar64_raw(use_d(-0.0)),UINT64_C(0x8000000000000000));
 if(scalar64_test::calls!=6)std::abort();
 ''')
 yield case('forwarder_constexpr',forward.replace('__device__ T bridge','__device__ constexpr T bridge'),False)
 yield case('forwarder_noinline',forward.replace('__device__ T bridge','__device__ __attribute__((noinline)) T bridge'),False)
 yield case('forwarder_reflection',forward.replace(' return cooperative_groups::reduce',' using Observed=decltype(portion);\n return cooperative_groups::reduce'),False)
 yield case('forwarder_result_observation',forward.replace(' return cooperative_groups::reduce(portion,value,cooperative_groups::plus<T>());',' return sizeof(cooperative_groups::reduce(portion,value,cooperative_groups::plus<T>()));'),False)
 yield case('forwarder_extra_arithmetic',forward.replace(' return cooperative_groups::reduce',' value+=T(0);\n return cooperative_groups::reduce'),False)
 yield case('forwarder_address_escape',forward+'\n__device__ auto saved = &bridge<double,cooperative_groups::thread_block_tile<32,cooperative_groups::thread_block>>;\n',False)
 yield case('forwarder_conditional_body',forward.replace(' return cooperative_groups::reduce','\n#if 1\n return cooperative_groups::reduce').replace('plus<T>());\n}', 'plus<T>());\n#endif\n}'),False)
suite.fixtures=fixtures
suite.main()
