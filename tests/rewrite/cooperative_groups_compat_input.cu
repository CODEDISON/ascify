#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void CooperativeBlock(float* output) {
  const cg::thread_block block = cg::this_thread_block();
  output[threadIdx.x] = static_cast<float>(threadIdx.x);
  cg::sync(block);
}

__global__ void CooperativeTile(float* output) {
  const auto block = cg::this_thread_block();
  const auto typed = cg::tiled_partition<32>(block);
  const cg::thread_block_tile<32> erased = typed;
  output[threadIdx.x] = static_cast<float>(threadIdx.x);
  typed.sync();
  cg::sync(typed);
  erased.sync();
  cg::sync(erased);
}
