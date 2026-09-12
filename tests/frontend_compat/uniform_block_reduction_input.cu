#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

template <typename T, typename Group>
__device__ T forward_sum(T value, Group& group) {
  return cg::reduce(group, value, cg::plus<T>());
}

template <typename T, unsigned BlockSize, unsigned GroupSize>
__global__ void uniform_sum(const T* input, T* output) {
  __shared__ cg::block_tile_memory<BlockSize> scratch;
  auto block = cg::this_thread_block(scratch);
  auto tile = cg::tiled_partition<GroupSize>(block);
  T value = input[block.thread_rank()];
  value = forward_sum(value, tile);
  output[block.thread_rank()] = value;
  value = forward_sum(value, tile);
  output[BlockSize + block.thread_rank()] = value;
}
template __global__ void uniform_sum<int, 128, 32>(const int*, int*);
template __global__ void uniform_sum<int, 128, 64>(const int*, int*);
template __global__ void uniform_sum<float, 128, 64>(const float*, float*);
template __global__ void uniform_sum<float, 256, 128>(const float*, float*);
template __global__ void uniform_sum<float, 512, 256>(const float*, float*);
template __global__ void uniform_sum<float, 1024, 512>(const float*, float*);
