#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
__global__ void ReduceFullWarp(const float* input, float* output, int* integer_output) {
  const auto block = cg::this_thread_block();
  const auto typed = cg::tiled_partition<32>(block);
  cg::thread_block_tile<32> tile = typed;
  output[block.thread_rank()] = cg::reduce(tile, input[block.thread_rank()], cg::plus<float>());
  integer_output[block.thread_rank()] = cg::reduce(typed, int(tile.thread_rank()) - 16, cg::plus<int>());
}
