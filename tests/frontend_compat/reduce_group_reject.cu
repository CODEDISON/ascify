#include <cooperative_groups/reduce.h>
__global__ void RejectMultiwarpReduction(int* output) {
  const auto block = cooperative_groups::this_thread_block();
  const auto group = cooperative_groups::tiled_partition<64>(block);
  output[0] = cooperative_groups::reduce(group, 1, cooperative_groups::plus<int>());
}
