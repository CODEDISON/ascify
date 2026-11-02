#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
__global__ void legacy_tile32(int* out) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<32>(block);
  int value = (int)tile.thread_rank();
  out[block.thread_rank()] = tile.shfl_down(value,1);
}
void launch_legacy(int* out, int threads) { legacy_tile32<<<1,threads>>>(out); }
