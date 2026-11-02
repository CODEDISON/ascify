#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
__device__ int closed_sum(cg::thread_group group, int* storage, int value) {
  int rank = (int)group.thread_rank();
  for (int step = (int)(group.size() / 2); step > 0; step /= 2) {
    storage[rank] = value;
    group.sync();
    if (rank < step) value += storage[rank + step];
    group.sync();
  }
  return rank == 0 ? value : -1;
}
[[noreturn]] __device__ void stop_marked() { __trap(); }
__global__ void fullwarp_groups(int* out) {
  extern __shared__ int workspace[];
  cg::thread_block block = cg::this_thread_block();
  int block_total = closed_sum(block, workspace, block.thread_rank());
  if (block.thread_rank() == 0) out[4] = block_total;
  block.sync();
  cg::thread_block_tile<16> tile = cg::tiled_partition<16>(block);
  int offset = block.thread_rank() - tile.thread_rank();
  stop_marked();
  int tile_total = closed_sum(tile, workspace + offset, tile.thread_rank());
  if (tile.thread_rank() == 0) out[block.thread_rank() / 16] = tile_total;
}
void launch_groups(int* out) {
  int threads = 64;
  fullwarp_groups<<<1, threads, threads * sizeof(int)>>>(out);
}
