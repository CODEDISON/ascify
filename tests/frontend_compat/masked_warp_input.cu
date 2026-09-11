#include <cuda_runtime.h>

__global__ void MaskedWarpInput(const float* values, float* shuffled,
                                unsigned int* ballots, unsigned int mask,
                                int predicate, unsigned int delta, int width) {
  const unsigned int rank = threadIdx.x;
  const unsigned int lane = rank & 31u;
  if ((mask & (1u << lane)) != 0) {
    ballots[rank] = __ballot_sync(mask, predicate && (lane & 1u));
    shuffled[rank] = __shfl_down_sync(mask, values[rank], delta, width);
  }
}
