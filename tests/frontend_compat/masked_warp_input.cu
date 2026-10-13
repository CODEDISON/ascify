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

__global__ void MaskedUpInput(const int* iv, const unsigned int* uv,
                              const float* fv, int* io, unsigned int* uo,
                              float* fo, unsigned int mask,
                              unsigned int delta, int width) {
  const unsigned int rank = threadIdx.x;
  if ((mask & (1u << (rank & 31u))) != 0) {
    io[rank] = __shfl_up_sync(mask, iv[rank], delta, width);
    uo[rank] = __shfl_up_sync(mask, uv[rank], delta, width);
    fo[rank] = __shfl_up_sync(mask, fv[rank], delta, width);
  }
}
