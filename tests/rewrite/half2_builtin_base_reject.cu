#include <cuda_fp16.h>

__device__ void UnprovenIndexCall();
__global__ void RejectBuiltinBase(half2* output, const half2* a, const half2* b) {
  output[0] = a[(UnprovenIndexCall(), threadIdx).x] + b[0];
}
