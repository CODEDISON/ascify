#include <cuda_fp16.h>

__device__ half2 UnprovenCall();
__global__ void RejectArithmetic(half2* output, const half2* input) {
  output[0] = input[0] + UnprovenCall();
}
