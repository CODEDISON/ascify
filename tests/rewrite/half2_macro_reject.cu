#include <cuda_fp16.h>

#define HALF2_READ(pointer) pointer[threadIdx.x]
__global__ void RejectMacroLeaf(half2* output, const half2* a, const half2* b) {
  output[0] = HALF2_READ(a) + b[0];
}
