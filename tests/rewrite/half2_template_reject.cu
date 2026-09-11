#include <cuda_fp16.h>

template <class T> __device__ T GenericAdd(T a, T b) { return a + b; }
__global__ void RejectSharedTemplateSource(half2* output, float* scalar,
                                         const half2* a, const half2* b) {
  output[0] = GenericAdd(a[0], b[0]);
  scalar[0] = GenericAdd(1.0f, 2.0f);
}
