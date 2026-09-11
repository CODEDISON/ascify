#include <cuda_fp16.h>

__device__ half2 make_half2(float, float);  // Must not capture constructor lowering.

__global__ void Half2Arithmetic(half2* output, const half2* a, const half2* b) {
  half2 initialized(0.0f, 1.0f);
  output[0] = __hadd2(a[0], b[0]);
  output[1] = __hfma2(a[1], b[1], initialized);
  output[2] = a[2] * b[2] + initialized;
  output[3] = __float2half2_rn(3.0f);
  output[4] = (0, a[4]) + b[4];
  output[5] = a[threadIdx.x] + b[threadIdx.x];
}

// A user-defined namesake and host constructors are outside device lowering.
namespace user {
struct half2 {
  half2(float, float);
};
void UserType() { half2 untouched(1.0f, 2.0f); }
}
void HostInitialization() { half2 host_value(1.0f, 2.0f); }
