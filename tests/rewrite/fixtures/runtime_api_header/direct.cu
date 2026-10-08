#include <cuda_runtime_api.h>

__global__ void increment(float* values, int count) {
  const int index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  if (index < count) values[index] += 1.0f;
}

cudaError_t allocate(float** values, int count) {
  cudaError_t status = cudaSetDevice(0);
  if (status != cudaSuccess) return status;
  return cudaMalloc(values, static_cast<size_t>(count) * sizeof(float));
}

void shape(dim3* result) {
  dim3 one;
  *result = one;
}
