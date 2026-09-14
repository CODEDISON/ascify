#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

constexpr int kCount = 1024;

__global__ void add_vectors(const float *a, const float *b, float *c, int count) {
  const int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < count) {
    c[index] = a[index] + b[index];
  }
}

void check_status(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA runtime error: %s\n", cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

int main() {
  float a[kCount], b[kCount], c[kCount];
  for (int index = 0; index < kCount; ++index) {
    a[index] = 1.0f;
    b[index] = 2.0f;
  }

  float *device_a = nullptr;
  float *device_b = nullptr;
  float *device_c = nullptr;
  const size_t bytes = sizeof(a);
  check_status(cudaMalloc(&device_a, bytes));
  check_status(cudaMalloc(&device_b, bytes));
  check_status(cudaMalloc(&device_c, bytes));
  check_status(cudaMemcpy(device_a, a, bytes, cudaMemcpyHostToDevice));
  check_status(cudaMemcpy(device_b, b, bytes, cudaMemcpyHostToDevice));

  constexpr int kThreads = 256;
  const int blocks = (kCount + kThreads - 1) / kThreads;
  add_vectors<<<blocks, kThreads>>>(device_a, device_b, device_c, kCount);
  check_status(cudaGetLastError());
  check_status(cudaDeviceSynchronize());
  check_status(cudaMemcpy(c, device_c, bytes, cudaMemcpyDeviceToHost));

  check_status(cudaFree(device_a));
  check_status(cudaFree(device_b));
  check_status(cudaFree(device_c));

  for (int index = 0; index < kCount; ++index) {
    if (c[index] != 3.0f) {
      std::fprintf(stderr, "Incorrect result at index %d: %g\n", index, c[index]);
      return EXIT_FAILURE;
    }
  }
  std::printf("Vector add passed: %d FP32 elements\n", kCount);
  return EXIT_SUCCESS;
}
