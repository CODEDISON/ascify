#include <cuda_runtime.h>
int read_device_name_and_units(int device) {
  cudaDeviceProp property;
  cudaError_t result = cudaGetDeviceProperties(&property, device);
  return result == cudaSuccess && property.name[0] ? property.multiProcessorCount : -1;
}
__global__ void full_warp_votes(const int* values, int* any, int* all) {
  unsigned int lane = threadIdx.x;
  any[lane] = __any_sync(0xffffffffU, values[lane]);
  all[lane] = __all_sync(0xffffffffU, values[lane]);
}
