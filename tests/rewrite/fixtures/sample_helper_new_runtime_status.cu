#include <helper_cuda.h>

void admittedStatusDomains(cudaDeviceProp* properties, int& symbol) {
  checkCudaErrors(cudaGetDeviceProperties(properties, 0));
  checkCudaErrors(cudaGetDeviceProperties_v2(properties, 0));
  checkCudaErrors(cudaMemcpyToSymbol(symbol, properties, sizeof(int)));
}
