#include <helper_cuda.h>

void admittedCurrentStatusDomains(cudaDeviceProp* properties, int& symbol) {
  checkCudaErrors(cudaGetDeviceProperties(properties, 0));
  checkCudaErrors(cudaMemcpyToSymbol(symbol, properties, sizeof(int)));
}
