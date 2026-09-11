#include <helper_cuda.h>
#include <helper_functions.h>

#ifndef MAX
#define MAX(left, right) (left > right ? left : right)
#endif

int admittedRetainedProviderFallback(int argc, const char** argv) {
  checkCudaErrors(cudaSetDevice(0));
  return findCudaDevice(argc, argv);
}
