#include <helper_cuda.h>
#include <helper_functions.h>

int acceptCudaThenFunctionsFindDevice(int argc, const char** argv) {
  checkCudaErrors(cudaSetDevice(0));
  getLastCudaError("cuda then helper functions");
  return findCudaDevice(argc, argv);
}
