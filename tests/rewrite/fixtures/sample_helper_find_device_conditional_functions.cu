#include <helper_cuda.h>

#ifdef COMMON_HELPER_CUDA_H_
#include <helper_functions.h>
#endif

int rejectConditionalFunctionsFindDevice(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
