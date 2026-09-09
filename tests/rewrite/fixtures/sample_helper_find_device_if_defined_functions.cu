#include <helper_cuda.h>

#if defined(COMMON_HELPER_CUDA_H_)
#include <helper_functions.h>
#endif

int rejectIfDefinedFunctionsFindDevice(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
