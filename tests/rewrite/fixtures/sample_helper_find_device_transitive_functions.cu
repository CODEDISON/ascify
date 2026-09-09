#include <helper_cuda.h>
#include "sample_helper_functions_wrapper.h"

int rejectTransitiveFunctionsFindDevice(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
