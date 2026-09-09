#include <helper_cuda.h>
#include <helper_functions.h>
#include "sample_helper_external_max.h"

int rejectCudaThenFunctionsExternalMax(int argc, const char** argv) {
  return ascifyTestExternalMax(findCudaDevice(argc, argv), 0);
}
