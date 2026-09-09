#include <helper_cuda.h>
#include "sample_helper_external_deprecated_max.h"

int rejectExternalDeprecatedMax(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
