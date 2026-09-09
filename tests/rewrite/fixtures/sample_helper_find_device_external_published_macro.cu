#include <helper_cuda.h>
#include "sample_helper_external_published_macro.h"

int rejectExternalPublishedMacroFindDevice(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
