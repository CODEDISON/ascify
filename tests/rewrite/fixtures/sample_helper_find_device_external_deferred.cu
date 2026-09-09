#include <helper_cuda.h>
#include "sample_helper_external_deferred_find.h"

int instantiateExternalFindDevice(int argc, const char** argv) {
  return deferredExternalFindDevice(argc, argv);
}
