#include <helper_cuda.h>
#include "sample_helper_external_elifdef_skipped.h"

int rejectSkippedExternalElifdefMax(int argc, const char** argv) {
  return ascifyTestExternalElifdefMax + findCudaDevice(argc, argv);
}
