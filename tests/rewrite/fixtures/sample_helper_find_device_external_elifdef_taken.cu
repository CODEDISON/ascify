#include <helper_cuda.h>
#include "sample_helper_external_elifdef_taken.h"

int rejectExternalElifdefMax(int argc, const char** argv) {
  return ascifyTestExternalElifdefMax + findCudaDevice(argc, argv);
}
