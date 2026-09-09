#include <helper_cuda.h>
#include "sample_helper_external_annotation_alias.h"

int rejectExternalAnnotationAliasFindDevice(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
