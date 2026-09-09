#define COMMON_HELPER_IMAGE_H_
#include <helper_functions.h>
#undef COMMON_HELPER_IMAGE_H_

#include <helper_cuda.h>
#include <helper_image.h>

int rejectProviderAncestryBypass(int argc, const char** argv) {
  sdkDumpBin(nullptr, 0, "unused.bin");
  return findCudaDevice(argc, argv);
}
