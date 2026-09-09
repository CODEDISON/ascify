#include <helper_cuda.h>

#pragma push_macro("ASCIFY_TEST_UNRELATED")
#define ASCIFY_TEST_UNRELATED 7
#pragma pop_macro("ASCIFY_TEST_UNRELATED")

int acceptUnrelatedMacroStackFindDevice(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
