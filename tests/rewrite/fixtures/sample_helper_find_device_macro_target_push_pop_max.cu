#include <helper_cuda.h>

#define ASCIFY_TEST_MAX_NAME "MAX"
#pragma push_macro(ASCIFY_TEST_MAX_NAME)
#include <helper_functions.h>
#pragma pop_macro(ASCIFY_TEST_MAX_NAME)

int rejectMacroTargetMaxStack(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
