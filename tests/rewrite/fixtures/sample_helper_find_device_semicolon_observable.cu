#include <helper_cuda.h>
#include <helper_functions.h>
#pragma push_macro("MAX");
#pragma pop_macro("MAX");

int observeHelperMacroWithSemicolon(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
