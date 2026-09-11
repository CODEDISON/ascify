#include <helper_cuda.h>
#include <helper_functions.h>
#ifndef MAX
#define MAX(left, right) (left > right ? left : right)
#endif
#define STRINGIZE_BODY_(x) #x
#define STRINGIZE_BODY(x) STRINGIZE_BODY_(x)
const char* observedMacroBody = STRINGIZE_BODY(MAX(1, 2));
int rejectedLaterExpansion(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
