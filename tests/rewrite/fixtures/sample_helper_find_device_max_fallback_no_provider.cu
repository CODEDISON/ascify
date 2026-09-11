#include <helper_cuda.h>
#ifndef MAX
#define MAX(left, right) (left > right ? left : right)
#endif
int rejectedFallbackWithoutProvider(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
