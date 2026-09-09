#include <helper_cuda.h>

#pragma clang deprecated(M\
AX)

int rejectBareCrDeprecatedMax(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
