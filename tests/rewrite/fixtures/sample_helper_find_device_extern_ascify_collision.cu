#include <helper_cuda.h>

extern "C" {
int ascify = 0;
}

int rejectExternAscifyDeclarationCollision(int argc, const char** argv) {
  return findCudaDevice(argc, argv) + ascify;
}
