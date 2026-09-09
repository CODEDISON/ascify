#include <helper_cuda.h>

int ascify = 0;

int rejectAscifyDeclarationCollision(int argc, const char** argv) {
  return findCudaDevice(argc, argv) + ascify;
}
