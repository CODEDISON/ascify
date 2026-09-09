#define ASCIFY_ASCIFY_CUDA_COMPAT_HPP 1
#include <helper_cuda.h>

int rejectCompatGuardCollision(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
