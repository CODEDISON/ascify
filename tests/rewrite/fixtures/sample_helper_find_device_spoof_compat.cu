#include <ascify/ascify_cuda_compat.hpp>
#include <helper_cuda.h>

int rejectSpoofCompatHeader(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
