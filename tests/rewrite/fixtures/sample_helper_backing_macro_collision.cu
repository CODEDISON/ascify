#define sampleCheckCudaErrors(...) 0
#include <helper_cuda.h>

void rejectBackingMacroCollision(void** pointer) {
  checkCudaErrors(cudaMalloc(pointer, 16));
}
