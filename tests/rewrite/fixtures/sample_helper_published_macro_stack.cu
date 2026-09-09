#pragma push_macro("ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS")
#include <helper_cuda.h>
#pragma pop_macro("ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS")

void rejectPublishedMacroStack(void** pointer) {
  checkCudaErrors(cudaMalloc(pointer, 16));
}
