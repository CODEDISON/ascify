// Keep positive inputs free of inactive helper uses: the raw-token contract
// deliberately retains such uses instead of interpreting conditional states.
#include <helper_cuda.h>

// The canonical declaration uses U; the emitted definition must use T.
template<class U>
void template_memory(U*, U*, size_t, int, const char**);

template<class T>
void template_memory(T* destination, T* source, size_t bytes,
                     int argc, const char** argv) {
  int device;
  device = findCudaDevice(argc, argv);
  (void)device;
  checkCudaErrors(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaFree(destination));
  getLastCudaError("template helper fixture");
}

void instantiate_memory(int argc, const char** argv) {
  template_memory<float>(nullptr, nullptr, 0, argc, argv);
#if ASCIFY_TEMPLATE_CASE == 0
  template_memory<int>(nullptr, nullptr, 0, argc, argv);
#endif
}
