#include <helper_cuda.h>

// A matching name/signature cannot borrow the SDK's trusted declaration set.
extern __host__ cudaError_t cudaGetDeviceProperties_v2(cudaDeviceProp*, int);
void rejectedUserRedeclaration(cudaDeviceProp* properties) {
  checkCudaErrors(cudaGetDeviceProperties_v2(properties, 0));
}
