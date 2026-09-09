#include <helper_cuda.h>
#include <helper_functions.h>

#ifdef COMMON_HELPER_CUDA_H_
constexpr int ascifyTestHelperCudaWasIncluded = 1;
#else
constexpr int ascifyTestHelperCudaWasIncluded = 0;
#endif

int rejectDownstreamGuardObservation(int argc, const char** argv) {
  return ascifyTestHelperCudaWasIncluded + findCudaDevice(argc, argv);
}
