#include <helper_cuda.h>
#include <helper_image.h>

int rejectCudaThenImageFindDevice(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
