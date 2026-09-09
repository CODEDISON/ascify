#define ascify ascify_user_macro
#include <helper_cuda.h>

int rejectAscifyMacroCollision(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
