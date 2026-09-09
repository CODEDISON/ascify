#define RuntimeLock ascify_user_runtime_lock
#include <helper_cuda.h>

int rejectActiveCompatMacroFindDevice(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
