#include <sample_helper_system_pragmas.h>
#include <helper_cuda.h>
#include <helper_functions.h>

int selectWithSystemPragmas(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
