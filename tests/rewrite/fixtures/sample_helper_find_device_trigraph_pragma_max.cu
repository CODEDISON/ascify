#include <helper_cuda.h>

#pragma pu??/
sh_macro("MAX")
#include <helper_functions.h>
#pragma po??/
p_macro("MAX")

int rejectTrigraphMaxStack(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
