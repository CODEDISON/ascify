#include <helper_cuda.h>

#pragma push_macro("MAX")
#include <helper_functions.h>
#pragma pop_macro("MAX")

int rejectMaxMacroStackObservation(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
