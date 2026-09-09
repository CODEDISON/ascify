#include <helper_cuda.h>

#pragma /*
*/ push_macro("MAX")
#include <helper_functions.h>
#pragma /*
*/ pop_macro("MAX")

int rejectCommentSplicedMaxStack(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
