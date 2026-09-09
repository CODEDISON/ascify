#include <helper_cuda.h>

#define ASCIFY_TEST_PUSH_MAX _Pragma("push_macro(\"MAX\")")
#define ASCIFY_TEST_POP_MAX _Pragma("pop_macro(\"MAX\")")
ASCIFY_TEST_PUSH_MAX
#include <helper_functions.h>
ASCIFY_TEST_POP_MAX

int rejectMacroGeneratedMaxStack(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
