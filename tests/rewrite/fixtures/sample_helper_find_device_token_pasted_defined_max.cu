#include <helper_cuda.h>

#define ASCIFY_TEST_CAT_INNER(left, right) left##right
#define ASCIFY_TEST_CAT(left, right) ASCIFY_TEST_CAT_INNER(left, right)
#define ASCIFY_TEST_DEFINED_INNER(name) defined(name)
#define ASCIFY_TEST_DEFINED(name) ASCIFY_TEST_DEFINED_INNER(name)

#if ASCIFY_TEST_DEFINED(ASCIFY_TEST_CAT(M, AX))
constexpr int ascifyTestHelperMaxWasDefined = 1;
#else
constexpr int ascifyTestHelperMaxWasDefined = 0;
#endif

int rejectTokenPastedDefinedMax(int argc, const char** argv) {
  return ascifyTestHelperMaxWasDefined + findCudaDevice(argc, argv);
}
