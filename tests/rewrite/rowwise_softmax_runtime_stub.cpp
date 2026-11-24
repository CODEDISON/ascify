#include <acl/acl.h>
#include <simt_api/asc_fp16.h>
#include <cstdint>

namespace {
int launches = 0;
}

extern "C" int test_softmax_launches() { return launches; }

#if ASCIFY_TEST_SHORT_SOFTMAX_RUNTIME
extern "C" int ascify950_softmax_supports_short_v1(
    aclrtStream, const half*, half*, int64_t, int columns) {
  return columns == 64;
}
#endif

extern "C" aclError ascify950_softmax_reg_recompute_launch_v1(
    aclrtStream, const half*, half*, int64_t rows, int columns, int, int) {
  ++launches;
  if (columns >= 4096) { return ACL_SUCCESS; }
#if ASCIFY_TEST_SHORT_SOFTMAX_RUNTIME
  if (columns == 64 && rows == 8) { return ACL_SUCCESS; }
#else
  (void)rows;
#endif
  return ACL_ERROR_RT_PARAM_INVALID;
}
