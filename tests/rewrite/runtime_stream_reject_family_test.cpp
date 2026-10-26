#define __aicore__
#define ASCIFY_TEST_PUBLIC_85_ACL
#include <ascify/ascify_cuda_compat.hpp>
#include <cassert>

int main() {
  int stream = 0, event = 0;
  // This family deliberately does not declare either native API or its state
  // enum in acl.h. A missing SDK gate must fail compilation before this runs.
  assert(ascify::cudaStreamQuery(&stream) == ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(ascify::cudaStreamWaitEvent(&stream, &event) == ACL_ERROR_FEATURE_UNSUPPORTED);
}
