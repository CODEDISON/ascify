#define __aicore__
#define __global__
#include <ascify/ascify_cuda_compat.hpp>
#include <cassert>
#include <csignal>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

template <typename Function> static void expect_trap(Function invoke) {
  const pid_t child = fork();
  assert(child >= 0);
  if (child == 0) {
    const rlimit no_core = {0, 0};
    setrlimit(RLIMIT_CORE, &no_core);
    invoke();
    _exit(99);
  }
  int status = 0;
  assert(waitpid(child, &status, 0) == child);
  assert(WIFSIGNALED(status));
  assert(WTERMSIG(status) == SIGILL || WTERMSIG(status) == SIGTRAP);
}

int main() {
  assert(ascify::shfl_up_sync(UINT32_MAX, int32_t{-7}, 1, 8) == -7);
  assert(ascify::shfl_up_sync(UINT32_MAX, uint32_t{0x80000001U}, 1, 8) == 0x80000001U);
  assert(ascify::shfl_up_sync(UINT32_MAX, 2.5f, 1, 8) == 2.5f);
#if defined(ASCIFY_TEST_EXPECT_PUBLIC_85)
  static_assert(ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85 == 1, "test must select public85");
  assert(AscendC::Simt::ascify_test_public_warp_shuffle_calls == 3);
  expect_trap([] { (void)ascify::shfl_up_sync(0xffU, 7, 1, 8); });
  expect_trap([] { (void)ascify::shfl_up_sync(UINT32_MAX, 7, 0, 1); });
#else
  static_assert(ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3 == 1, "test must select legacy");
  assert(ascify_test_legacy_warp_shuffle_calls == 3);
  assert(ascify::shfl_up_sync(UINT32_MAX, int64_t{-0x100000001LL}, 1, 32) == -0x100000001LL);
  assert(ascify::shfl_up_sync(UINT32_MAX, uint64_t{0x8000000100000001ULL}, 1, 32) == 0x8000000100000001ULL);
  ascify_test_legacy_active_mask = 0xffU;
  assert(ascify::shfl_up_sync(0xffU, int32_t{-9}, UINT32_MAX, 8) == -9);
  assert(ascify_test_legacy_up_delta == 0 && ascify_test_legacy_up_width == 8);
  assert(ascify::shfl_up_sync(0xffU, uint32_t{0xffffffffU}, 7, 8) == UINT32_MAX);
  assert(ascify_test_legacy_up_delta == 7);
  assert(ascify::shfl_up_sync(0xffU, -0.0f, 8, 8) == 0.0f);
  assert(ascify_test_legacy_up_delta == 0);
  assert(ascify_test_legacy_warp_shuffle_calls == 8);
  expect_trap([] { (void)ascify::shfl_up_sync(0xffU, int64_t{3}, 1, 8); });
  expect_trap([] { (void)ascify::shfl_up_sync(0xffU, uint64_t{3}, 1, 8); });
  expect_trap([] { (void)ascify::shfl_up_sync(0xffU, 3.0, 1, 8); });
  expect_trap([] { (void)ascify::shfl_up_sync(0xffU, 3LL, 1, 8); });
  expect_trap([] { (void)ascify::shfl_up_sync(0xffU, 7, 0, 1); });
  expect_trap([] { (void)ascify::shfl_up_sync(0xffffU, 7, 1, 8); });
#endif
}
