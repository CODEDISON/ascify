#include <ascify/masked_warp_compat.hpp>
#include <csignal>
#include <cstdio>
#include <initializer_list>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

namespace model = masked_warp_test;

static void reset(uint32_t mask, int width) {
  model::active = mask;
  model::up_mode = true;
  model::shuffle_mode = true;
  model::shuffled = false;
  model::ballot_phase = 0;
  model::native_up_calls = 0;
  model::lane = 0;
  // Exact transport, including float signed zero, infinities, quiet NaN
  // payloads and subnormals. This test never performs float arithmetic.
  const uint32_t patterns[] = {0, 0x80000000U, 0x7f800000U, 0xff800000U,
      0x7fc12345U, 0xffc54321U, 1, 0x80000001U, 0x7fffffffU,
      0xffffffffU, 0xaaaaaaaaU, 0x55555555U, 0x3f800000U, 0xbf800000U};
  for (unsigned i = 0; i < 32; ++i) {
    model::masks[i] = mask;
    model::widths[i] = width;
    model::lane_bits[i] = patterns[i % (sizeof(patterns) / sizeof(patterns[0]))];
  }
}

template <typename T>
static void check(uint32_t mask, unsigned lane, unsigned delta, int width) {
  reset(mask, width);
  model::lane = lane;
  const T got = ascify::detail::converged_shfl_up_sync(
      mask, model::up_value<T>(lane), delta, width);
  const unsigned offset = lane % static_cast<unsigned>(width);
  const unsigned source = delta > offset ? lane : lane - delta;
  const uint32_t expected = (mask & (uint32_t{1} << source)) != 0
      ? model::lane_bits[source] : model::bits(model::undefined_source_value<T>());
  assert(model::bits(got) == expected);
  // Even an out-of-range delta must reach the native instruction. In a real
  // warp another lane can still read this lane's input at the same instruction.
  assert(model::shuffled && model::native_up_calls == 1);
}

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

static void invoke(uint32_t mask, int width) {
  ascify::detail::converged_shfl_up_sync(mask, model::up_value<int32_t>(model::lane), 1, width);
}

int main() {
  const uint32_t masks[] = {UINT32_MAX, 0xffffU, 0xffU, 0xffff0000U,
      0xff000000U, 0x55555555U, 0xaaaaaaaaU, 0x00f0000fU, 1, 0x80000000U};
  const int widths[] = {2, 4, 8, 16, 32};
  unsigned checks = 0, rejects = 0;
  for (uint32_t mask : masks) {
    for (int width : widths) {
      const unsigned w = static_cast<unsigned>(width);
      const unsigned deltas[] = {0, 1, w - 1, w, w + 1, 31, 32, 255, UINT32_MAX};
      for (unsigned lane = 0; lane < 32; ++lane) {
        if ((mask & (uint32_t{1} << lane)) == 0) continue;
        for (unsigned delta : deltas) {
          check<int32_t>(mask, lane, delta, width);
          check<uint32_t>(mask, lane, delta, width);
          check<float>(mask, lane, delta, width);
          checks += 3;
        }
        // Lane zero uses a huge delta while lane one reads its input. The
        // per-lane native-call assertion catches a return-before-shuffle edit.
        check<uint32_t>(mask, lane, lane % 2 == 0 ? UINT32_MAX : 1, width);
        ++checks;
      }
    }
  }
  for (int width : {0, 1, 3, 6, 33, -1}) {
    expect_trap([=] { reset(0xffU, width); invoke(0xffU, width); });
    ++rejects;
  }
  expect_trap([] { reset(0xffU, 8); model::widths[7] = 3; invoke(0xffU, 8); });
  ++rejects;
  expect_trap([] { reset(0xffU, 8); model::masks[7] = 0x7fU; invoke(0xffU, 8); });
  ++rejects;
  expect_trap([] { reset(0xffU, 8); for (auto& m : model::masks) m = 0xffffU; invoke(0xffffU, 8); });
  ++rejects;
  expect_trap([] { reset(0xffU, 8); for (auto& m : model::masks) m = 0x7fU; invoke(0x7fU, 8); });
  ++rejects;
  expect_trap([] { reset(0xffU, 8); for (unsigned i = 0; i < 32; ++i) model::masks[i] = i < 4 ? 0xfU : 0xf0U; invoke(0xfU, 8); });
  ++rejects;
  expect_trap([] { reset(0xffU, 8); for (auto& m : model::masks) m = 0; invoke(0, 8); });
  ++rejects;
  expect_trap([] { reset(0, 8); invoke(0, 8); });
  ++rejects;
  reset(0xffU, 8);
  unsigned value_calls = 0, delta_calls = 0, width_calls = 0, mask_calls = 0;
  const auto value = [&] { ++value_calls; return model::up_value<uint32_t>(0); };
  const auto delta = [&] { ++delta_calls; return UINT32_MAX; };
  const auto width = [&] { ++width_calls; return 8; };
  const auto mask = [&] { ++mask_calls; return 0xffU; };
  (void)ascify::detail::converged_shfl_up_sync(mask(), value(), delta(), width());
  assert(value_calls == 1 && delta_calls == 1 && width_calls == 1 && mask_calls == 1);
  assert(model::native_up_calls == 1);
  printf("masked_up host bit transport checks=%u rejects=%u single_evaluation=1\n", checks, rejects);
}
