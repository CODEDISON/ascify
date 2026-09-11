#include <ascify/masked_warp_compat.hpp>
#include <csignal>
#include <cstdio>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

namespace model = masked_warp_test;

static void reset(uint32_t active, bool shuffle_mode) {
  model::active = active;
  model::shuffle_mode = shuffle_mode;
  model::shuffled = false;
  model::ballot_phase = 0;
  model::lane = 0;
  for (unsigned i = 0; i < 32; ++i) {
    model::masks[i] = active;
    model::widths[i] = 32;
    model::predicates[i] = (i & 1U) != 0;
  }
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

template <typename T>
static void check_shuffle(uint32_t mask, unsigned lane, unsigned delta, int width) {
  reset(mask, true);
  model::lane = lane;
  for (int& item : model::widths) item = width;
  const T got = ascify::detail::converged_shfl_down_sync(
      mask, model::value<T>(lane), delta, width);
  const unsigned offset = lane % static_cast<unsigned>(width);
  const unsigned source = delta >= static_cast<unsigned>(width) - offset
      ? lane : lane + delta;
  const T expected = (mask & (uint32_t{1} << source)) != 0
      ? model::value<T>(source) : model::undefined_source_value<T>();
  assert(got == expected);
  assert(model::shuffled);
}

int main() {
  const uint32_t masks[] = {UINT32_MAX, 0xffffU, 0xffU, 0xfU,
                            0x55555555U, 0xaaaaaaaaU, 1U, 0x80000000U};
  const int widths[] = {1, 2, 4, 8, 16, 32};
  const unsigned deltas[] = {0, 1, 2, 7, 16, 31, 32};
  unsigned shuffles = 0, ballots = 0;
  for (uint32_t mask : masks) {
    for (unsigned lane = 0; lane < 32; ++lane) {
      if ((mask & (uint32_t{1} << lane)) == 0) continue;
      reset(mask, false);
      model::lane = lane;
      assert(ascify::detail::converged_ballot_sync(
          mask, model::predicates[lane] ? -7 : 0) == (mask & 0xaaaaaaaaU));
      ++ballots;
      for (int width : widths) {
        for (unsigned delta : deltas) {
          check_shuffle<int32_t>(mask, lane, delta, width);
          check_shuffle<uint32_t>(mask, lane, delta, width);
          check_shuffle<float>(mask, lane, delta, width);
          shuffles += 3;
        }
      }
    }
  }

  expect_trap([] { // All named lanes have not reached this instruction.
    reset(0xffffU, false);
    for (uint32_t& mask : model::masks) mask = UINT32_MAX;
    ascify::detail::converged_ballot_sync(UINT32_MAX, 0);
  });
  expect_trap([] { // An error in a different active lane rejects this lane too.
    reset(UINT32_MAX, false);
    model::masks[7] = 0xffffff7fU;
    ascify::detail::converged_ballot_sync(UINT32_MAX, 0);
  });
  expect_trap([] { // Concurrent disjoint masks exceed this restricted domain.
    reset(UINT32_MAX, false);
    for (unsigned i = 0; i < 32; ++i)
      model::masks[i] = i < 16 ? 0xffffU : 0xffff0000U;
    ascify::detail::converged_ballot_sync(0xffffU, 0);
  });
  expect_trap([] {
    reset(UINT32_MAX, false);
    for (uint32_t& mask : model::masks) mask = 0;
    ascify::detail::converged_ballot_sync(0, 0);
  });
  expect_trap([] { // Host-only malformed intrinsic response.
    reset(0, false);
    ascify::detail::converged_ballot_sync(0, 0);
  });
  expect_trap([] { // An invalid width in another lane must reject collectively.
    reset(UINT32_MAX, true);
    model::widths[7] = 3;
    ascify::detail::converged_shfl_down_sync(
        UINT32_MAX, model::value<int32_t>(0), 1, 32);
  });
  printf("masked_warp host guard model ballots=%u shuffles=%u rejects=6\n",
         ballots, shuffles);
}
