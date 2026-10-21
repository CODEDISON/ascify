// Host arithmetic model for the actual product helper; no device/runtime claim.
#define __aicore__
#include <ascify/ascify_cuda_compat.hpp>
#include <cassert>
#include <cstdint>
#include <cstdio>

static std::uint64_t checked = 0;
static void check(unsigned int a, unsigned int b, unsigned int c) {
  // Independent widened arithmetic: radix division instead of helper shifts,
  // signed widened differences, and an explicit low-word mask at the end.
  std::uint64_t expected = c;
  std::uint64_t divisor = 1;
  for (unsigned int lane = 0; lane != 4; ++lane, divisor *= 256) {
    const std::int64_t x = (std::uint64_t(a) / divisor) % 256;
    const std::int64_t y = (std::uint64_t(b) / divisor) % 256;
    const std::int64_t difference = x - y;
    expected += difference < 0 ? -difference : difference;
  }
  expected &= UINT64_C(0xffffffff);
  const auto actual = ascify::packed_u8_sad4_add(a, b, c);
  if (actual != expected) {
    std::fprintf(stderr, "SAD mismatch a=%08x b=%08x c=%08x actual=%08x expected=%08llx\n",
                 a, b, c, actual, static_cast<unsigned long long>(expected));
    std::abort();
  }
  ++checked;
}

int main() {
  const unsigned int carries[] = {0u, 1u, 255u, 0x80000000u,
                                 0xfffffc03u, 0xfffffffeu, 0xffffffffu};
  // Exhaust every byte pair at each of four byte positions. Distinct upper
  // and lower padding catches accidental packed borrow or lane reversal.
  for (unsigned int lane = 0; lane != 4; ++lane)
    for (unsigned int a = 0; a != 256; ++a)
      for (unsigned int b = 0; b != 256; ++b)
        for (const auto c : carries) {
          const unsigned int mask = ~(255u << (lane * 8));
          check((0xa5963c69u & mask) | (a << (lane * 8)),
                (0xa5963c69u & mask) | (b << (lane * 8)), c);
        }
  const unsigned int words[] = {0u, 1u, 255u, 256u, 0xffffffffu,
      0x80808080u, 0x7f7f7f7fu, 0xff00ff00u, 0x00ff00ffu,
      0x01020304u, 0x04030201u, 0xff000001u, 0x010000ffu};
  for (const auto a : words)
    for (const auto b : words)
      for (const auto c : carries) check(a, b, c);
  std::uint32_t state = 0xa341316cu;
  auto next = [&]() {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
  };
  for (unsigned int index = 0; index != 100000; ++index) {
    const auto a = next(), b = next(), c = next();
    check(a, b, c);
  }
  // Explicit carry examples distinguish wraparound from saturation.
  assert(ascify::packed_u8_sad4_add(0u, 0xffffffffu, 0xffffffffu) == 1019u);
  assert(ascify::packed_u8_sad4_add(0u, 0xffffffffu, 0xfffffc04u) == 0u);
  std::printf("packed byte SAD host arithmetic: %llu comparisons passed (no NPU evidence)\n",
              static_cast<unsigned long long>(checked));
}
