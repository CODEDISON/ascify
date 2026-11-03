#ifndef ASCIFY_TEST_MASKED_ACCUMULATION_WARP_H
#define ASCIFY_TEST_MASKED_ACCUMULATION_WARP_H
#include_next <simt_api/device_warp_functions.h>
#include <cstdlib>

namespace masked_accumulation_test {
inline unsigned lane = 0, calls = 0;
inline int integers[32];
inline float floats[32];
inline uint64_t words[32];
inline unsigned selected(unsigned delta, int width) {
  if (width != 32) std::abort();
  ++calls;
  const unsigned result = delta >= 32U - lane ? lane : lane + delta;
  // The target completion must never read even a discarded inactive register.
  if (!(ascify_test_legacy_active_mask & (uint32_t(1) << result))) std::abort();
  return result;
}
}
inline int asc_shfl_down(int, unsigned delta, int width) {
  return masked_accumulation_test::integers[masked_accumulation_test::selected(delta, width)];
}
inline float asc_shfl_down(float, unsigned delta, int width) {
  return masked_accumulation_test::floats[masked_accumulation_test::selected(delta, width)];
}
inline uint64_t asc_shfl_down(uint64_t, unsigned delta, int width) {
  return masked_accumulation_test::words[masked_accumulation_test::selected(delta, width)];
}
double asc_shfl_down(double, unsigned, int) = delete;
#endif
