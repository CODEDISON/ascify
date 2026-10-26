#ifndef ASCIFY_TEST_SCALAR64_WARP_H
#define ASCIFY_TEST_SCALAR64_WARP_H

// Layer the model on the established host-only SDK double. The exact uint64
// overloads distinguish bit transport from accidental floating conversion.
#include_next <simt_api/device_warp_functions.h>

namespace scalar64_test {
inline unsigned calls = 0;
inline uint64_t input = 0;
inline uint64_t output = 0;
inline unsigned delta = 0;
inline int source_lane = 0;
inline int width = 0;
inline bool supply_output = false;
inline void reset() {
  calls = 0;
  input = output = 0;
  delta = 0;
  source_lane = width = 0;
  supply_output = false;
  ascify_test_legacy_active_mask = UINT32_MAX;
}
}

inline uint64_t asc_shfl_down(uint64_t value, unsigned delta, int width) {
  ++scalar64_test::calls;
  scalar64_test::input = value;
  scalar64_test::delta = delta;
  scalar64_test::width = width;
  return scalar64_test::supply_output ? scalar64_test::output : value;
}
inline uint64_t asc_shfl(uint64_t value, int source, int width) {
  ++scalar64_test::calls;
  scalar64_test::input = value;
  scalar64_test::source_lane = source;
  scalar64_test::width = width;
  return scalar64_test::supply_output ? scalar64_test::output : value;
}
double asc_shfl_down(double, unsigned, int) = delete;
double asc_shfl(double, int, int) = delete;
#endif
