#ifndef ASCIFY_TEST_MASKED_WARP_STUBS_H
#define ASCIFY_TEST_MASKED_WARP_STUBS_H

#include <cassert>
#include <stdint.h>

#define __SIMT_DEVICE_FUNCTIONS_DECL__

// A host model for guard and routing checks only. It does not model device
// scheduling, convergence, memory ordering, or the implementation of intrinsics.
namespace masked_warp_test {
inline uint32_t active;
inline uint32_t masks[32];
inline int widths[32];
inline bool predicates[32];
inline unsigned lane;
inline unsigned ballot_phase;
inline bool shuffle_mode;
inline bool shuffled;

template <typename T> T value(unsigned source_lane) {
  return static_cast<T>(1000 + source_lane);
}

template <typename T> T undefined_source_value() {
  return static_cast<T>(9137);
}

inline bool valid_width(int width) {
  return width == 1 || width == 2 || width == 4 || width == 8 ||
         width == 16 || width == 32;
}
}  // namespace masked_warp_test

inline uint32_t asc_activemask() { return masked_warp_test::active; }

inline uint32_t asc_ballot(int32_t predicate) {
  using namespace masked_warp_test;
  assert(ballot_phase < 2);
  uint32_t result = 0;
  for (unsigned i = 0; i < 32; ++i) {
    const bool current = ballot_phase == 0
        ? masks[i] != active || active == 0
        : shuffle_mode ? !valid_width(widths[i]) : predicates[i];
    if (i == lane) assert((predicate != 0) == current);
    if ((active & (uint32_t{1} << i)) != 0 && current)
      result |= uint32_t{1} << i;
  }
  ++ballot_phase;
  return result;
}

template <typename T>
inline T asc_shfl_down(T local_value, unsigned int delta, int width) {
  using namespace masked_warp_test;
  assert(ballot_phase == 2);
  assert(local_value == value<T>(lane));
  assert(width == widths[lane]);
  shuffled = true;
  unsigned source = lane + delta;
  if (source >= (lane / static_cast<unsigned>(width) + 1U) * width)
    source = lane;
  if ((active & (uint32_t{1} << source)) == 0)
    return undefined_source_value<T>();
  return value<T>(source);
}

#endif
