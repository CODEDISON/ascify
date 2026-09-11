#ifndef ASCIFY_TEST_STUB_DEVICE_WARP_FUNCTIONS_H
#define ASCIFY_TEST_STUB_DEVICE_WARP_FUNCTIONS_H

#include <stdint.h>

#define __SIMT_DEVICE_FUNCTIONS_DECL__
inline uint32_t asc_activemask() { return UINT32_MAX; }
inline uint32_t asc_ballot(int predicate) { return predicate ? UINT32_MAX : 0; }

inline int ascify_test_legacy_warp_reduce_add_calls = 0;
inline int ascify_test_legacy_warp_reduce_max_calls = 0;
inline int ascify_test_legacy_warp_reduce_min_calls = 0;
inline int ascify_test_legacy_warp_shuffle_calls = 0;

inline float asc_reduce_add(float value) {
  ++ascify_test_legacy_warp_reduce_add_calls;
  return value;
}

inline int32_t asc_reduce_add(int32_t value) {
  ++ascify_test_legacy_warp_reduce_add_calls;
  return value;
}

inline uint32_t asc_reduce_add(uint32_t value) {
  ++ascify_test_legacy_warp_reduce_add_calls;
  return value;
}

inline float asc_reduce_max(float value) {
  ++ascify_test_legacy_warp_reduce_max_calls;
  return value;
}

inline int32_t asc_reduce_max(int32_t value) {
  ++ascify_test_legacy_warp_reduce_max_calls;
  return value;
}

inline uint32_t asc_reduce_max(uint32_t value) {
  ++ascify_test_legacy_warp_reduce_max_calls;
  return value;
}

inline float asc_reduce_min(float value) {
  ++ascify_test_legacy_warp_reduce_min_calls;
  return value;
}

inline int32_t asc_reduce_min(int32_t value) {
  ++ascify_test_legacy_warp_reduce_min_calls;
  return value;
}

inline uint32_t asc_reduce_min(uint32_t value) {
  ++ascify_test_legacy_warp_reduce_min_calls;
  return value;
}

template<typename T>
inline T asc_shfl(T value, int, int) {
  ++ascify_test_legacy_warp_shuffle_calls;
  return value;
}

template<typename T>
inline T asc_shfl_up(T value, unsigned int, int) {
  ++ascify_test_legacy_warp_shuffle_calls;
  return value;
}

template<typename T>
inline T asc_shfl_down(T value, unsigned int, int) {
  ++ascify_test_legacy_warp_shuffle_calls;
  return value;
}

template<typename T>
inline T asc_shfl_xor(T value, int, int) {
  ++ascify_test_legacy_warp_shuffle_calls;
  return value;
}

#endif
