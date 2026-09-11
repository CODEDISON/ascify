#ifndef ASCIFY_HALF2_COMPAT_HPP
#define ASCIFY_HALF2_COMPAT_HPP

#include <simt_api/asc_fp16.h>

namespace ascify {

// Keep source constructor lookup independent of user make_half2 overloads.
__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 make_half2_from_halves(
    half low, half high) {
  half2 result;
  result.x = low;
  result.y = high;
  return result;
}

// CANN 9.1 supplies an explicit two-component round-to-nearest conversion.
// CUDA's scalar broadcast applies that same conversion to both components.
__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 float2half2_rn(float value) {
  return __floats2half2_rn(value, value);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline float2 half22float2(half2 value) {
  float2 result;
  result.x = __low2float(value);
  result.y = __high2float(value);
  return result;
}

}  // namespace ascify

#endif  // ASCIFY_HALF2_COMPAT_HPP
