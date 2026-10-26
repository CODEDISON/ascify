/*
License for Berkeley SoftFloat Release 3e

John R. Hauser
2018 January 20

The following applies to the whole of SoftFloat Release 3e as well as to
each source file individually.

Copyright 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018 The Regents of the
University of California.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions, and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// C++ adaptation of the pinned SoftFloat 3e F64 add closure.
// Fixed RNE; 8086-SSE NaN payload policy; no flags/global state. Integer-only binary64 RNE add backend; no other FP64 operation is implied.

#ifndef ASCIFY_BINARY64_RNE_V1_HPP
#define ASCIFY_BINARY64_RNE_V1_HPP
#include <cstdint>

// The target TUs define this to the already verified SIMT function qualifiers.
// The separate CPU source leaves it as ordinary inline, with no CANN headers.
#ifndef ASCIFY_BINARY64_INLINE
#define ASCIFY_BINARY64_INLINE inline
#endif

namespace ascify { namespace detail { namespace binary64_rne_v1 {

ASCIFY_BINARY64_INLINE bool sign(uint64_t value) { return (value >> 63) != 0; }
ASCIFY_BINARY64_INLINE int32_t exponent(uint64_t value) { return int32_t((value >> 52) & 0x7ff); }
ASCIFY_BINARY64_INLINE uint64_t fraction(uint64_t value) { return value & UINT64_C(0x000fffffffffffff); }
ASCIFY_BINARY64_INLINE uint64_t pack(bool negative, int32_t exp, uint64_t sig) {
  // As in SoftFloat, addition is intentional: sig may carry an implicit bit
  // into the packed exponent. Every caller passes a nonnegative exponent.
  return (uint64_t(negative) << 63) + (uint64_t(exp) << 52) + sig;
}
ASCIFY_BINARY64_INLINE bool is_nan(uint64_t value) {
  return exponent(value) == 0x7ff && fraction(value) != 0;
}
ASCIFY_BINARY64_INLINE uint64_t propagate_nan(uint64_t left, uint64_t right) {
  // 8086-SSE specialization, with exception-flag effects omitted: quiet the
  // first NaN operand. An existing NaN's sign/payload otherwise survive.
  return (is_nan(left) ? left : right) | UINT64_C(0x0008000000000000);
}
ASCIFY_BINARY64_INLINE uint64_t shift_right_jam(uint64_t value, uint32_t distance) {
  // Original primitive requires distance > 0. Extend only its zero-distance
  // case, and guard every shift so no path shifts a 64-bit value by >= 64.
  if (!distance) return value;
  if (distance >= 64) return value != 0;
  return (value >> distance) | uint64_t((value << (64 - distance)) != 0);
}
ASCIFY_BINARY64_INLINE int32_t leading_zeros(uint64_t value) {
  // Equivalent normalization primitive, using six fixed integer tests instead
  // of a lookup table or an unproved compiler/runtime clz intrinsic.
  if (!value) return 64;
  int32_t count = 0;
  if (!(value >> 32)) { count += 32; value <<= 32; }
  if (!(value >> 48)) { count += 16; value <<= 16; }
  if (!(value >> 56)) { count += 8; value <<= 8; }
  if (!(value >> 60)) { count += 4; value <<= 4; }
  if (!(value >> 62)) { count += 2; value <<= 2; }
  if (!(value >> 63)) ++count;
  return count;
}
ASCIFY_BINARY64_INLINE uint64_t round_pack(bool negative, int32_t exp, uint64_t sig) {
  uint32_t round_bits = uint32_t(sig & 0x3ff);
  if (exp < 0) {
    sig = shift_right_jam(sig, uint32_t(-exp));
    exp = 0;
    round_bits = uint32_t(sig & 0x3ff);
  } else if (exp > 0x7fd ||
             (exp == 0x7fd && sig + 0x200 >= UINT64_C(0x8000000000000000))) {
    return pack(negative, 0x7ff, 0);
  }
  sig = (sig + 0x200) >> 10;
  if (round_bits == 0x200) sig &= ~UINT64_C(1);
  if (!sig) exp = 0;
  return pack(negative, exp, sig);
}
ASCIFY_BINARY64_INLINE uint64_t normalize_round_pack(bool negative, int32_t exp, uint64_t sig) {
  const int32_t shift = leading_zeros(sig) - 1;
  exp -= shift;
  if (shift >= 10 && uint32_t(exp) < 0x7fd)
    return pack(negative, sig ? exp : 0, sig << uint32_t(shift - 10));
  return round_pack(negative, exp, sig << uint32_t(shift));
}

ASCIFY_BINARY64_INLINE uint64_t add_magnitudes(uint64_t left, uint64_t right, bool negative) {
  const int32_t exp_left = exponent(left), exp_right = exponent(right);
  uint64_t sig_left = fraction(left), sig_right = fraction(right);
  const int32_t difference = exp_left - exp_right;
  int32_t exp_result;
  uint64_t sig_result;
  if (!difference) {
    if (!exp_left) return left + sig_right;
    if (exp_left == 0x7ff)
      return (sig_left | sig_right) ? propagate_nan(left, right) : left;
    exp_result = exp_left;
    sig_result = (UINT64_C(0x0020000000000000) + sig_left + sig_right) << 9;
  } else {
    sig_left <<= 9;
    sig_right <<= 9;
    if (difference < 0) {
      if (exp_right == 0x7ff)
        return sig_right ? propagate_nan(left, right) : pack(negative, 0x7ff, 0);
      exp_result = exp_right;
      if (exp_left) sig_left += UINT64_C(0x2000000000000000);
      else sig_left <<= 1;
      sig_left = shift_right_jam(sig_left, uint32_t(-difference));
    } else {
      if (exp_left == 0x7ff) return sig_left ? propagate_nan(left, right) : left;
      exp_result = exp_left;
      if (exp_right) sig_right += UINT64_C(0x2000000000000000);
      else sig_right <<= 1;
      sig_right = shift_right_jam(sig_right, uint32_t(difference));
    }
    sig_result = UINT64_C(0x2000000000000000) + sig_left + sig_right;
    if (sig_result < UINT64_C(0x4000000000000000)) {
      --exp_result;
      sig_result <<= 1;
    }
  }
  return round_pack(negative, exp_result, sig_result);
}

ASCIFY_BINARY64_INLINE uint64_t subtract_magnitudes(uint64_t left, uint64_t right, bool negative) {
  int32_t exp_left = exponent(left);
  const int32_t exp_right = exponent(right);
  uint64_t sig_left = fraction(left), sig_right = fraction(right);
  const int32_t difference = exp_left - exp_right;
  if (!difference) {
    if (exp_left == 0x7ff) {
      if (sig_left | sig_right) return propagate_nan(left, right);
      return UINT64_C(0xfff8000000000000);  // 8086-SSE default invalid NaN.
    }
    if (sig_left == sig_right) return 0;  // Exact cancellation: +0 in RNE.
    // Avoid the upstream unsigned-to-signed implementation-defined conversion.
    uint64_t sig_difference;
    if (sig_left < sig_right) {
      negative = !negative;
      sig_difference = sig_right - sig_left;
    } else sig_difference = sig_left - sig_right;
    if (exp_left) --exp_left;
    int32_t shift = leading_zeros(sig_difference) - 11;
    int32_t exp_result = exp_left - shift;
    if (exp_result < 0) { shift = exp_left; exp_result = 0; }
    return pack(negative, exp_result, sig_difference << uint32_t(shift));
  }
  sig_left <<= 10;
  sig_right <<= 10;
  int32_t exp_result;
  uint64_t sig_result;
  if (difference < 0) {
    negative = !negative;
    if (exp_right == 0x7ff)
      return sig_right ? propagate_nan(left, right) : pack(negative, 0x7ff, 0);
    sig_left += exp_left ? UINT64_C(0x4000000000000000) : sig_left;
    sig_left = shift_right_jam(sig_left, uint32_t(-difference));
    sig_right |= UINT64_C(0x4000000000000000);
    exp_result = exp_right;
    sig_result = sig_right - sig_left;
  } else {
    if (exp_left == 0x7ff) return sig_left ? propagate_nan(left, right) : left;
    sig_right += exp_right ? UINT64_C(0x4000000000000000) : sig_right;
    sig_right = shift_right_jam(sig_right, uint32_t(difference));
    sig_left |= UINT64_C(0x4000000000000000);
    exp_result = exp_left;
    sig_result = sig_left - sig_right;
  }
  return normalize_round_pack(negative, exp_result - 1, sig_result);
}

ASCIFY_BINARY64_INLINE uint64_t add_rne_bits(uint64_t left, uint64_t right) {
  const bool negative = sign(left);
  return negative == sign(right) ? add_magnitudes(left, right, negative)
                                 : subtract_magnitudes(left, right, negative);
}

} } } // namespace ascify::detail::binary64_rne_v1
#endif
