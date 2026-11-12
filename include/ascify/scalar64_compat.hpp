#ifndef ASCIFY_SCALAR64_COMPAT_HPP
#define ASCIFY_SCALAR64_COMPAT_HPP

#include <stdint.h>
#include <math.h>
#include <type_traits>

#if defined(ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3) || \
    defined(ASCIFY_HAS_LEGACY_COOPERATIVE_GROUPS)
#define ASCIFY_SCALAR64_LEGACY_PROVIDER 1
#include <simt_api/device_warp_functions.h>
#include <ascify/masked_warp_compat.hpp>
#define ASCIFY_SCALAR64_DEVICE __SIMT_DEVICE_FUNCTIONS_DECL__ inline __attribute__((always_inline))
#else
#define ASCIFY_SCALAR64_DEVICE __aicore__ inline __attribute__((always_inline))
#endif

// The exact backend was independently checked against Fraction and exercised
// with typed binary64 values on 950DT. Other architectures do not inherit it.
#if defined(ASCIFY_SCALAR64_LEGACY_PROVIDER) && defined(__NPU_ARCH__) && \
    __NPU_ARCH__ == 3510 && defined(__has_builtin)
#if __has_builtin(__builtin_bit_cast)
#define ASCIFY_SCALAR64_VERIFIED_TARGET 1
#ifdef ASCIFY_BINARY64_INLINE
#error "ASCIFY_BINARY64_INLINE is reserved by the scalar64 backend"
#endif
#ifdef ASCIFY_BINARY64_RNE_V1_HPP
#error "The scalar64 device backend cannot inherit a previously included host backend"
#endif
#define ASCIFY_BINARY64_INLINE ASCIFY_SCALAR64_DEVICE
#include <ascify/binary64_rne_v1.hpp>
#undef ASCIFY_BINARY64_INLINE
#endif
#endif

namespace ascify {
namespace detail {

// This is a closed builtin-type domain. An eight-byte class, enum, vector or
// user conversion never acquires the transport capability from its size.
template<class T> struct scalar64_transport_type : std::integral_constant<bool,
    (std::is_same<T, long>::value || std::is_same<T, unsigned long>::value ||
     std::is_same<T, long long>::value ||
     std::is_same<T, unsigned long long>::value ||
     std::is_same<T, double>::value) && sizeof(T) == 8> {};

template<class T> ASCIFY_SCALAR64_DEVICE uint64_t scalar64_to_bits(T value) {
  static_assert(scalar64_transport_type<T>::value,
                "Ascify scalar64 transport requires an exact builtin64 type");
#if defined(ASCIFY_SCALAR64_VERIFIED_TARGET)
  return __builtin_bit_cast(uint64_t, value);
#else
  static_assert(!std::is_same<T, T>::value,
                "Ascify scalar64 requires the verified 950 legacy SIMT bit_cast target");
  return 0;
#endif
}

template<class T> ASCIFY_SCALAR64_DEVICE T scalar64_from_bits(uint64_t value) {
  static_assert(scalar64_transport_type<T>::value,
                "Ascify scalar64 transport requires an exact builtin64 type");
#if defined(ASCIFY_SCALAR64_VERIFIED_TARGET)
  return __builtin_bit_cast(T, value);
#else
  static_assert(!std::is_same<T, T>::value,
                "Ascify scalar64 requires the verified 950 legacy SIMT bit_cast target");
  return T();
#endif
}

template<class T> ASCIFY_SCALAR64_DEVICE T scalar64_shfl_down(
    uint32_t mask, T value, unsigned int delta, int width = 32) {
#if defined(ASCIFY_SCALAR64_VERIFIED_TARGET)
  require_converged_warp_mask(mask);
  if (asc_ballot(width != 32) != 0) reject_device_contract();
  // A large delta must not overwrite the native packed control fields. All
  // lanes still execute the shuffle, including lanes retaining their value.
  const unsigned int native_delta = delta >= 32U ? 0U : delta;
  return scalar64_from_bits<T>(asc_shfl_down(
      scalar64_to_bits(value), native_delta, 32));
#else
  static_assert(!std::is_same<T, T>::value,
                "Ascify scalar64 shuffle requires the verified 950 target");
  return value;
#endif
}

template<class T> ASCIFY_SCALAR64_DEVICE T scalar64_shfl_index(
    uint32_t mask, T value, int source_lane, int width = 32) {
#if defined(ASCIFY_SCALAR64_VERIFIED_TARGET)
  require_converged_warp_mask(mask);
  // Index shuffle retains its full-warp contract. Do not infer a new sparse
  // rendezvous or narrow-width implementation from down-shuffle admission.
  if (asc_ballot(mask != UINT32_MAX || width != 32 ||
                 source_lane < 0 || source_lane >= 32) != 0)
    reject_device_contract();
  return scalar64_from_bits<T>(asc_shfl(
      scalar64_to_bits(value), source_lane, 32));
#else
  static_assert(!std::is_same<T, T>::value,
                "Ascify scalar64 shuffle requires the verified 950 target");
  return value;
#endif
}

template<class T> struct scalar_add_type : std::integral_constant<bool,
    std::is_same<T, int>::value || std::is_same<T, float>::value ||
    std::is_same<T, double>::value> {};

template<class T> ASCIFY_SCALAR64_DEVICE T scalar_add_rne(T left, T right) {
  static_assert(scalar_add_type<T>::value,
                "Ascify scalar add admits only int, float and binary64 double");
  if constexpr (std::is_same<T, double>::value) {
#if defined(ASCIFY_SCALAR64_VERIFIED_TARGET)
    return scalar64_from_bits<double>(binary64_rne_v1::add_rne_bits(
        scalar64_to_bits(left), scalar64_to_bits(right)));
#else
    static_assert(!std::is_same<T, double>::value,
                  "Ascify binary64 RNE add requires the verified 950 target");
    return left;
#endif
  } else {
    // Keep the original builtin int/float operation and its rounding/order.
    return left + right;
  }
}

// Exact IEEE-754 float->double widening with integer bit logic: the
// target's SIMT entry cannot call the rtlib __extendsfdf2 helper, and the
// scalar += projection delivers float right-hand sides to double slots.
ASCIFY_SCALAR64_DEVICE double widen_f32_to_f64(float value) {
  const unsigned bits = __builtin_bit_cast(unsigned, value);
  const unsigned exponent = (bits >> 23) & 0xffu;
  const unsigned fraction = bits & 0x7fffffu;
  const unsigned long long sign =
      static_cast<unsigned long long>(bits >> 31) << 63;
  if (exponent == 0xffu)
    return __builtin_bit_cast(double,
        sign | 0x7ff0000000000000ull |
        (static_cast<unsigned long long>(fraction) << 29));
  if (exponent == 0) {
    if (fraction == 0) return __builtin_bit_cast(double, sign);
    int normalized = -126;
    unsigned mantissa = fraction;
    while ((mantissa & 0x800000u) == 0) { mantissa <<= 1; --normalized; }
    mantissa &= 0x7fffffu;
    return __builtin_bit_cast(double,
        sign | (static_cast<unsigned long long>(normalized + 1023) << 52) |
        (static_cast<unsigned long long>(mantissa) << 29));
  }
  return __builtin_bit_cast(double,
      sign | (static_cast<unsigned long long>(exponent - 127 + 1023) << 52) |
      (static_cast<unsigned long long>(fraction) << 29));
}

// Exact IEEE-754 double->float narrowing (round to nearest even) in
// integer logic: the SIMT entry cannot call the rtlib __truncdfsf2 helper.
ASCIFY_SCALAR64_DEVICE float narrow_f64_to_f32(double value) {
  const unsigned long long bits =
      __builtin_bit_cast(unsigned long long, value);
  const unsigned sign = static_cast<unsigned>(bits >> 63) << 31;
  const int exponent = static_cast<int>((bits >> 52) & 0x7ff) - 1023;
  const unsigned long long fraction = bits & 0xfffffffffffffull;
  if (((bits >> 52) & 0x7ff) == 0x7ff)
    return __builtin_bit_cast(float,
        sign | 0x7f800000u | static_cast<unsigned>(fraction >> 29));
  if (exponent > 127)
    return __builtin_bit_cast(float, sign | 0x7f800000u);
  if (exponent >= -126) {
    unsigned long long mantissa = fraction >> 29;
    const unsigned long long remainder = fraction & 0x1fffffffull;
    if (remainder > 0x10000000ull ||
        (remainder == 0x10000000ull && (mantissa & 1ull) != 0))
      ++mantissa;
    if ((mantissa >> 23) != 0)
      return __builtin_bit_cast(float,
          sign | (static_cast<unsigned>(exponent + 128) << 23));
    return __builtin_bit_cast(float,
        sign | (static_cast<unsigned>(exponent + 127) << 23) |
        static_cast<unsigned>(mantissa));
  }
  if (exponent < -149)
    return __builtin_bit_cast(float, sign);
  // Subnormal: 1.fraction * 2^exponent into f32 subnormal grid.
  unsigned long long mantissa = fraction | (1ull << 52);
  const int shift = -exponent - 126 + 29;
  unsigned long long quotient = mantissa >> shift;
  const unsigned long long rem = mantissa & ((1ull << shift) - 1);
  const unsigned long long half = 1ull << (shift - 1);
  if (rem > half || (rem == half && (quotient & 1ull) != 0))
    ++quotient;
  if (quotient >= (1ull << 23))
    return __builtin_bit_cast(float, sign | 0x3f800000u);
  return __builtin_bit_cast(float, sign | static_cast<unsigned>(quotient));
}

template<class T> ASCIFY_SCALAR64_DEVICE void scalar_add_assign_rne(
    T& left, T right) {
  static_assert(scalar_add_type<T>::value && !std::is_volatile<T>::value,
                "Ascify scalar += requires a nonvolatile builtin scalar");
  // Binding the reference evaluates the proved lvalue address exactly once.
  // Its old value is read only after the RHS argument has finished evaluating.
  const T previous = left;
  left = scalar_add_rne<T>(previous, right);
}

// float right-hand side widened exactly in integer logic, then the same
// binary64 RNE add (no rtlib conversion call inside the SIMT entry). Only
// the verified target defines the binary64 add this builds on.
#if defined(ASCIFY_SCALAR64_VERIFIED_TARGET)
ASCIFY_SCALAR64_DEVICE void scalar_add_assign_rne(
    double& left, float right) {
  const double previous = left;
  left = scalar_add_rne<double>(previous, widen_f32_to_f64(right));
}
#endif

// Double sqrt through the float domain: exact RNE narrowing, native float
// sqrt, exact widening back. The domain error stays 2^-24-class relative,
// far inside the 1e-5-class absolute gates the samples hold.
ASCIFY_SCALAR64_DEVICE double scalar64_sqrt_f32(double value) {
  return widen_f32_to_f64(sqrtf(narrow_f64_to_f32(value)));
}

} // namespace detail
} // namespace ascify

#undef ASCIFY_SCALAR64_DEVICE
#endif
