#ifndef ASCIFY_SCALAR64_COMPAT_HPP
#define ASCIFY_SCALAR64_COMPAT_HPP

#include <stdint.h>
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

template<class T> ASCIFY_SCALAR64_DEVICE void scalar_add_assign_rne(
    T& left, T right) {
  static_assert(scalar_add_type<T>::value && !std::is_volatile<T>::value,
                "Ascify scalar += requires a nonvolatile builtin scalar");
  // Binding the reference evaluates the proved lvalue address exactly once.
  // Its old value is read only after the RHS argument has finished evaluating.
  const T previous = left;
  left = scalar_add_rne<T>(previous, right);
}

} // namespace detail
} // namespace ascify

#undef ASCIFY_SCALAR64_DEVICE
#endif
