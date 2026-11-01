#ifndef ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_HOST_MATH_H_
#define ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_HOST_MATH_H_

// CUDA's float max(a, b) returns fmaxf(a, b), including its NaN behavior.
// Clang's CUDA wrapper exposes only the device overload. These opt-in parser
// templates admit float/float max, int/int min/max, and the two exact
// int/unsigned-int min orders; no argument is narrowed.
// Ascify rewrites proven references to target builtins or a finite helper.
namespace ascify_frontend_compat_detail {
template <bool> struct host_float_max_enabled {};
template <> struct host_float_max_enabled<true> { using type = int; };
template <bool> struct host_int_min_enabled {};
template <> struct host_int_min_enabled<true> { using type = int; };
template <bool> struct host_mixed_uint_min_enabled {};
template <> struct host_mixed_uint_min_enabled<true> { using type = int; };
template <bool> struct host_int_max_enabled {};
template <> struct host_int_max_enabled<true> { using type = int; };
}

template <class A, class B,
          typename ascify_frontend_compat_detail::host_float_max_enabled<
              __is_same(A, float) && __is_same(B, float)>::type = 0>
#if defined(__CUDACC__) || defined(__CUDA__)
__attribute__((host))
#endif
inline float max(A a, B b) {
  return __builtin_fmaxf(a, b);
}

// Use value parameters: CUDA integer min returns a value and each actual
// argument is evaluated once. The target spelling is proven separately.
template <class A, class B,
          typename ascify_frontend_compat_detail::host_int_min_enabled<
              __is_same(A, int) && __is_same(B, int)>::type = 0>
#if defined(__CUDACC__) || defined(__CUDA__)
__attribute__((host))
#endif
inline int min(A a, B b) {
  return a < b ? a : b;
}

// CUDA converts the signed operand to unsigned int before taking the minimum.
// The conditional's usual arithmetic conversion preserves that value domain.
template <class A, class B,
          typename ascify_frontend_compat_detail::host_mixed_uint_min_enabled<
              (__is_same(A, int) && __is_same(B, unsigned int)) ||
              (__is_same(A, unsigned int) && __is_same(B, int))>::type = 0>
#if defined(__CUDACC__) || defined(__CUDA__)
__attribute__((host))
#endif
inline unsigned int min(A a, B b) {
  return a < b ? a : b;
}

template <class A, class B,
          typename ascify_frontend_compat_detail::host_int_max_enabled<
              __is_same(A, int) && __is_same(B, int)>::type = 0>
#if defined(__CUDACC__) || defined(__CUDA__)
__attribute__((host))
#endif
inline int max(A a, B b) {
  return a > b ? a : b;
}

#endif
