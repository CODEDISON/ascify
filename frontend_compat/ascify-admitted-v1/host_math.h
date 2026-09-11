#ifndef ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_HOST_MATH_H_
#define ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_HOST_MATH_H_

// CUDA's float max(a, b) returns fmaxf(a, b), including its NaN behavior.
// Clang's CUDA wrapper exposes only the device overload. This opt-in parser
// template admits exactly float/float host calls; no argument is narrowed.
// Ascify rewrites references proven to use this declaration to the builtin.
namespace ascify_frontend_compat_detail {
template <bool> struct host_float_max_enabled {};
template <> struct host_float_max_enabled<true> { using type = int; };
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

#endif
