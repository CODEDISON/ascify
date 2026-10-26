#ifndef ASCIFY_TEST_SCALAR64_ORACLE_HPP
#define ASCIFY_TEST_SCALAR64_ORACLE_HPP
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

// Literal IEEE-754 RNE cases, independently validated against exact rational
// arithmetic in the engineering probe. No product helper computes expected.
struct Scalar64Case { uint64_t left, right, expected; };
constexpr Scalar64Case scalar64_cases[] = {
  {0x4340000000000000, 0x3ff0000000000000, 0x4340000000000000}, // 2^53 + 1 ties even
  {0x4340000000000001, 0x3ff0000000000000, 0x4340000000000002}, // odd tie
  {0x4340000000000001, 0xc340000000000000, 0x4000000000000000}, // cancellation to 2
  {0x3ff0000000000000, 0x3ca0000000000000, 0x3ff0000000000000}, // 1 + half ulp
  {0x3ff0000000000001, 0x3ca0000000000000, 0x3ff0000000000002},
  {0x3ff0000000000000, 0x3ca0000000000001, 0x3ff0000000000001}, // above midpoint
  {0x3ff0000000000000, 0x3c9fffffffffffff, 0x3ff0000000000000}, // below midpoint
  {0x8000000000000000, 0x8000000000000000, 0x8000000000000000}, // negative zero
  {0x8000000000000000, 0x0000000000000000, 0x0000000000000000},
  {0x0000000000000001, 0x0000000000000001, 0x0000000000000002}, // min subnormal
  {0x8000000000000001, 0x8000000000000001, 0x8000000000000002},
  {0x000fffffffffffff, 0x0000000000000001, 0x0010000000000000}, // subnormal boundary
  {0x0010000000000000, 0x800fffffffffffff, 0x0000000000000001},
  {0x0010000000000000, 0x0000000000000001, 0x0010000000000001},
  {0x3ff0000000000000, 0xbff0000000000000, 0x0000000000000000},
  {0x7fefffffffffffff, 0x7fefffffffffffff, 0x7ff0000000000000}, // overflow
  {0xffefffffffffffff, 0xffefffffffffffff, 0xfff0000000000000},
  {0x7ff8000000001234, 0xfff8000000005678, 0x7ff8000000001234}, // fixed NaN payload policy
  {0x7ff0000000000042, 0x3ff0000000000000, 0x7ff8000000000042},
  {0x7ff0000000000000, 0xfff0000000000000, 0xfff8000000000000},
};

template<class T> T scalar64_from_raw(uint64_t raw) {
  static_assert(sizeof(T) == sizeof(raw));
  T result;
  std::memcpy(&result, &raw, sizeof(result));
  return result;
}
template<class T> uint64_t scalar64_raw(T value) {
  static_assert(sizeof(T) == sizeof(uint64_t));
  uint64_t result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}
inline void scalar64_expect(uint64_t actual, uint64_t expected) {
  if (actual != expected) {
    std::fprintf(stderr, "raw mismatch: got=%016llx expected=%016llx\n",
        static_cast<unsigned long long>(actual), static_cast<unsigned long long>(expected));
    std::abort();
  }
}
template<class Function> void scalar64_check_add(Function function) {
  for (const auto& item : scalar64_cases)
    scalar64_expect(scalar64_raw(function(scalar64_from_raw<double>(item.left),
        scalar64_from_raw<double>(item.right))), item.expected);
}
template<class Function> void scalar64_check_assign(Function function) {
  for (const auto& item : scalar64_cases) {
    double value = scalar64_from_raw<double>(item.left);
    function(value, scalar64_from_raw<double>(item.right));
    scalar64_expect(scalar64_raw(value), item.expected);
  }
}
#endif
