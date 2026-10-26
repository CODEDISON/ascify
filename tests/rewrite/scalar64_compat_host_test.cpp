#include <ascify/ascify_cuda_compat.hpp>
#include <ascify/cooperative_groups_compat.hpp>
#include "scalar64_oracle.hpp"

struct EightBytes { uint64_t value; };
enum class WideEnum : uint64_t { value = 1 };
static_assert(!ascify::detail::scalar64_transport_type<EightBytes>::value);
static_assert(!ascify::detail::scalar64_transport_type<WideEnum>::value);
static_assert(!ascify::detail::scalar64_transport_type<float>::value);

template<class T> static void transport() {
  const uint64_t words[] = {0, 0x8000000000000000, 1, 0x000fffffffffffff,
    0x7ff0000000000001, 0x7ff8000000001234, 0xfedcba9876543210,
    0x3ff0000000000001};
  const unsigned deltas[] = {0, 1, 31, 32, 0xffffffffU};
  for (const auto word : words) for (const auto delta : deltas) {
    scalar64_test::reset();
    const auto value = scalar64_from_raw<T>(word);
    const auto result = ascify::shfl_down_sync(UINT32_MAX, value, delta, 32);
    scalar64_expect(scalar64_raw(result), word);
    scalar64_expect(scalar64_test::input, word);
    if (scalar64_test::calls != 1 || scalar64_test::width != 32 ||
        scalar64_test::delta != (delta >= 32 ? 0 : delta)) std::abort();
    scalar64_test::reset();
    scalar64_test::supply_output = true;
    scalar64_test::output = word ^ UINT64_C(0x123456789abcdef0);
    scalar64_expect(scalar64_raw(ascify::shfl_sync(UINT32_MAX, value, 31, 32)),
        scalar64_test::output);
    scalar64_expect(scalar64_test::input, word);
    if (scalar64_test::calls != 1 || scalar64_test::source_lane != 31) std::abort();
  }
}

int main(int argc, char** argv) {
  scalar64_test::reset();
  const double value = scalar64_from_raw<double>(0x3ff0000000000001);
  if (argc == 2) {
    if (std::strcmp(argv[1], "invalid-width") == 0)
      (void)ascify::shfl_down_sync(UINT32_MAX, value, 1, 16);
    else if (std::strcmp(argv[1], "mask-mismatch") == 0)
      (void)ascify::shfl_down_sync(0xffffU, value, 1, 32);
    else if (std::strcmp(argv[1], "index-partial") == 0) {
      ascify_test_legacy_active_mask = 0xffffU;
      (void)ascify::shfl_sync(0xffffU, value, 1, 32);
    } else if (std::strcmp(argv[1], "index-range") == 0)
      (void)ascify::shfl_sync(UINT32_MAX, value, 32, 32);
    else if (std::strcmp(argv[1], "inactive") == 0) {
      ascify_test_legacy_active_mask = 0;
      (void)ascify::shfl_down_sync(0, value, 1, 32);
    } else return 2;
    // A no-op trap/model or accidental release from rejection must fail.
    return 99;
  }
  transport<double>();
  transport<long long>();
  transport<unsigned long long>();
  if constexpr (sizeof(long) == 8) transport<long>();
  if constexpr (sizeof(unsigned long) == 8) transport<unsigned long>();
  scalar64_check_add([](double a, double b) {
    return ascify::detail::scalar_add_rne<double>(a, b);
  });
  scalar64_check_assign([](double& a, double b) {
    ascify::detail::scalar_add_assign_rne<double>(a, b);
  });
  if (ascify::detail::scalar_add_rne<int>(27, 14) != 41) return 3;
  const float first = ascify::detail::scalar_add_rne<float>(16777216.f, 1.f);
  if (ascify::detail::scalar_add_rne<float>(first, -16777216.f) != 0.f) return 4;
  double values[2] = {value, 0};
  unsigned index = 0;
  ascify::detail::scalar_add_assign_rne<double>(values[index++],
      scalar64_from_raw<double>(0x3ca0000000000000));
  if (index != 1) return 5;
  scalar64_expect(scalar64_raw(values[0]), 0x3ff0000000000002);
  scalar64_expect(scalar64_raw(values[1]), 0);
  const auto tile = ascify_cg::tiled_partition<32>(ascify_cg::this_thread_block());
  scalar64_test::reset();
  scalar64_expect(scalar64_raw(tile.shfl_down(value, 1)), 0x3ff0000000000001);
  scalar64_expect(scalar64_raw(tile.shfl(value, 31)), 0x3ff0000000000001);
  if (scalar64_test::calls != 2) return 6;
  std::puts("scalar64 host model: exact raw words, int/float order, single evaluation and shuffle dispatch passed");
}
