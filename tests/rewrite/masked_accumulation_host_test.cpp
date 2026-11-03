#include <ascify/ascify_cuda_compat.hpp>
#include <array>
#include <cassert>
#include <cfenv>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <type_traits>
#include <vector>

namespace model = masked_accumulation_test;
template<class T> uint64_t bits(T value) {
  uint64_t result = 0;
  std::memcpy(&result, &value, sizeof value);
  return result;
}
template<class T> T scalar(unsigned lane, unsigned pattern) {
  if (pattern == 0) return T(int(lane) - 15);
  if (pattern == 1) return T(-0.0);
  if (pattern == 2) return T((lane & 1) ? 23 : -19);
  if constexpr (std::is_same<T, int>::value) return int(lane * 73U % 997U) - 498;
  else if (pattern == 3) return T((lane & 1) ? -1e20 : 1e20);
  else return T((int(lane) - 15) * 0.00003125);
}
template<class T> T plus(T a, T b) {
  // Independent native host IEEE operation, never the target integer RNE code.
  volatile T result = a + b;
  return result;
}
template<class T> T prefix_reference(const std::vector<T>& values) {
  assert(!values.empty());
  if (values.size() == 1) return values[0];
  std::vector<T> even, odd;
  for (unsigned i = 0; i < values.size(); ++i)
    ((i & 1U) ? odd : even).push_back(values[i]);
  // This recursive even/odd partition gives the original down-tree order;
  // no absent value is replaced by floating positive zero.
  const T left = prefix_reference(even);
  return odd.empty() ? left : plus(left, prefix_reference(odd));
}
template<class T> void provide(const std::array<T, 32>& values) {
  for (unsigned i = 0; i < 32; ++i) {
    if constexpr (std::is_same<T, int>::value) model::integers[i] = values[i];
    else if constexpr (std::is_same<T, float>::value) model::floats[i] = values[i];
    else model::words[i] = bits(values[i]);
  }
}
struct Counts { uint64_t steps = 0, defined_results = 0, prefix_leaders = 0, safe_shuffles = 0; };
template<class T> void check(uint32_t mask, unsigned pattern, Counts& count) {
  std::array<T, 32> values, original;
  std::array<bool, 32> defined;
  for (unsigned i = 0; i < 32; ++i) {
    values[i] = original[i] = scalar<T>(i, pattern);
    defined[i] = (mask & (uint32_t(1) << i)) != 0;
  }
  std::vector<T> prefix;
  for (unsigned i = 0; i < 32 && (mask & (uint32_t(1) << i)); ++i)
    prefix.push_back(values[i]);
  const bool is_prefix = mask == UINT32_MAX || ((mask & (mask + 1U)) == 0);
  ascify_test_legacy_active_mask = mask;
  for (unsigned offset = 16; offset; offset >>= 1) {
    provide(values);
    const auto old = original;
    const auto was_defined = defined;
    const unsigned before = model::calls;
    unsigned active = 0;
    for (unsigned lane = 0; lane < 32; ++lane) {
      if (!(mask & (uint32_t(1) << lane))) continue;
      ++active;
      model::lane = lane;
      // Exercise the existing protected 3D CUDA lane mapping as well.
      blockDim = {4, 4, 4};
      threadIdx = {lane % 4, (lane / 4) % 4, lane / 16};
      const T rhs = ascify::shfl_down_sync(mask, values[lane],
          ascify::detail::masked_accumulation_delta(mask, offset), 32);
      ascify::detail::scalar_masked_add_assign_rne<T>(values[lane], rhs, mask, offset);

      // Original CUDA reference: track its undefined results without choosing
      // a value for them. Compare every result that the source actually defines.
      const unsigned source = lane + offset < 32 ? lane + offset : lane;
      defined[lane] = was_defined[lane] && was_defined[source];
      if (defined[lane]) {
        original[lane] = plus(old[lane], old[source]);
        assert(bits(values[lane]) == bits(original[lane]));
        ++count.defined_results;
      }
      ++count.steps;
    }
    assert(model::calls - before == active);
    count.safe_shuffles += active;
  }
  if (is_prefix) {
    assert(!prefix.empty());
    assert(bits(values[0]) == bits(prefix_reference(prefix)));
    ++count.prefix_leaders;
    if constexpr (std::is_same<T, int>::value) {
      int total = 0;
      for (int value : prefix) total += value;
      assert(values[0] == total);
    }
  }
}
int main() {
  assert(std::fesetround(FE_TONEAREST) == 0);
  std::set<uint32_t> masks{UINT32_MAX};
  for (unsigned n = 1; n <= 32; ++n) {
    masks.insert(UINT32_MAX >> (32 - n));
    masks.insert(UINT32_MAX << (32 - n));
  }
  for (unsigned i = 0; i < 32; ++i) {
    masks.insert(uint32_t(1) << i);
    for (unsigned j = i + 1; j < 32; ++j)
      masks.insert((uint32_t(1) << i) | (uint32_t(1) << j));
  }
  std::mt19937 random(0x173a9);
  for (unsigned n = 0; n < 2048; ++n) masks.insert(random() | 1U);
  Counts count;
  for (auto mask : masks) for (unsigned pattern = 0; pattern < 5; ++pattern) {
    check<int>(mask, pattern, count);
    check<float>(mask, pattern, count);
    check<double>(mask, pattern, count);
  }
  // Unsigned large deltas remain the defined self-source path, never a shift
  // by32 or an overflowing lane+delta calculation.
  ascify_test_legacy_active_mask = UINT32_MAX;
  blockDim = {32, 1, 1};
  for (unsigned lane = 0; lane < 32; ++lane) {
    threadIdx = {lane, 0, 0};
    assert(ascify::detail::masked_accumulation_delta(UINT32_MAX, UINT32_MAX) == 0);
    assert(ascify::detail::masked_accumulation_source_present(UINT32_MAX, UINT32_MAX));
  }
  std::printf("masked accumulation masks=%zu steps=%llu defined_results=%llu prefix_leaders=%llu safe_shuffles=%llu\n",
      masks.size(), (unsigned long long)count.steps, (unsigned long long)count.defined_results,
      (unsigned long long)count.prefix_leaders, (unsigned long long)count.safe_shuffles);
}
