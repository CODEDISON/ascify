#include <ascify/cooperative_groups_compat.hpp>

#include <cassert>
#include <cstring>
#include <iostream>

namespace {
uint4 input[32];
unsigned int current_lane;
unsigned int calls;

unsigned int component(const uint4& value, unsigned int index) {
  switch (index) {
    case 0: return value.x;
    case 1: return value.y;
    case 2: return value.z;
    default: return value.w;
  }
}
}

// A host model of the scalar native primitive checks the actual facade's
// component values and source-lane permutation. It is not an NPU simulator.
namespace cooperative_groups {
template <unsigned int Size, typename ParentT>
template <typename T>
T thread_block_tile<Size, ParentT>::shfl_xor(T value, unsigned int mask) const {
  static_assert(std::is_same<T, unsigned int>::value);
  assert(calls < 4);
  const unsigned int word = calls++;
  assert(value == component(input[current_lane], word));
  return component(input[current_lane ^ mask], word);
}

template <unsigned int Size>
thread_block_tile<Size, thread_block> tiled_partition(const thread_block&) {
  return {};
}
}

int main() {
  for (unsigned int lane = 0; lane != 32; ++lane) {
    input[lane] = uint4{0x9e3779b9U * (lane * 4U + 1U),
                        0x9e3779b9U * (lane * 4U + 2U),
                        0x9e3779b9U * (lane * 4U + 3U),
                        0x9e3779b9U * (lane * 4U + 4U)};
  }
  input[0] = uint4{0U, 0xffffffffU, 0x7fc00001U, 0x80000000U};
  const auto tile = ascify_cg::tiled_partition<32>(ascify_cg::this_thread_block());
  for (unsigned int mask = 0; mask != 32; ++mask) {
    for (current_lane = 0; current_lane != 32; ++current_lane) {
      calls = 0;
      const uint4 actual = tile.shfl_xor(input[current_lane], mask);
      assert(calls == 4);
      assert(std::memcmp(&actual, &input[current_lane ^ mask], sizeof actual) == 0);
    }
  }
  std::cout << "uint4 facade: 1024 lane/mask pairs, 4096 exact 32-bit components passed\n";
}
