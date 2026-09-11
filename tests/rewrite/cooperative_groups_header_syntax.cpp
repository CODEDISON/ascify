#include <ascify/cooperative_groups_compat.hpp>

namespace cg = ascify_cg;

int main() {
  const cg::thread_block block = cg::this_thread_block();
  cg::sync(block);
  const auto tile = cg::tiled_partition<32>(block);
  static_assert(std::is_same<decltype(tile.thread_rank()), unsigned int>::value);
  static_assert(std::is_same<decltype(tile.size()), unsigned int>::value);
  static_assert(std::is_same<decltype(tile.meta_group_rank()), unsigned int>::value);
  static_assert(decltype(tile)::size() == 32);
  (void)decltype(tile)::thread_rank();
  (void)decltype(tile)::meta_group_rank();
  (void)tile.shfl_up(1U, 1);
  (void)tile.shfl_xor(2U, 1);
  (void)tile.shfl_xor(uint4{0U, 0xffffffffU, 0x7f800000U, 0x80000000U}, 3);
  return 0;
}
