#include <ascify/cooperative_groups_compat.hpp>

namespace cg = ascify_cg;
#include <type_traits>

void CheckTargetStatic16Sync(cg::thread_block block) {
  const auto typed = cg::tiled_partition<16>(block);
  const cg::thread_block_tile<16> tile = typed;
  const cg::thread_group from_block = block;
  const cg::thread_group from_typed = typed;
  const cg::thread_group from_tile = tile;
  static_assert(std::is_same<decltype(tile.size()), unsigned int>::value, "static size ABI");
  static_assert(std::is_same<decltype(from_tile.size()), unsigned long long>::value, "erased size ABI");
  static_assert(std::is_same<decltype(from_block.thread_rank()), unsigned long long>::value, "erased rank ABI");
  tile.sync();
  cg::sync(tile);
  from_block.sync();
  cg::sync(from_typed);
  from_tile.sync();
}
