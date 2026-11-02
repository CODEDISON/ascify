#include <ascify/cooperative_groups_compat.hpp>
#include <type_traits>

void CheckTargetStatic16Sync(cooperative_groups::thread_block block) {
  const auto typed = cooperative_groups::tiled_partition<16>(block);
  const cooperative_groups::thread_block_tile<16> tile = typed;
  const cooperative_groups::thread_group from_block = block;
  const cooperative_groups::thread_group from_typed = typed;
  const cooperative_groups::thread_group from_tile = tile;
  static_assert(std::is_same<decltype(tile.size()), unsigned int>::value, "static size ABI");
  static_assert(std::is_same<decltype(from_tile.size()), unsigned long long>::value, "erased size ABI");
  static_assert(std::is_same<decltype(from_block.thread_rank()), unsigned long long>::value, "erased rank ABI");
  tile.sync();
  cooperative_groups::sync(tile);
  from_block.sync();
  cooperative_groups::sync(from_typed);
  from_tile.sync();
}
