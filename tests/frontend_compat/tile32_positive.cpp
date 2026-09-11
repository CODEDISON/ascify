#include <cooperative_groups.h>

unsigned int TileRegisterOperations(cooperative_groups::thread_block block,
                                    unsigned int input) {
  const auto tile = cooperative_groups::tiled_partition<32>(block);
  static_assert(std::is_same<decltype(tile.thread_rank()), unsigned int>::value);
  static_assert(std::is_same<decltype(tile.size()), unsigned int>::value);
  static_assert(std::is_same<decltype(tile.meta_group_rank()), unsigned int>::value);
  static_assert(decltype(tile)::size() == 32);
  (void)decltype(tile)::thread_rank();
  (void)decltype(tile)::meta_group_rank();
  return tile.shfl_up(input, 1) + tile.shfl_xor(input, 2) +
         tile.thread_rank() + tile.meta_group_rank() + tile.size();
}
