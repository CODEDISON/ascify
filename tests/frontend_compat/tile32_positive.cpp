#include <cooperative_groups.h>

struct alignas(16) uint4 { unsigned int x, y, z, w; };

uint4 TilePackedRegisters(cooperative_groups::thread_block block, uint4 input) {
  return cooperative_groups::tiled_partition<32>(block).shfl_xor(input, 3);
}

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

static_assert(std::is_same<decltype(cooperative_groups::thread_block::thread_rank()), unsigned int>::value);
static_assert(std::is_same<decltype(cooperative_groups::thread_block::size()), unsigned int>::value);
void ConvertedParentTile(cooperative_groups::thread_block block) {
  const auto typed = cooperative_groups::tiled_partition<32>(block);
  cooperative_groups::thread_block_tile<32> erased = typed;
  static_assert(std::is_member_function_pointer<decltype(&decltype(erased)::meta_group_size)>::value);
  static_assert(!std::is_member_function_pointer<decltype(&decltype(typed)::meta_group_size)>::value);
  (void)erased.meta_group_rank();
  (void)erased.meta_group_size();
  (void)erased.shfl_down(2.0f, 1);
  (void)erased.shfl(2, 0);
}
