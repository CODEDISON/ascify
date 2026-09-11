#include <cooperative_groups.h>

void RejectUnprovenTileOperation(cooperative_groups::thread_block block) {
#if ASCIFY_TILE_REJECT_CASE == 1
  const auto tile = cooperative_groups::tiled_partition<32>(block);
  tile.sync();
#elif ASCIFY_TILE_REJECT_CASE == 2
  const auto tile = cooperative_groups::tiled_partition<32>(block);
  cooperative_groups::sync(tile);
#elif ASCIFY_TILE_REJECT_CASE == 3
  const auto tile = cooperative_groups::tiled_partition<16>(block);
#elif ASCIFY_TILE_REJECT_CASE == 4
  const auto tile = cooperative_groups::tiled_partition<64>(block);
#elif ASCIFY_TILE_REJECT_CASE == 5
  const auto tile = cooperative_groups::tiled_partition<32>(block);
  const auto nested = cooperative_groups::tiled_partition<32>(tile);
#elif ASCIFY_TILE_REJECT_CASE == 6
  const auto tile = cooperative_groups::tiled_partition<32>(block);
  (void)tile.shfl_up(1.0, 1);
#else
#error "Unknown negative case"
#endif
}
