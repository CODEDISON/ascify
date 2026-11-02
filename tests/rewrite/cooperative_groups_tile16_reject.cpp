#include <ascify/cooperative_groups_compat.hpp>
void RejectTile16RegisterOperations(cooperative_groups::thread_block block) {
  const auto tile = cooperative_groups::tiled_partition<16>(block);
#if ASCIFY_TILE16_REJECT_CASE == 1
  (void)tile.shfl(1, 0);
#elif ASCIFY_TILE16_REJECT_CASE == 2
  (void)tile.shfl_down(1.0, 1);
#elif ASCIFY_TILE16_REJECT_CASE == 3
  (void)cooperative_groups::reduce(tile, 1, cooperative_groups::plus<int>{});
#else
#error "ASCIFY_TILE16_REJECT_CASE must select a negative test"
#endif
}
