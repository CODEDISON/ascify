#include <ascify/cooperative_groups_compat.hpp>

namespace cg = ascify_cg;
void RejectTile16RegisterOperations(cg::thread_block block) {
  const auto tile = cg::tiled_partition<16>(block);
#if ASCIFY_TILE16_REJECT_CASE == 1
  (void)tile.shfl(1, 0);
#elif ASCIFY_TILE16_REJECT_CASE == 2
  (void)tile.shfl_down(1.0, 1);
#elif ASCIFY_TILE16_REJECT_CASE == 3
  (void)cg::reduce(tile, 1, cg::plus<int>{});
#else
#error "ASCIFY_TILE16_REJECT_CASE must select a negative test"
#endif
}
