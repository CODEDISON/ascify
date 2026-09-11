#include <cooperative_groups/reduce.h>
void RejectBlockReduction(cooperative_groups::thread_block block) {
  (void)cooperative_groups::reduce(block, 1, cooperative_groups::plus<int>());
}
