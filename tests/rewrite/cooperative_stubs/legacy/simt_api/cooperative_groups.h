#ifndef ASCIFY_TEST_NATIVE_COOPERATIVE_GROUPS_H
#define ASCIFY_TEST_NATIVE_COOPERATIVE_GROUPS_H

#include <simt_api/device_types.h>

namespace cooperative_groups {

struct thread_block {
  static void sync() {}
  static unsigned int thread_rank();
  static unsigned int size();
};

struct coalesced_group {
  void sync() const {}
};

template <unsigned int Size, typename ParentT = void>
struct thread_block_tile {
  void sync() const {}
  static unsigned long long thread_rank();
  template <typename T> T shfl_up(T, unsigned int) const;
  template <typename T> T shfl_down(T, unsigned int) const;
  template <typename T> T shfl(T, int) const;
  template <typename T> T shfl_xor(T, unsigned int) const;
};

template <unsigned int Size>
thread_block_tile<Size, thread_block> tiled_partition(const thread_block&);

inline thread_block this_thread_block() { return {}; }

}  // namespace cooperative_groups

#endif
