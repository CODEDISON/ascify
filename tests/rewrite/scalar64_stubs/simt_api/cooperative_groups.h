#ifndef ASCIFY_TEST_SCALAR64_CG_H
#define ASCIFY_TEST_SCALAR64_CG_H
// Only a host linkage model. It cannot establish warp scheduling or NPU ABI.
#include <simt_api/device_warp_functions.h>
namespace cooperative_groups {
struct thread_block {
  static void sync() {}
  static unsigned thread_rank() { return 0; }
  static unsigned size() { return 32; }
};
template<unsigned Size, class Parent = void> struct thread_block_tile {
  static unsigned long long thread_rank() { return 0; }
  template<class T> T shfl(T value, int source) const {
    return asc_shfl(value, source, Size);
  }
  template<class T> T shfl_down(T value, unsigned delta) const {
    return asc_shfl_down(value, delta, Size);
  }
  template<class T> T shfl_up(T value, unsigned delta) const {
    return asc_shfl_up(value, delta, Size);
  }
  template<class T> T shfl_xor(T value, unsigned lane) const {
    return asc_shfl_xor(value, lane, Size);
  }
};
inline thread_block this_thread_block() { return {}; }
template<unsigned Size>
thread_block_tile<Size, thread_block> tiled_partition(const thread_block&) { return {}; }
}
#endif
