#ifndef ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_COOPERATIVE_GROUPS_H_
#define ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_COOPERATIVE_GROUPS_H_

#include <type_traits>

struct uint4;

#if defined(__CUDACC__) || defined(__CUDA__)
#define ASCIFY_FRONTEND_COMPAT_DEVICE_ __device__
#else
#define ASCIFY_FRONTEND_COMPAT_DEVICE_
#endif

namespace cooperative_groups {

class thread_block {
 public:
  ASCIFY_FRONTEND_COMPAT_DEVICE_ void sync() const;
};

ASCIFY_FRONTEND_COMPAT_DEVICE_ thread_block this_thread_block();

ASCIFY_FRONTEND_COMPAT_DEVICE_ inline void sync(const thread_block& group) {
  group.sync();
}

// Only the full-warp, register-only tile surface verified against CANN 9.1.
// Tile synchronization is deliberately absent: the native tile sync is only
// a block memory fence, which is not proof of CUDA's collective barrier.
template <unsigned int Size, typename ParentT = void>
class thread_block_tile {
  static_assert(Size == 32, "Ascify admits only full-warp tile size 32");
  static_assert(std::is_same<ParentT, thread_block>::value,
                "Ascify admits only tiles partitioned from a thread block");
  ASCIFY_FRONTEND_COMPAT_DEVICE_ thread_block_tile();

 public:
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned int thread_rank();
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static constexpr unsigned int size() { return Size; }
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned int meta_group_rank();

  template <typename T>
  ASCIFY_FRONTEND_COMPAT_DEVICE_
  typename std::enable_if<std::is_same<T, int>::value ||
                              std::is_same<T, unsigned int>::value ||
                              std::is_same<T, float>::value, T>::type
  shfl_up(T value, unsigned int delta) const;

  template <typename T>
  ASCIFY_FRONTEND_COMPAT_DEVICE_
  typename std::enable_if<std::is_same<T, int>::value ||
                              std::is_same<T, unsigned int>::value ||
                              std::is_same<T, float>::value, T>::type
  shfl_xor(T value, unsigned int lane_mask) const;

  template <typename T>
  ASCIFY_FRONTEND_COMPAT_DEVICE_
  typename std::enable_if<std::is_same<T, uint4>::value, T>::type
  shfl_xor(T value, unsigned int lane_mask) const;
};

template <unsigned int Size>
ASCIFY_FRONTEND_COMPAT_DEVICE_ thread_block_tile<Size, thread_block>
tiled_partition(const thread_block& parent);

}  // namespace cooperative_groups

#undef ASCIFY_FRONTEND_COMPAT_DEVICE_

#endif  // ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_COOPERATIVE_GROUPS_H_
