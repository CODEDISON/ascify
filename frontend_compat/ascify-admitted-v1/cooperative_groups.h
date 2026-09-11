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
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static void sync();
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned int thread_rank();
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned int size();
};
ASCIFY_FRONTEND_COMPAT_DEVICE_ thread_block this_thread_block();
ASCIFY_FRONTEND_COMPAT_DEVICE_ inline void sync(const thread_block& group) {
  group.sync();
}

template <unsigned int Size, typename ParentT = void> class thread_block_tile;
namespace ascify_detail {
// Register collectives only. A native tile fence is not a collective barrier.
template <unsigned int Size> class tile_register_operations {
  static_assert(Size == 32, "Ascify admits only full-warp tile size 32; multi-warp subgroup synchronization is not implemented");
 protected:
  ASCIFY_FRONTEND_COMPAT_DEVICE_ tile_register_operations();
 public:
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned int thread_rank();
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static constexpr unsigned int size() { return Size; }
  template <typename T> ASCIFY_FRONTEND_COMPAT_DEVICE_
  typename std::enable_if<std::is_same<T, int>::value ||
      std::is_same<T, unsigned int>::value || std::is_same<T, float>::value, T>::type
  shfl_up(T value, unsigned int delta) const;
  template <typename T> ASCIFY_FRONTEND_COMPAT_DEVICE_
  typename std::enable_if<std::is_same<T, int>::value ||
      std::is_same<T, unsigned int>::value || std::is_same<T, float>::value, T>::type
  shfl(T value, int source_rank) const;
  template <typename T> ASCIFY_FRONTEND_COMPAT_DEVICE_
  typename std::enable_if<std::is_same<T, int>::value ||
      std::is_same<T, unsigned int>::value || std::is_same<T, float>::value, T>::type
  shfl_down(T value, unsigned int delta) const;
  template <typename T> ASCIFY_FRONTEND_COMPAT_DEVICE_
  typename std::enable_if<std::is_same<T, int>::value ||
      std::is_same<T, unsigned int>::value || std::is_same<T, float>::value ||
      std::is_same<T, uint4>::value, T>::type
  shfl_xor(T value, unsigned int lane_mask) const;
};
}  // namespace ascify_detail

template <unsigned int Size, typename ParentT>
class thread_block_tile : public ascify_detail::tile_register_operations<Size> {
  static_assert(std::is_same<ParentT, thread_block>::value,
                "Ascify admits only tiles partitioned from a thread block");
  ASCIFY_FRONTEND_COMPAT_DEVICE_ thread_block_tile();
 public:
  ASCIFY_FRONTEND_COMPAT_DEVICE_ operator thread_block_tile<Size, void>() const;
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned int meta_group_rank();
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned int meta_group_size();
};
template <unsigned int Size>
class thread_block_tile<Size, void> : public ascify_detail::tile_register_operations<Size> {
 public:
  template <typename ParentT, typename std::enable_if<
      std::is_same<ParentT, thread_block>::value, int>::type = 0>
  ASCIFY_FRONTEND_COMPAT_DEVICE_ thread_block_tile(const thread_block_tile<Size, ParentT>&);
  ASCIFY_FRONTEND_COMPAT_DEVICE_ unsigned int meta_group_rank() const;
  ASCIFY_FRONTEND_COMPAT_DEVICE_ unsigned int meta_group_size() const;
};

template <unsigned int Size>
ASCIFY_FRONTEND_COMPAT_DEVICE_ thread_block_tile<Size, thread_block>
tiled_partition(const thread_block& parent);
}  // namespace cooperative_groups
#undef ASCIFY_FRONTEND_COMPAT_DEVICE_
#endif
