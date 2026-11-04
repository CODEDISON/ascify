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
// Value domain only; no class/enum obtains this capability from sizeof alone.
template<class T> struct scalar64_transport_type : std::integral_constant<bool,
    (std::is_same<T, long>::value || std::is_same<T, unsigned long>::value ||
     std::is_same<T, long long>::value ||
     std::is_same<T, unsigned long long>::value ||
     std::is_same<T, double>::value) && sizeof(T) == 8> {};
// Full-warp register collectives and synchronization with runtime participation checks.
template <unsigned int Size> class tile_register_operations {
  static_assert(Size == 32, "Ascify admits only full-warp tile size 32; multi-warp subgroup synchronization is not implemented");
 protected:
  ASCIFY_FRONTEND_COMPAT_DEVICE_ tile_register_operations();
 public:
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned int thread_rank();
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static constexpr unsigned int size() { return Size; }
  ASCIFY_FRONTEND_COMPAT_DEVICE_ void sync() const;
  template <typename T> ASCIFY_FRONTEND_COMPAT_DEVICE_
  typename std::enable_if<std::is_same<T, int>::value ||
      std::is_same<T, unsigned int>::value || std::is_same<T, float>::value, T>::type
  shfl_up(T value, unsigned int delta) const;
  template <typename T> ASCIFY_FRONTEND_COMPAT_DEVICE_
  typename std::enable_if<std::is_same<T, int>::value ||
      std::is_same<T, unsigned int>::value || std::is_same<T, float>::value ||
      scalar64_transport_type<T>::value, T>::type
  shfl(T value, int source_rank) const;
  template <typename T> ASCIFY_FRONTEND_COMPAT_DEVICE_
  typename std::enable_if<std::is_same<T, int>::value ||
      std::is_same<T, unsigned int>::value || std::is_same<T, float>::value ||
      scalar64_transport_type<T>::value, T>::type
  shfl_down(T value, unsigned int delta) const;
  template <typename T> ASCIFY_FRONTEND_COMPAT_DEVICE_
  typename std::enable_if<std::is_same<T, int>::value ||
      std::is_same<T, unsigned int>::value || std::is_same<T, float>::value ||
      std::is_same<T, uint4>::value, T>::type
  shfl_xor(T value, unsigned int lane_mask) const;
};
// Rank/size/sync only. A mandatory source proof establishes complete-warp
// participation at every call/epoch before this projection can publish.
template <> class tile_register_operations<16> {
 protected:
  ASCIFY_FRONTEND_COMPAT_DEVICE_ tile_register_operations();
 public:
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned int thread_rank();
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static constexpr unsigned int size() { return 16; }
  ASCIFY_FRONTEND_COMPAT_DEVICE_ void sync() const;
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

template <unsigned int Size, typename ParentT>
ASCIFY_FRONTEND_COMPAT_DEVICE_ inline void
sync(const thread_block_tile<Size, ParentT>& group) {
  group.sync();
}

// Erased values may originate only from an authenticated block or static16
// tile. The mandatory whole-TU proof rejects representation observations,
// escapes, source overloads and unproved control/call/launch paths.
class thread_group final {
 public:
  ASCIFY_FRONTEND_COMPAT_DEVICE_ thread_group(const thread_block&);
  template <typename ParentT, typename std::enable_if<
      std::is_same<ParentT, thread_block>::value ||
      std::is_same<ParentT, void>::value, int>::type = 0>
  ASCIFY_FRONTEND_COMPAT_DEVICE_ thread_group(const thread_block_tile<16, ParentT>&);
  ASCIFY_FRONTEND_COMPAT_DEVICE_ unsigned long long size() const;
  ASCIFY_FRONTEND_COMPAT_DEVICE_ unsigned long long thread_rank() const;
  ASCIFY_FRONTEND_COMPAT_DEVICE_ void sync() const;
};
ASCIFY_FRONTEND_COMPAT_DEVICE_ inline void sync(const thread_group& group) {
  group.sync();
}

// A parsing projection for a separately proven whole-block scratch domain.
// Ordinary thread_block_tile still rejects sizes above 32. Ascify's mandatory
// AST gate rejects every unproved use of the distinct types below; their
// identities may not be observed by user code, overloads, or type traits.
template <unsigned BlockSize> struct block_tile_memory {};
template <unsigned BlockSize> struct scratch_thread_block {
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static void sync();
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned int thread_rank();
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned int size();
};
template <unsigned BlockSize>
ASCIFY_FRONTEND_COMPAT_DEVICE_ scratch_thread_block<BlockSize>
this_thread_block(block_tile_memory<BlockSize>& storage);
template <unsigned BlockSize>
ASCIFY_FRONTEND_COMPAT_DEVICE_ void sync(const scratch_thread_block<BlockSize>& group);
template <unsigned Size, unsigned BlockSize> struct scratch_thread_block_tile {
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned int thread_rank();
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static constexpr unsigned int size() { return Size; }
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned int meta_group_rank();
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned int meta_group_size();
};
template <unsigned Size, unsigned BlockSize>
ASCIFY_FRONTEND_COMPAT_DEVICE_ scratch_thread_block_tile<Size, BlockSize>
tiled_partition(const scratch_thread_block<BlockSize>& parent);

// Opaque parsing values for a separately authenticated register-only group
// contract. Mandatory whole-TU proof must reject all unproved operations,
// representation observations, aliases, copies and escapes before publication.
class grid_group {
  ASCIFY_FRONTEND_COMPAT_DEVICE_ grid_group();
 public:
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned long long thread_rank();
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned long long size();
  ASCIFY_FRONTEND_COMPAT_DEVICE_ bool is_valid() const;
  ASCIFY_FRONTEND_COMPAT_DEVICE_ void sync() const;
};
ASCIFY_FRONTEND_COMPAT_DEVICE_ grid_group this_grid();
class coalesced_group {
 protected:
  ASCIFY_FRONTEND_COMPAT_DEVICE_ coalesced_group();
 public:
  ASCIFY_FRONTEND_COMPAT_DEVICE_ unsigned size() const;
  ASCIFY_FRONTEND_COMPAT_DEVICE_ unsigned num_threads() const;
  ASCIFY_FRONTEND_COMPAT_DEVICE_ unsigned thread_rank() const;
  ASCIFY_FRONTEND_COMPAT_DEVICE_ unsigned meta_group_rank() const;
  ASCIFY_FRONTEND_COMPAT_DEVICE_ unsigned meta_group_size() const;
  ASCIFY_FRONTEND_COMPAT_DEVICE_ void sync() const;
  template<class T> ASCIFY_FRONTEND_COMPAT_DEVICE_ T shfl(T,int) const;
};
ASCIFY_FRONTEND_COMPAT_DEVICE_ coalesced_group coalesced_threads();
template<unsigned Size,class Parent> ASCIFY_FRONTEND_COMPAT_DEVICE_
coalesced_group binary_partition(const thread_block_tile<Size,Parent>&,bool);
ASCIFY_FRONTEND_COMPAT_DEVICE_ coalesced_group binary_partition(const coalesced_group&,bool);
template<class Label> ASCIFY_FRONTEND_COMPAT_DEVICE_
coalesced_group labeled_partition(const coalesced_group&,Label);
template<class Label,unsigned Size,class Parent> ASCIFY_FRONTEND_COMPAT_DEVICE_
coalesced_group labeled_partition(const thread_block_tile<Size,Parent>&,Label);
}  // namespace cooperative_groups
#undef ASCIFY_FRONTEND_COMPAT_DEVICE_
#endif
