#ifndef ASCIFY_COOPERATIVE_GROUPS_COMPAT_HPP
#define ASCIFY_COOPERATIVE_GROUPS_COMPAT_HPP

#include <type_traits>

#if !defined(__has_include)
#error "Ascify cooperative groups compatibility requires __has_include support"
#elif __has_include(<simt_api/cooperative_groups.h>) &&                     \
    __has_include(<simt_api/device_functions.h>) &&                        \
    __has_include(<simt_api/device_sync_functions.h>) &&                   \
    __has_include(<simt_api/device_warp_functions.h>)
#define ASCIFY_HAS_LEGACY_COOPERATIVE_GROUPS 1
#elif __has_include(<simt_api/kernel_simt_intf.h>)
#error "Ascify cooperative groups compatibility is unsupported on public CANN 8.5: no native cooperative_groups header"
#else
#error "Ascify cooperative groups compatibility requires the verified legacy CANN cooperative_groups headers"
#endif

#if defined(ASCIFY_HAS_LEGACY_COOPERATIVE_GROUPS)
#include <simt_api/device_functions.h>
#include <simt_api/device_sync_functions.h>
#include <simt_api/device_warp_functions.h>
#include <simt_api/cooperative_groups.h>
#include <ascify/device_contract_reject.hpp>
#include <ascify/scalar64_compat.hpp>

namespace cooperative_groups {

// CUDA exposes free sync(group). The verified legacy CANN header only exposes
// thread_block::sync(), whose implementation reaches the work-item barrier.
// Keep this overload concrete: accepting arbitrary groups would incorrectly
// admit tile/coalesced synchronization backed only by a block memory fence.
__SIMT_DEVICE_FUNCTIONS_DECL__ inline void sync(const thread_block& group) {
  group.sync();
}

}  // namespace cooperative_groups

// CUDA's static tile rank/size have unsigned-int type. Native CANN returns
// unsigned long long; exposing that type would change overload resolution.
// The translator maps the admitted CUDA namespace to this narrow facade.
namespace ascify_cg {

using thread_block = ::cooperative_groups::thread_block;

__SIMT_DEVICE_FUNCTIONS_DECL__ inline thread_block this_thread_block() {
  return ::cooperative_groups::this_thread_block();
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline void sync(const thread_block& group) {
  group.sync();
}

template <unsigned int Size, typename ParentT = void> class thread_block_tile;
namespace ascify_detail {
template <unsigned int Size> class tile_register_operations {
  static_assert(Size == 32, "Ascify admits only full-warp tile size 32; multi-warp subgroup synchronization is not implemented");
 protected:
  using native_tile = ::cooperative_groups::thread_block_tile<Size, thread_block>;
  native_tile native_;
  __SIMT_DEVICE_FUNCTIONS_DECL__ explicit tile_register_operations(native_tile value)
      : native_(value) {}
 public:
  __SIMT_DEVICE_FUNCTIONS_DECL__ static unsigned int thread_rank() {
    return static_cast<unsigned int>(native_tile::thread_rank());
  }
  __SIMT_DEVICE_FUNCTIONS_DECL__ static constexpr unsigned int size() { return Size; }
  __SIMT_DEVICE_FUNCTIONS_DECL__ void sync() const {
    // The verified 950 native tile sync is only a block-scope memory fence.
    // Fence publication, rendezvous of every active warp lane, then fence
    // subsequent memory access. A complete tile must reach this call together;
    // never substitute a block barrier when another warp bypasses this call.
    if (asc_activemask() != 0xffffffffu) {
      ::ascify::detail::reject_device_contract();
    }
    asc_threadfence_block();
    if (asc_ballot(1) != 0xffffffffu) {
      ::ascify::detail::reject_device_contract();
    }
    asc_threadfence_block();
  }
  template <typename T> __SIMT_DEVICE_FUNCTIONS_DECL__
  typename std::enable_if<std::is_same<T, int>::value ||
      std::is_same<T, unsigned int>::value || std::is_same<T, float>::value, T>::type
  shfl_up(T value, unsigned int delta) const { return native_.shfl_up(value, delta); }
  template <typename T> __SIMT_DEVICE_FUNCTIONS_DECL__
  typename std::enable_if<std::is_same<T, int>::value ||
      std::is_same<T, unsigned int>::value || std::is_same<T, float>::value ||
      ::ascify::detail::scalar64_transport_type<T>::value, T>::type
  shfl(T value, int source_rank) const {
    if constexpr (::ascify::detail::scalar64_transport_type<T>::value)
      return ::ascify::detail::scalar64_shfl_index(UINT32_MAX, value, source_rank, 32);
    else return native_.shfl(value, source_rank);
  }
  template <typename T> __SIMT_DEVICE_FUNCTIONS_DECL__
  typename std::enable_if<std::is_same<T, int>::value ||
      std::is_same<T, unsigned int>::value || std::is_same<T, float>::value ||
      ::ascify::detail::scalar64_transport_type<T>::value, T>::type
  shfl_down(T value, unsigned int delta) const {
    if constexpr (::ascify::detail::scalar64_transport_type<T>::value)
      return ::ascify::detail::scalar64_shfl_down(UINT32_MAX, value, delta, 32);
    else return native_.shfl_down(value, delta);
  }
  template <typename T> __SIMT_DEVICE_FUNCTIONS_DECL__
  typename std::enable_if<std::is_same<T, int>::value ||
      std::is_same<T, unsigned int>::value || std::is_same<T, float>::value, T>::type
  shfl_xor(T value, unsigned int lane_mask) const { return native_.shfl_xor(value, lane_mask); }
  template <typename T> __SIMT_DEVICE_FUNCTIONS_DECL__
  typename std::enable_if<std::is_same<T, uint4>::value, T>::type
  shfl_xor(T value, unsigned int lane_mask) const {
    static_assert(sizeof(unsigned int) == 4 && sizeof(uint4) == 16,
                  "uint4 shuffle requires four 32-bit components");
    uint4 result;
    result.x = native_.shfl_xor(value.x, lane_mask);
    result.y = native_.shfl_xor(value.y, lane_mask);
    result.z = native_.shfl_xor(value.z, lane_mask);
    result.w = native_.shfl_xor(value.w, lane_mask);
    return result;
  }
};
// The source publication proof restricts this helper to a complete physical
// warp reaching the same call/epoch. It is not a named partial-mask barrier.
__SIMT_DEVICE_FUNCTIONS_DECL__ inline void full_warp_group_sync_v1() {
  if (asc_activemask() != 0xffffffffu)
    ::ascify::detail::reject_device_contract();
  asc_threadfence_block();
  if (asc_ballot(1) != 0xffffffffu)
    ::ascify::detail::reject_device_contract();
  asc_threadfence_block();
}

// Deliberately separate from the existing Size==32 register facade: tile16
// gains only rank/size/sync, never shuffle, reduce or scalar64 transport.
template <> class tile_register_operations<16> {
 protected:
  using native_tile = ::cooperative_groups::thread_block_tile<16, thread_block>;
  native_tile native_;
  __SIMT_DEVICE_FUNCTIONS_DECL__ explicit tile_register_operations(native_tile value)
      : native_(value) {}
 public:
  __SIMT_DEVICE_FUNCTIONS_DECL__ static unsigned int thread_rank() {
    return static_cast<unsigned int>(native_tile::thread_rank());
  }
  __SIMT_DEVICE_FUNCTIONS_DECL__ static constexpr unsigned int size() { return 16; }
  __SIMT_DEVICE_FUNCTIONS_DECL__ void sync() const { full_warp_group_sync_v1(); }
};
}  // namespace ascify_detail

template <unsigned int Size, typename ParentT>
class thread_block_tile : public ascify_detail::tile_register_operations<Size> {
  static_assert(std::is_same<ParentT, thread_block>::value,
                "Ascify admits only tiles partitioned from a thread block");
  using base = ascify_detail::tile_register_operations<Size>;
  friend class thread_block_tile<Size, void>;
 public:
  __SIMT_DEVICE_FUNCTIONS_DECL__ explicit thread_block_tile(typename base::native_tile value)
      : base(value) {}
  __SIMT_DEVICE_FUNCTIONS_DECL__ operator thread_block_tile<Size, void>() const {
    return thread_block_tile<Size, void>(*this);
  }
  __SIMT_DEVICE_FUNCTIONS_DECL__ static unsigned int meta_group_rank() {
    return thread_block::thread_rank() / Size;
  }
  __SIMT_DEVICE_FUNCTIONS_DECL__ static unsigned int meta_group_size() {
    return (thread_block::size() + Size - 1) / Size;
  }
};

template <unsigned int Size>
class thread_block_tile<Size, void> : public ascify_detail::tile_register_operations<Size> {
  using base = ascify_detail::tile_register_operations<Size>;
  unsigned int rank_;
  unsigned int count_;
 public:
  template <typename ParentT, typename std::enable_if<
      std::is_same<ParentT, thread_block>::value, int>::type = 0>
  __SIMT_DEVICE_FUNCTIONS_DECL__ thread_block_tile(const thread_block_tile<Size, ParentT>& value)
      : base(value.native_), rank_(value.meta_group_rank()), count_(value.meta_group_size()) {}
  __SIMT_DEVICE_FUNCTIONS_DECL__ unsigned int meta_group_rank() const { return rank_; }
  __SIMT_DEVICE_FUNCTIONS_DECL__ unsigned int meta_group_size() const { return count_; }
};

template <unsigned int Size>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline thread_block_tile<Size, thread_block>
tiled_partition(const thread_block& parent) {
  return thread_block_tile<Size, thread_block>(
      ::cooperative_groups::tiled_partition<Size>(parent));
}

template <unsigned int Size, typename ParentT>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline void
sync(const thread_block_tile<Size, ParentT>& group) {
  group.sync();
}

// Only authenticated block/tile16 origins and closed by-value copies are
// admitted by the mandatory source proof. Do not downcast a sliced native
// thread_group to a derived object, or expose representation to caller code.
class thread_group final {
  enum class kind { block, full_warp_tile16 };
  kind kind_;
 public:
  __SIMT_DEVICE_FUNCTIONS_DECL__ thread_group(const thread_block&)
      : kind_(kind::block) {}
  template <typename ParentT, typename std::enable_if<
      std::is_same<ParentT, thread_block>::value ||
      std::is_same<ParentT, void>::value, int>::type = 0>
  __SIMT_DEVICE_FUNCTIONS_DECL__ thread_group(const thread_block_tile<16, ParentT>&)
      : kind_(kind::full_warp_tile16) {}
  // The real CUDA erased group uses ULL; static block/tile methods use uint.
  __SIMT_DEVICE_FUNCTIONS_DECL__ unsigned long long size() const {
    return kind_ == kind::block ? thread_block::size() : 16u;
  }
  __SIMT_DEVICE_FUNCTIONS_DECL__ unsigned long long thread_rank() const {
    return kind_ == kind::block ? thread_block::thread_rank()
        : ascify_detail::tile_register_operations<16>::thread_rank();
  }
  __SIMT_DEVICE_FUNCTIONS_DECL__ void sync() const {
    if (kind_ == kind::block) thread_block::sync();
    else ascify_detail::full_warp_group_sync_v1();
  }
};

__SIMT_DEVICE_FUNCTIONS_DECL__ inline void sync(const thread_group& group) {
  group.sync();
}

template <typename T> struct plus {
  static_assert(std::is_same<T, int>::value || std::is_same<T, float>::value ||
                std::is_same<T, double>::value,
                "Ascify cooperative plus supports int, float and exact binary64 double");
  __SIMT_DEVICE_FUNCTIONS_DECL__ T operator()(T left, T right) const {
    return ::ascify::detail::scalar_add_rne<T>(left, right);
  }
};

template <unsigned int Size, typename ParentT, typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline
 typename std::enable_if<Size == 32 && (std::is_same<T, int>::value ||
                                     std::is_same<T, float>::value ||
                                     std::is_same<T, double>::value), T>::type
reduce(const thread_block_tile<Size, ParentT>& group, T value, plus<T> operation) {
  // A complete tile must execute this register collective together. In
  // particular, do not substitute a block barrier when another warp diverges.
  if (asc_activemask() != 0xffffffffu) { ::ascify::detail::reject_device_contract(); }
  for (unsigned int offset = Size / 2; offset != 0; offset >>= 1) {
    value = operation(value, group.shfl_down(value, offset));
  }
  return group.shfl(value, 0);
}

}  // namespace ascify_cg
#include <ascify/uniform_block_reduction_compat.hpp>
#endif

#endif  // ASCIFY_COOPERATIVE_GROUPS_COMPAT_HPP
