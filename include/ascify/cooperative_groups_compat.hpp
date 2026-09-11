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

template <unsigned int Size, typename ParentT = void>
class thread_block_tile {
  static_assert(Size == 32, "Ascify admits only full-warp tile size 32");
  static_assert(std::is_same<ParentT, thread_block>::value,
                "Ascify admits only tiles partitioned from a thread block");
  using native_tile =
      ::cooperative_groups::thread_block_tile<Size, thread_block>;
  native_tile native_;

 public:
  __SIMT_DEVICE_FUNCTIONS_DECL__ explicit thread_block_tile(native_tile value)
      : native_(value) {}
  __SIMT_DEVICE_FUNCTIONS_DECL__ static unsigned int thread_rank() {
    return static_cast<unsigned int>(native_tile::thread_rank());
  }
  __SIMT_DEVICE_FUNCTIONS_DECL__ static constexpr unsigned int size() {
    return Size;
  }
  __SIMT_DEVICE_FUNCTIONS_DECL__ static unsigned int meta_group_rank() {
    return thread_block::thread_rank() / Size;
  }
  template <typename T>
  __SIMT_DEVICE_FUNCTIONS_DECL__
  typename std::enable_if<std::is_same<T, int>::value ||
                              std::is_same<T, unsigned int>::value ||
                              std::is_same<T, float>::value, T>::type
  shfl_up(T value, unsigned int delta) const {
    return native_.shfl_up(value, delta);
  }
  template <typename T>
  __SIMT_DEVICE_FUNCTIONS_DECL__
  typename std::enable_if<std::is_same<T, int>::value ||
                              std::is_same<T, unsigned int>::value ||
                              std::is_same<T, float>::value, T>::type
  shfl_xor(T value, unsigned int lane_mask) const {
    return native_.shfl_xor(value, lane_mask);
  }
};

template <unsigned int Size>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline thread_block_tile<Size, thread_block>
tiled_partition(const thread_block& parent) {
  return thread_block_tile<Size, thread_block>(
      ::cooperative_groups::tiled_partition<Size>(parent));
}

}  // namespace ascify_cg
#endif

#endif  // ASCIFY_COOPERATIVE_GROUPS_COMPAT_HPP
