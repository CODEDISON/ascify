#ifndef ASCIFY_REGISTER_TILE32_COMPAT_HPP
#define ASCIFY_REGISTER_TILE32_COMPAT_HPP

#include <ascify/cooperative_groups_compat.hpp>

namespace ascify_cg {

// An owned register-exchange view with nominal width32. The final physical
// warp may have a contiguous prefix of members. This type does not construct
// a native cooperative_groups tile and provides no tile barrier or reduction.
class register_tile32_v1 {
  unsigned int block_rank_;
  unsigned int block_size_;
  unsigned int member_mask_;

  __SIMT_DEVICE_FUNCTIONS_DECL__ __attribute__((always_inline)) void check_active() const {
    if (asc_activemask() != member_mask_)
      ::ascify::detail::reject_device_contract();
  }

  __SIMT_DEVICE_FUNCTIONS_DECL__ __attribute__((always_inline)) void check_xor(unsigned int lane_mask) const {
    check_active();
    const unsigned int source = thread_rank() ^ lane_mask;
    if (lane_mask >= 32 || source >= 32 ||
        (member_mask_ & (1u << source)) == 0)
      ::ascify::detail::reject_device_contract();
  }

public:
  __SIMT_DEVICE_FUNCTIONS_DECL__ __attribute__((always_inline)) explicit register_tile32_v1(
      const thread_block& parent)
      : block_rank_(static_cast<unsigned int>(parent.thread_rank())),
        block_size_(static_cast<unsigned int>(parent.size())), member_mask_(0) {
    if (parent.size() == 0 || parent.size() > 1024 ||
        parent.thread_rank() >= parent.size())
      ::ascify::detail::reject_device_contract();
    const unsigned int remaining = block_size_ - (block_rank_ & ~31u);
    const unsigned int members = remaining < 32 ? remaining : 32;
    member_mask_ = members == 32 ? 0xffffffffu : (1u << members) - 1u;
    check_active();
  }

  register_tile32_v1(const register_tile32_v1&) = delete;
  register_tile32_v1& operator=(const register_tile32_v1&) = delete;

  __SIMT_DEVICE_FUNCTIONS_DECL__ __attribute__((always_inline)) unsigned int thread_rank() const {
    return block_rank_ & 31u;
  }
  __SIMT_DEVICE_FUNCTIONS_DECL__ __attribute__((always_inline)) static constexpr unsigned int size() {
    return 32;
  }
  __SIMT_DEVICE_FUNCTIONS_DECL__ __attribute__((always_inline)) unsigned int meta_group_rank() const {
    return block_rank_ >> 5;
  }
  __SIMT_DEVICE_FUNCTIONS_DECL__ __attribute__((always_inline)) unsigned int meta_group_size() const {
    return (block_size_ + 31u) >> 5;
  }

  __SIMT_DEVICE_FUNCTIONS_DECL__ __attribute__((always_inline)) unsigned int shfl_up(
      unsigned int value, unsigned int delta) const {
    check_active();
    const unsigned int normalized_delta = delta < 32 ? delta : 0;
    return ::asc_shfl_up(value, normalized_delta, 32);
  }

  __SIMT_DEVICE_FUNCTIONS_DECL__ __attribute__((always_inline)) uint4 shfl_xor(
      uint4 value, unsigned int lane_mask) const {
    static_assert(sizeof(unsigned int) == 4 && sizeof(uint4) == 16 &&
                      alignof(uint4) == 16,
                  "register tile requires the physical four-component uint4");
    check_xor(lane_mask);
    uint4 result;
    result.x = ::asc_shfl_xor(value.x, lane_mask, 32);
    result.y = ::asc_shfl_xor(value.y, lane_mask, 32);
    result.z = ::asc_shfl_xor(value.z, lane_mask, 32);
    result.w = ::asc_shfl_xor(value.w, lane_mask, 32);
    return result;
  }
};

__SIMT_DEVICE_FUNCTIONS_DECL__ __attribute__((always_inline)) inline register_tile32_v1
register_partition32_v1(const thread_block& parent) {
  return register_tile32_v1(parent);
}

}  // namespace ascify_cg

#endif
