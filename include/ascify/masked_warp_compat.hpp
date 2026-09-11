#ifndef ASCIFY_MASKED_WARP_COMPAT_HPP
#define ASCIFY_MASKED_WARP_COMPAT_HPP

#include <stdint.h>
#include <type_traits>
#include <simt_api/device_warp_functions.h>
#include <ascify/device_contract_reject.hpp>

namespace ascify {
namespace detail {

// The legacy CANN primitives have no CUDA mask or rendezvous argument. Admit
// only a group that is already executing together: every active lane must pass
// exactly the current active mask. The ballot makes rejection collective even
// when only one lane supplies an inconsistent mask. This is a restricted
// implementation, not a barrier for lanes that have yet to reach this call.
__SIMT_DEVICE_FUNCTIONS_DECL__ inline void require_converged_warp_mask(
    uint32_t mask) {
  const uint32_t active = asc_activemask();
  const uint32_t invalid = asc_ballot(mask != active || active == 0);
  if (invalid != 0 || active == 0) {
    reject_device_contract();
  }
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline uint32_t converged_ballot_sync(
    uint32_t mask, int predicate) {
  require_converged_warp_mask(mask);
  return asc_ballot(predicate != 0) & mask;
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T converged_shfl_down_sync(
    uint32_t mask, T value, unsigned int delta, int width = 32) {
  static_assert(std::is_same<T, int32_t>::value ||
                    std::is_same<T, uint32_t>::value ||
                    std::is_same<T, float>::value,
                "Ascify converged shuffle supports int32_t, uint32_t and float only");
  require_converged_warp_mask(mask);
  // tile<1> is an admitted CANN tile and forwards width=1 to this primitive.
  // Validate collectively before any lane can leave for an invalid width.
  const bool valid_width = width == 1 || width == 2 || width == 4 ||
                           width == 8 || width == 16 || width == 32;
  if (asc_ballot(!valid_width) != 0) {
    reject_device_contract();
  }
  // Preserve physical lane addressing and native width boundaries. A source
  // lane outside mask has an undefined CUDA result; never substitute zero or
  // compact the selected lanes into a different group.
  return asc_shfl_down(value, delta, width);
}

}  // namespace detail
}  // namespace ascify

#endif  // ASCIFY_MASKED_WARP_COMPAT_HPP
