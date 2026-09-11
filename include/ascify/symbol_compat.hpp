#ifndef ASCIFY_SYMBOL_COMPAT_HPP
#define ASCIFY_SYMBOL_COMPAT_HPP

#include "ascify_cuda_compat.hpp"

#include <acl/acl.h>
#include <cstddef>
#include <memory>
#include <type_traits>

namespace ascify {

// Defined by ascify_cuda_compat.hpp. Every call reuses its current-device
// lifecycle handling; symbol identities and sizes are never cached here.
inline aclError cudaRuntimeEnsureReady();

// Keep a reference to the symbol object: array decay loses the distinction
// between a registered symbol token and an ordinary pointer's stored value.
// CCEC owns the original __constant__/device declaration and registration.
template <typename T>
inline aclError cudaMemcpyToSymbol(
    const T &symbol, const void *source, std::size_t count,
    std::size_t offset = 0,
    aclrtMemcpyKind kind = ACL_MEMCPY_HOST_TO_DEVICE) {
#if defined(ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3) && \
    defined(ACL_MAJOR_VERSION) && defined(ACL_MINOR_VERSION) && \
    ((ACL_MAJOR_VERSION > 1) || \
     (ACL_MAJOR_VERSION == 1 && ACL_MINOR_VERSION >= 17))
  if (kind != ACL_MEMCPY_HOST_TO_DEVICE &&
      kind != ACL_MEMCPY_DEVICE_TO_DEVICE) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  if (count != 0 && source == nullptr) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) return ready;

  const void *token = static_cast<const void *>(std::addressof(symbol));
  std::size_t symbol_bytes = 0;
  const aclError size_status = aclrtGetSymbolSize(token, &symbol_bytes);
  if (size_status != ACL_SUCCESS) return size_status;
  if (offset > symbol_bytes || count > symbol_bytes - offset) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }

  // Even a zero-byte call validates its symbol and delegates to the SDK.
  // Never emulate a device write with host memory or fabricate success.
  return aclrtMemcpyToSymbol(token, source, count, offset, kind);
#else
  (void)symbol;
  (void)source;
  (void)count;
  (void)offset;
  (void)kind;
  static_assert(!std::is_same<T, T>::value,
                "Ascify symbol copies require the admitted CANN 9.1 "
                "legacy SIMT / ACL 1.17 symbol API");
  __builtin_unreachable();
#endif
}

}  // namespace ascify

#endif  // ASCIFY_SYMBOL_COMPAT_HPP
