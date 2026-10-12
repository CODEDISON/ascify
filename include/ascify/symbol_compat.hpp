#ifndef ASCIFY_SYMBOL_COMPAT_HPP
#define ASCIFY_SYMBOL_COMPAT_HPP

#include "ascify_cuda_compat.hpp"

#include <acl/acl.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

namespace ascify {

// Defined by ascify_cuda_compat.hpp. Every call reuses its current-device
// lifecycle handling; symbol identities and sizes are never cached here.
inline aclError cudaRuntimeEnsureReady();

namespace detail {

// DPP emits real device globals, but some CCEC versions do not register their
// host tokens with ACL. The converter supplies an address kernel in the same
// translation unit as each admitted definition. Retain only its host callback:
// device addresses must be resolved again after a device/context/reset change.
struct DeviceSymbolResolver {
  using Resolve = aclError (*)(void **);
  const void *token;
  std::size_t bytes;
  Resolve resolve;
  DeviceSymbolResolver *next;

  static DeviceSymbolResolver *&head() {
    static DeviceSymbolResolver *value = nullptr;
    return value;
  }
  static std::atomic_flag &lock() {
    static std::atomic_flag value = ATOMIC_FLAG_INIT;
    return value;
  }
  struct Guard {
    Guard() { while (lock().test_and_set(std::memory_order_acquire)) {} }
    ~Guard() { lock().clear(std::memory_order_release); }
  };
  DeviceSymbolResolver(const void *identity, std::size_t size, Resolve callback)
      : token(identity), bytes(size), resolve(callback), next(nullptr) {
    Guard guard;
    next = head();
    head() = this;
  }
  ~DeviceSymbolResolver() {
    Guard guard;
    auto **current = &head();
    while (*current != nullptr && *current != this) current = &(*current)->next;
    if (*current == this) *current = next;
  }
  DeviceSymbolResolver(const DeviceSymbolResolver &) = delete;
  DeviceSymbolResolver &operator=(const DeviceSymbolResolver &) = delete;
  static Resolve find(const void *identity, std::size_t &size) {
    Guard guard;
    for (auto *entry = head(); entry != nullptr; entry = entry->next) {
      if (entry->token == identity) {
        size = entry->bytes;
        return entry->resolve;
      }
    }
    return nullptr;
  }
};

inline aclError copyResolvedSymbol(
    DeviceSymbolResolver::Resolve resolve, std::size_t bytes, void *destination,
    const void *source, std::size_t count, std::size_t offset,
    aclrtMemcpyKind kind, bool to_symbol) {
  if (offset > bytes || count > bytes - offset) return ACL_ERROR_RT_PARAM_INVALID;
  void *address = nullptr;
  const aclError status = resolve(&address);
  if (status != ACL_SUCCESS) return status;
  if (address == nullptr) return ACL_ERROR_RT_PARAM_INVALID;
  auto *at_offset = static_cast<unsigned char *>(address) + offset;
  return to_symbol ? ::aclrtMemcpy(at_offset, bytes - offset, source, count, kind)
                   : ::aclrtMemcpy(destination, count, at_offset, count, kind);
}

// ACL version numbers do not guarantee these optional declarations. Keep the
// argument dependent until instantiation, but resolve it to a built-in pointer
// so a symbol type's namespace cannot supply substitute APIs through ADL.
template <typename>
struct SymbolTokenPointer {
  using type = const void *;
};

// The SDK enum belongs to the global namespace. Passing it directly would
// allow ADL to find an application declaration added after this header.
struct SymbolCopyKind {
  aclrtMemcpyKind value;
  operator aclrtMemcpyKind() const { return value; }
};

template <typename T, typename = void>
struct HasSymbolCopyApi : std::false_type {};

template <typename T>
struct HasSymbolCopyApi<T, std::void_t<
    decltype(aclrtGetSymbolSize(
        std::declval<typename SymbolTokenPointer<T>::type>(),
        std::declval<std::size_t *>())),
    decltype(aclrtMemcpyToSymbol(
        std::declval<typename SymbolTokenPointer<T>::type>(),
        std::declval<const void *>(), std::declval<std::size_t>(),
        std::declval<std::size_t>(), std::declval<SymbolCopyKind>()))>>
    : std::true_type {};

template <typename T, typename = void>
struct HasSymbolReadApi : std::false_type {};

template <typename T>
struct HasSymbolReadApi<T, std::void_t<
    decltype(aclrtGetSymbolSize(
        std::declval<typename SymbolTokenPointer<T>::type>(),
        std::declval<std::size_t *>())),
    decltype(aclrtMemcpyFromSymbol(
        std::declval<void *>(), std::declval<std::size_t>(),
        std::declval<typename SymbolTokenPointer<T>::type>(),
        std::declval<std::size_t>(), std::declval<std::size_t>(),
        std::declval<SymbolCopyKind>()))>> : std::true_type {};

}  // namespace detail

// Keep a reference to the symbol object: array decay loses the distinction
// between a registered symbol token and an ordinary pointer's stored value.
// CCEC owns the original __constant__/device storage. Generated resolvers
// address that exact storage; other tokens retain the native SDK error path.
template <typename T>
inline aclError cudaMemcpyToSymbol(
    const T &symbol, const void *source, std::size_t count,
    std::size_t offset = 0,
    aclrtMemcpyKind kind = ACL_MEMCPY_HOST_TO_DEVICE) {
#if defined(ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3)
  if constexpr (detail::HasSymbolCopyApi<T>::value) {
    if (kind != ACL_MEMCPY_HOST_TO_DEVICE &&
        kind != ACL_MEMCPY_DEVICE_TO_DEVICE) {
      return ACL_ERROR_RT_PARAM_INVALID;
    }
    if (count != 0 && source == nullptr) {
      return ACL_ERROR_RT_PARAM_INVALID;
    }
    const aclError ready = cudaRuntimeEnsureReady();
    if (ready != ACL_SUCCESS) return ready;

    typename detail::SymbolTokenPointer<T>::type token =
        static_cast<const void *>(std::addressof(symbol));
    std::size_t symbol_bytes = 0;
    if (auto resolve = detail::DeviceSymbolResolver::find(token, symbol_bytes)) {
      return detail::copyResolvedSymbol(resolve, symbol_bytes, nullptr, source,
                                         count, offset, kind, true);
    }
    const aclError size_status = aclrtGetSymbolSize(token, &symbol_bytes);
    if (size_status != ACL_SUCCESS) return size_status;
    if (offset > symbol_bytes || count > symbol_bytes - offset) {
      return ACL_ERROR_RT_PARAM_INVALID;
    }

    // Even a zero-byte call validates its symbol and delegates to the SDK.
    // Never emulate a device write with host memory or fabricate success.
    return aclrtMemcpyToSymbol(token, source, count, offset,
                               detail::SymbolCopyKind{kind});
  } else
#endif
  {
    (void)symbol;
    (void)source;
    (void)count;
    (void)offset;
    (void)kind;
    static_assert(!std::is_same<T, T>::value,
                  "Ascify symbol copies require legacy SIMT headers and SDK "
                  "declarations of aclrtGetSymbolSize and aclrtMemcpyToSymbol");
    __builtin_unreachable();
  }
}

template <typename T>
inline aclError cudaMemcpyFromSymbol(
    void *destination, const T &symbol, std::size_t count,
    std::size_t offset = 0,
    aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_HOST) {
#if defined(ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3)
  if constexpr (detail::HasSymbolReadApi<T>::value) {
    if ((kind != ACL_MEMCPY_DEVICE_TO_HOST && kind != ACL_MEMCPY_DEVICE_TO_DEVICE) ||
        (count != 0 && destination == nullptr)) return ACL_ERROR_RT_PARAM_INVALID;
    const aclError ready = cudaRuntimeEnsureReady();
    if (ready != ACL_SUCCESS) return ready;
    typename detail::SymbolTokenPointer<T>::type token =
        static_cast<const void *>(std::addressof(symbol));
    std::size_t symbol_bytes = 0;
    if (auto resolve = detail::DeviceSymbolResolver::find(token, symbol_bytes)) {
      return detail::copyResolvedSymbol(resolve, symbol_bytes, destination, nullptr,
                                         count, offset, kind, false);
    }
    const aclError status = aclrtGetSymbolSize(token, &symbol_bytes);
    if (status != ACL_SUCCESS) return status;
    if (offset > symbol_bytes || count > symbol_bytes - offset)
      return ACL_ERROR_RT_PARAM_INVALID;
    return aclrtMemcpyFromSymbol(destination, count, token, count, offset,
                                 detail::SymbolCopyKind{kind});
  } else
#endif
  {
    (void)destination; (void)symbol; (void)count; (void)offset; (void)kind;
    static_assert(!std::is_same<T, T>::value,
                  "Ascify symbol reads require legacy SIMT headers and SDK "
                  "declarations of aclrtGetSymbolSize and aclrtMemcpyFromSymbol");
    __builtin_unreachable();
  }
}

}  // namespace ascify

// Emitted immediately after an admitted device-global definition. The address
// kernel and the original consumer kernels are compiled into the same module.
// A real default-stream synchronization completes the address write before the
// host reads it. All allocation, launch, synchronization, copy and free errors
// propagate; no device address or runtime success is fabricated.
#define ASCIFY_REGISTER_DEVICE_SYMBOL(symbol, tag)                            \
  static ASCIFY_GLOBAL void tag##_address(::std::uintptr_t *output) {                 \
    if (::threadIdx.x == 0) *output = reinterpret_cast<::std::uintptr_t>(__builtin_addressof(symbol));     \
  }                                                                          \
  static ::aclError tag##_resolve(void **address) {                              \
    ::std::uintptr_t *device_result = nullptr;                                       \
    ::std::uintptr_t host_result = 0;                                                \
    ::aclError status = ::ascify::cudaMalloc(                                    \
        reinterpret_cast<void **>(&device_result), sizeof(host_result));      \
    if (status != ::ACL_SUCCESS) return status;                                 \
    tag##_address<<<1, 32, 0, nullptr>>>(device_result);                        \
    status = ::ascify::cudaGetLastError();                                     \
    if (status == ::ACL_SUCCESS) status = ::ascify::cudaStreamSynchronize(nullptr); \
    if (status == ::ACL_SUCCESS) status = ::ascify::cudaMemcpy(                   \
        &host_result, device_result, sizeof(host_result), ::ACL_MEMCPY_DEVICE_TO_HOST); \
    const ::aclError released = ::ascify::cudaFree(device_result);               \
    if (status == ::ACL_SUCCESS) status = released;                              \
    if (status == ::ACL_SUCCESS) *address = reinterpret_cast<void *>(host_result); \
    return status;                                                           \
  }                                                                          \
  static ::ascify::detail::DeviceSymbolResolver tag##_registration(            \
      static_cast<const void *>(__builtin_addressof(symbol)), sizeof(symbol), &tag##_resolve)

#endif  // ASCIFY_SYMBOL_COMPAT_HPP
