#ifndef ASCIFY_DEVICE_MEMORY_COMPAT_HPP
#define ASCIFY_DEVICE_MEMORY_COMPAT_HPP

#include <stddef.h>

namespace ascify {

// Internal lowering target: the frontend proves two distinct private,
// trivially-copyable 64-byte objects and sizeof(destination). As with memcpy,
// the source program must provide valid object lifetimes. This function
// adds no memory barrier, allocation, address-space cast, or wider access.
template<size_t Bytes, typename Destination, typename Source, size_t Count>
__aicore__ inline __attribute__((always_inline)) void *device_memcpy_private(
    Destination *destination, const Source (&source)[Count]) {
  static_assert(Bytes == 64, "Ascify private memcpy currently admits exactly 64 bytes");
  static_assert(sizeof(Destination) == Bytes && sizeof(source) == Bytes,
                "Ascify private memcpy requires the proved target object sizes");
  unsigned char *out = reinterpret_cast<unsigned char *>(destination);
  const unsigned char *in = reinterpret_cast<const unsigned char *>(source);
#if defined(__NPU_ARCH__)
#pragma unroll
#endif
  for (size_t index = 0; index < Bytes; ++index)
    out[index] = in[index];
  return destination;
}

} // namespace ascify

#endif
