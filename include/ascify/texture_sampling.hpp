#ifndef ASCIFY_TEXTURE_SAMPLING_HPP
#define ASCIFY_TEXTURE_SAMPLING_HPP

#include <ascify/ascify_cuda_compat.hpp>

namespace ascify {
namespace detail {

// Private host/device ABI. The public CUDA token remains an opaque 64-bit
// integer; every nonzero address below is owned by the real ACL allocator.
struct TextureSamplingDescriptor {
  uint64_t data;
  uint64_t pitch_bytes;
  uint64_t layer_bytes;
  uint32_t width;
  uint32_t height;
  uint32_t layers;
  uint32_t normalized;
  uint32_t wrap_x;
  uint32_t wrap_y;
  uint32_t linear;
  uint32_t layered;
  uint64_t error;
};
static_assert(sizeof(TextureSamplingDescriptor) == 64, "texture descriptor ABI");
static_assert(alignof(TextureSamplingDescriptor) == 8, "texture descriptor alignment");
static_assert(offsetof(TextureSamplingDescriptor, width) == 24, "texture dimension ABI");
static_assert(offsetof(TextureSamplingDescriptor, error) == 56, "texture error ABI");
static_assert(std::is_trivially_copyable<TextureSamplingDescriptor>::value,
              "texture descriptor must be copied without host pointers");

#if defined(ASCIFY_SIMT_LEGACY_HAS_GM_ATOMICS) && \
    defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)

__aicore__ ASCIFY_FORCEINLINE float textureSamplingReject(
    const TextureSamplingDescriptor *descriptor) {
  // Keep the device healthy on a domain rejection. The blocking host API
  // reads this sticky GM status after synchronization and returns an error.
  atomic_or_global(reinterpret_cast<uint32_t *>(descriptor->error), 1U);
  return __builtin_nanf("");
}

__aicore__ ASCIFY_FORCEINLINE int textureAddressIndex(int value, int extent,
                                                    bool wrap) {
  if (!wrap) return value < 0 ? 0 : value >= extent ? extent - 1 : value;
  // The coordinate reduction below gives [-1,extent]. Avoid signed remainder
  // and preserve the last/first texel pair at a linear-filter wrap seam.
  return value < 0 ? extent - 1 : value >= extent ? 0 : value;
}

__aicore__ ASCIFY_FORCEINLINE float textureLoadFloat(
    const TextureSamplingDescriptor *descriptor, int x, int y, int layer) {
  x = textureAddressIndex(x, static_cast<int>(descriptor->width), descriptor->wrap_x != 0);
  y = textureAddressIndex(y, static_cast<int>(descriptor->height), descriptor->wrap_y != 0);
  const uint64_t address = descriptor->data + uint64_t(layer) * descriptor->layer_bytes +
      uint64_t(y) * descriptor->pitch_bytes + uint64_t(x) * sizeof(float);
  const float value = *reinterpret_cast<const float *>(address);
  // CUDA hardware's non-finite texel/filter behavior is outside the admitted
  // numerical contract. Surface it through the same observable sticky error.
  return __builtin_isfinite(value) ? value : textureSamplingReject(descriptor);
}

__aicore__ ASCIFY_FORCEINLINE float textureCoordinate(
    float coordinate, uint32_t extent, bool normalized, bool wrap) {
  if (normalized) {
    if (wrap) coordinate -= floorf(coordinate);
    else coordinate = coordinate < 0.0f ? 0.0f : coordinate > 1.0f ? 1.0f : coordinate;
    return coordinate * static_cast<float>(extent);
  }
  // Unnormalized wrap is normalized to clamp when constructing the descriptor.
  const float bound = static_cast<float>(extent);
  return coordinate < 0.0f ? 0.0f : coordinate > bound ? bound : coordinate;
}

template<bool Approximate = false>
__aicore__ ASCIFY_FORCEINLINE float textureSampleFloat(
    unsigned long long token, float x, float y, int layer, bool layered) {
  const auto *descriptor = reinterpret_cast<const TextureSamplingDescriptor *>(token);
  if (!descriptor->width || !descriptor->height ||
      !__builtin_isfinite(x) || !__builtin_isfinite(y) ||
      (descriptor->layered != 0) != layered || layer < 0 ||
      static_cast<uint32_t>(layer) >= descriptor->layers)
    return textureSamplingReject(descriptor);
  x = textureCoordinate(x, descriptor->width, descriptor->normalized != 0,
                        descriptor->wrap_x != 0);
  y = textureCoordinate(y, descriptor->height, descriptor->normalized != 0,
                        descriptor->wrap_y != 0);
  if (!descriptor->linear) {
    return textureLoadFloat(descriptor, static_cast<int>(floorf(x)),
                            static_cast<int>(floorf(y)), layer);
  }
  const float xb = x - 0.5f, yb = y - 0.5f;
  const float xf = floorf(xb), yf = floorf(yb);
  const float a = xb - xf, b = yb - yf;
  const float aq = a * 256.0f, bq = b * 256.0f;
  // Admit exactly representable 8-fraction-bit weights. No rounding tie or
  // undocumented coordinate quantization is guessed for other inputs.
  // The separately named fast wrappers opt into FP32 bilinear filtering.
  // They do not emulate CUDA's fixed-point coordinate quantization or ties.
  if (!Approximate && (aq != floorf(aq) || bq != floorf(bq)))
    return textureSamplingReject(descriptor);
  const int ix = static_cast<int>(xf), iy = static_cast<int>(yf);
  const float t00 = textureLoadFloat(descriptor, ix, iy, layer);
  // Only the nonzero-weight texels participate in this bounded finite sampler.
  const float row0 = a == 0.0f ? t00 :
      (1.0f - a) * t00 + a * textureLoadFloat(descriptor, ix + 1, iy, layer);
  if (b == 0.0f) return row0;
  const float t01 = textureLoadFloat(descriptor, ix, iy + 1, layer);
  const float row1 = a == 0.0f ? t01 :
      (1.0f - a) * t01 + a * textureLoadFloat(descriptor, ix + 1, iy + 1, layer);
  return (1.0f - b) * row0 + b * row1;
}
#endif

} // namespace detail
} // namespace ascify
#endif
