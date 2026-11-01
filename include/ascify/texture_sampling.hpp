#ifndef ASCIFY_TEXTURE_SAMPLING_HPP
#define ASCIFY_TEXTURE_SAMPLING_HPP

#include <ascify/texture_types.hpp>

namespace ascify {
namespace detail {

// Every v2 kind is nonzero, so a v1 float helper rejects it. Every v2 filter
// encoding is nonzero, so a v1 unsigned helper rejects it as well. Both old
// helpers still find the sticky error pointer at offset 64. Version alone in
// the old padding would not protect old readers from a changed channel kind.
enum TextureSampleKind : uint32_t {
  TextureSampleFloat = 0x54580211U,
  TextureSampleUnsigned = 0x54580212U,
  TextureSampleCubeFloat = 0x54580231U,
  TextureSampleRGBAElement = 0x54580221U,
  TextureSampleRGBANormalized = 0x54580222U
};
inline constexpr uint32_t TextureDescriptorVersion = 0x54580002U;
inline constexpr uint32_t TextureFilterPoint = 0x32544600U;
inline constexpr uint32_t TextureFilterLinear = 0x32544601U;
// Existing v2 readers admit only 0/1 and reject mode2 before any texel read.
// Keep those encodings and the descriptor layout/error offset unchanged.
enum TextureAddressMode : uint32_t {
  TextureAddressClamp = 0, TextureAddressWrap = 1, TextureAddressMirror = 2
};
enum TextureTopology : uint32_t {
  TextureTopologyPlain = 0, TextureTopologyLayered = 1, TextureTopologyCube = 2
};

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
  uint32_t wrap_x; // Legacy spelling; now holds TextureAddressMode.
  uint32_t wrap_y;
  uint32_t linear;
  uint32_t layered; // Legacy spelling; TextureTopology, including distinct Cube=2.
  uint32_t unsigned_element; // Legacy spelling; v2 holds TextureSampleKind.
  uint32_t abi_version;
  uint64_t error;
};
static_assert(sizeof(TextureSamplingDescriptor) == 72, "texture descriptor ABI");
static_assert(alignof(TextureSamplingDescriptor) == 8, "texture descriptor alignment");
static_assert(offsetof(TextureSamplingDescriptor, width) == 24, "texture dimension ABI");
static_assert(offsetof(TextureSamplingDescriptor, unsigned_element) == 56, "texture channel ABI");
static_assert(offsetof(TextureSamplingDescriptor, linear) == 48, "texture filter ABI");
static_assert(offsetof(TextureSamplingDescriptor, abi_version) == 60, "texture version ABI");
static_assert(offsetof(TextureSamplingDescriptor, error) == 64, "texture error ABI");
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
  x = textureAddressIndex(x, static_cast<int>(descriptor->width), descriptor->wrap_x == TextureAddressWrap);
  y = textureAddressIndex(y, static_cast<int>(descriptor->height), descriptor->wrap_y == TextureAddressWrap);
  const uint64_t address = descriptor->data + uint64_t(layer) * descriptor->layer_bytes +
      uint64_t(y) * descriptor->pitch_bytes + uint64_t(x) * sizeof(float);
  const float value = *reinterpret_cast<const float *>(address);
  // CUDA hardware's non-finite texel/filter behavior is outside the admitted
  // numerical contract. Surface it through the same observable sticky error.
  return __builtin_isfinite(value) ? value : textureSamplingReject(descriptor);
}

// Initial cubemap support deliberately admits only exact texel centers on
// power-of-two square faces. A rounded projection is not a center proof.
__aicore__ ASCIFY_FORCEINLINE bool textureCubeCenterIndex(
    float axis, uint32_t extent, int &index) {
  const float pixel = (axis + 1.0f) * (0.5f * static_cast<float>(extent)) - 0.5f;
  if (!__builtin_isfinite(pixel) || pixel < 0.0f ||
      pixel >= static_cast<float>(extent) || pixel != floorf(pixel)) return false;
  index = static_cast<int>(pixel); // Finite and within [0,2^20) first.
  // The integer numerator and its division by a bounded power of two are
  // exact FP32 dyadics. Reject adjacent floats that rounded to this pixel.
  const int numerator = 2 * index + 1 - static_cast<int>(extent);
  return axis == static_cast<float>(numerator) / static_cast<float>(extent);
}

__aicore__ ASCIFY_FORCEINLINE float textureSampleCubeFloat(
    unsigned long long token, float x, float y, float z) {
  const auto *descriptor = reinterpret_cast<const TextureSamplingDescriptor *>(token);
  const uint32_t width = descriptor->width;
  // Distinct kind and topology reject ordinary 2D/layered handles and make
  // old v1/v2 readers reject new cube descriptors before a texel GM read.
  if (descriptor->abi_version != TextureDescriptorVersion ||
      descriptor->unsigned_element != TextureSampleCubeFloat ||
      descriptor->layered != TextureTopologyCube || descriptor->layers != 6 ||
      !width || width > (1U << 20) || (width & (width - 1)) ||
      descriptor->height != width || descriptor->normalized != 1 ||
      descriptor->wrap_x > TextureAddressMirror || descriptor->wrap_y > TextureAddressMirror ||
      (descriptor->linear != TextureFilterPoint && descriptor->linear != TextureFilterLinear) ||
      !__builtin_isfinite(x) || !__builtin_isfinite(y) || !__builtin_isfinite(z))
    return textureSamplingReject(descriptor);
  const float ax = fabsf(x), ay = fabsf(y), az = fabsf(z);
  float s = 0.0f, t = 0.0f;
  int face = 0;
  // CUDA 13 Programming Guide Table6. Only a strict, unit major axis is
  // admitted; no tie, zero-vector, general division or seam rule is guessed.
  if (ax > ay && ax > az && ax == 1.0f) {
    face = x > 0.0f ? 0 : 1; s = x > 0.0f ? -z : z; t = -y;
  } else if (ay > ax && ay > az && ay == 1.0f) {
    face = y > 0.0f ? 2 : 3; s = x; t = y > 0.0f ? z : -z;
  } else if (az > ax && az > ay && az == 1.0f) {
    face = z > 0.0f ? 4 : 5; s = z > 0.0f ? x : -x; t = -y;
  } else return textureSamplingReject(descriptor);
  int ix = 0, iy = 0;
  if (!textureCubeCenterIndex(s, width, ix) ||
      !textureCubeCenterIndex(t, width, iy)) return textureSamplingReject(descriptor);
  // At an exact center point/linear agree, with zero other weights. There
  // are no cross-face or edge accesses in this explicitly bounded contract.
  return textureLoadFloat(descriptor, ix, iy, face);
}

__aicore__ ASCIFY_FORCEINLINE float textureCoordinate(
    float coordinate, uint32_t extent, bool normalized, uint32_t mode) {
  if (normalized) {
    if (mode == TextureAddressWrap) coordinate -= floorf(coordinate);
    else if (mode == TextureAddressMirror) {
      // A period-two triangle is even in x. Reduce a finite nonnegative FP32
      // value without converting a huge coordinate to an integer, subtracting
      // a tiny negative fraction from one, or overflowing 2*floor(abs(x)/2).
      const float magnitude = fabsf(coordinate);
      const float period = magnitude - 2.0f * floorf(magnitude * 0.5f);
      coordinate = period > 1.0f ? 2.0f - period : period;
    }
    else coordinate = coordinate < 0.0f ? 0.0f : coordinate > 1.0f ? 1.0f : coordinate;
    return coordinate * static_cast<float>(extent);
  }
  // Unnormalized wrap/mirror become clamp when constructing the descriptor.
  const float bound = static_cast<float>(extent);
  return coordinate < 0.0f ? 0.0f : coordinate > bound ? bound : coordinate;
}

__aicore__ ASCIFY_FORCEINLINE bool textureSamplingHeader(
    const TextureSamplingDescriptor *descriptor, TextureSampleKind expected,
    float x, float y, int layer, bool layered) {
  // Validate tags before any texel GM read, including before choosing the
  // scalar/vector load width. Descriptors are allocator-owned, not caller
  // pointers; the existing opaque-token lifetime contract is unchanged.
  return descriptor->abi_version == TextureDescriptorVersion &&
      descriptor->unsigned_element == expected &&
      (descriptor->linear == TextureFilterPoint || descriptor->linear == TextureFilterLinear) &&
      descriptor->width && descriptor->width <= (1U << 20) &&
      descriptor->height && descriptor->height <= (1U << 20) &&
      descriptor->layers && descriptor->layers <= (1U << 20) &&
      descriptor->normalized <= 1 && descriptor->wrap_x <= TextureAddressMirror &&
      descriptor->wrap_y <= TextureAddressMirror &&
      (!descriptor->wrap_x || descriptor->normalized) &&
      (!descriptor->wrap_y || descriptor->normalized) && descriptor->layered <= 1 &&
      (descriptor->layered != 0) == layered && layer >= 0 &&
      static_cast<uint32_t>(layer) < descriptor->layers &&
      __builtin_isfinite(x) && __builtin_isfinite(y);
}

__aicore__ ASCIFY_FORCEINLINE bool textureSampleCoordinates(
    const TextureSamplingDescriptor *descriptor, float &x, float &y) {
  x = textureCoordinate(x, descriptor->width, descriptor->normalized != 0,
                        descriptor->wrap_x);
  y = textureCoordinate(y, descriptor->height, descriptor->normalized != 0,
                        descriptor->wrap_y);
  // Clamp/reduce first, then bound the transformed values before *any*
  // float-to-int conversion. This also covers extreme finite inputs,
  // normalized multiplication and negative zero without undefined casts.
  return __builtin_isfinite(x) && __builtin_isfinite(y) && x >= 0.0f && y >= 0.0f &&
      x <= static_cast<float>(descriptor->width) &&
      y <= static_cast<float>(descriptor->height);
}

struct TextureLinearCoordinates { int x, y; float a, b; };
template<bool Approximate>
__aicore__ ASCIFY_FORCEINLINE bool textureLinearCoordinates(
    const TextureSamplingDescriptor *descriptor, float x, float y,
    TextureLinearCoordinates &result) {
  const float xb = x - 0.5f, yb = y - 0.5f;
  const float xf = floorf(xb), yf = floorf(yb);
  if (!__builtin_isfinite(xf) || !__builtin_isfinite(yf) || xf < -1.0f || yf < -1.0f ||
      xf > static_cast<float>(descriptor->width) || yf > static_cast<float>(descriptor->height))
    return false;
  const float a = xb - xf, b = yb - yf;
  const float aq = a * 256.0f, bq = b * 256.0f;
  // Precise accepts exact 8-fraction-bit weights. Fast explicitly uses FP32
  // bilinear weights and does not emulate CUDA fixed-point quantization/ties.
  if (!Approximate && (aq != floorf(aq) || bq != floorf(bq))) return false;
  result = {static_cast<int>(xf), static_cast<int>(yf), a, b};
  return true;
}

template<bool Approximate = false>
__aicore__ ASCIFY_FORCEINLINE float textureSampleFloat(
    unsigned long long token, float x, float y, int layer, bool layered) {
  const auto *descriptor = reinterpret_cast<const TextureSamplingDescriptor *>(token);
  if (!textureSamplingHeader(descriptor, TextureSampleFloat, x, y, layer, layered) ||
      !textureSampleCoordinates(descriptor, x, y))
    return textureSamplingReject(descriptor);
  if (descriptor->linear == TextureFilterPoint) {
    return textureLoadFloat(descriptor, static_cast<int>(floorf(x)),
                            static_cast<int>(floorf(y)), layer);
  }
  TextureLinearCoordinates coordinates;
  if (!textureLinearCoordinates<Approximate>(descriptor, x, y, coordinates))
    return textureSamplingReject(descriptor);
  const int ix = coordinates.x, iy = coordinates.y;
  const float a = coordinates.a, b = coordinates.b;
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

// Element-type integer reads preserve all 32 payload bits. In particular,
// neither the load nor the return goes through float, even under fast policy.
__aicore__ ASCIFY_FORCEINLINE unsigned int textureSampleUnsigned(
    unsigned long long token, float x, float y, int layer, bool layered) {
  static_assert(sizeof(unsigned int) == 4, "texture unsigned int fetch width");
  const auto *descriptor = reinterpret_cast<const TextureSamplingDescriptor *>(token);
  if (!textureSamplingHeader(descriptor, TextureSampleUnsigned, x, y, layer, layered) ||
      descriptor->linear != TextureFilterPoint || !textureSampleCoordinates(descriptor, x, y)) {
    (void)textureSamplingReject(descriptor);
    return 0;
  }
  const int ix = textureAddressIndex(static_cast<int>(floorf(x)),
      static_cast<int>(descriptor->width), descriptor->wrap_x == TextureAddressWrap);
  const int iy = textureAddressIndex(static_cast<int>(floorf(y)),
      static_cast<int>(descriptor->height), descriptor->wrap_y == TextureAddressWrap);
  const uint64_t address = descriptor->data + uint64_t(layer) * descriptor->layer_bytes +
      uint64_t(iy) * descriptor->pitch_bytes + uint64_t(ix) * sizeof(unsigned int);
  return *reinterpret_cast<const unsigned int *>(address);
}

#if defined(ASCIFY_SIMT_LEGACY_HAS_VECTOR_CONSTRUCTORS)
__aicore__ ASCIFY_FORCEINLINE bool textureRGBAWrapCoordinateAdmitted(
    const TextureSamplingDescriptor *descriptor, float x, float y) {
  // For a tiny negative FP32 input the exact wrapped fraction is below one,
  // but the existing FP32 subtraction can round to one. A point read would
  // then wrap to texel zero instead of the mathematical final texel. Reject
  // this unresolved boundary for the new RGBA domain; retain legacy scalar
  // coordinate behavior. Caller has already proved both inputs finite.
  return !descriptor->normalized ||
      ((descriptor->wrap_x != TextureAddressWrap || x >= 0.0f || x - floorf(x) != 1.0f) &&
       (descriptor->wrap_y != TextureAddressWrap || y >= 0.0f || y - floorf(y) != 1.0f));
}
__aicore__ ASCIFY_FORCEINLINE uchar4 textureLoadRGBA(
    const TextureSamplingDescriptor *descriptor, int x, int y, int layer) {
  x = textureAddressIndex(x, static_cast<int>(descriptor->width), descriptor->wrap_x == TextureAddressWrap);
  y = textureAddressIndex(y, static_cast<int>(descriptor->height), descriptor->wrap_y == TextureAddressWrap);
  const uint64_t address = descriptor->data + uint64_t(layer) * descriptor->layer_bytes +
      uint64_t(y) * descriptor->pitch_bytes + uint64_t(x) * 4;
  const auto *bytes = reinterpret_cast<const unsigned char *>(address);
  uchar4 value;
  value.x = bytes[0]; value.y = bytes[1]; value.z = bytes[2]; value.w = bytes[3];
  return value;
}
__aicore__ ASCIFY_FORCEINLINE float4 textureNormalizeRGBA(uchar4 value) {
  float4 result;
  result.x = static_cast<float>(value.x) / 255.0f;
  result.y = static_cast<float>(value.y) / 255.0f;
  result.z = static_cast<float>(value.z) / 255.0f;
  result.w = static_cast<float>(value.w) / 255.0f;
  return result;
}
__aicore__ ASCIFY_FORCEINLINE float4 textureBlendRGBA(float4 left, float4 right, float weight) {
  float4 result;
  result.x = (1.0f - weight) * left.x + weight * right.x;
  result.y = (1.0f - weight) * left.y + weight * right.y;
  result.z = (1.0f - weight) * left.z + weight * right.z;
  result.w = (1.0f - weight) * left.w + weight * right.w;
  return result;
}
__aicore__ ASCIFY_FORCEINLINE float4 textureRejectRGBA(
    const TextureSamplingDescriptor *descriptor) {
  const float error = textureSamplingReject(descriptor);
  float4 result;
  result.x = result.y = result.z = result.w = error;
  return result;
}
__aicore__ ASCIFY_FORCEINLINE uchar4 textureSampleRGBAElement(
    unsigned long long token, float x, float y, int layer, bool layered) {
  const auto *descriptor = reinterpret_cast<const TextureSamplingDescriptor *>(token);
  if (!textureSamplingHeader(descriptor, TextureSampleRGBAElement, x, y, layer, layered) ||
      descriptor->linear != TextureFilterPoint || !textureRGBAWrapCoordinateAdmitted(descriptor,x,y) ||
      !textureSampleCoordinates(descriptor, x, y)) {
    (void)textureSamplingReject(descriptor);
    return uchar4{};
  }
  return textureLoadRGBA(descriptor, static_cast<int>(floorf(x)),
                         static_cast<int>(floorf(y)), layer);
}
template<bool Approximate = false>
__aicore__ ASCIFY_FORCEINLINE float4 textureSampleRGBANormalized(
    unsigned long long token, float x, float y, int layer, bool layered) {
  const auto *descriptor = reinterpret_cast<const TextureSamplingDescriptor *>(token);
  if (!textureSamplingHeader(descriptor, TextureSampleRGBANormalized, x, y, layer, layered) ||
      !textureRGBAWrapCoordinateAdmitted(descriptor,x,y) ||
      !textureSampleCoordinates(descriptor, x, y)) return textureRejectRGBA(descriptor);
  if (descriptor->linear == TextureFilterPoint)
    return textureNormalizeRGBA(textureLoadRGBA(descriptor, static_cast<int>(floorf(x)),
                                               static_cast<int>(floorf(y)), layer));
  TextureLinearCoordinates coordinates;
  if (!textureLinearCoordinates<Approximate>(descriptor, x, y, coordinates))
    return textureRejectRGBA(descriptor);
  const int ix = coordinates.x, iy = coordinates.y;
  const float a = coordinates.a, b = coordinates.b;
  const float4 t00 = textureNormalizeRGBA(textureLoadRGBA(descriptor, ix, iy, layer));
  const float4 row0 = a == 0.0f ? t00 : textureBlendRGBA(t00,
      textureNormalizeRGBA(textureLoadRGBA(descriptor, ix + 1, iy, layer)), a);
  if (b == 0.0f) return row0;
  const float4 t01 = textureNormalizeRGBA(textureLoadRGBA(descriptor, ix, iy + 1, layer));
  const float4 row1 = a == 0.0f ? t01 : textureBlendRGBA(t01,
      textureNormalizeRGBA(textureLoadRGBA(descriptor, ix + 1, iy + 1, layer)), a);
  return textureBlendRGBA(row0, row1, b);
}
#endif // Native vector type domain.
#endif

} // namespace detail
} // namespace ascify
#endif
