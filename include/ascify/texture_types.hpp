#ifndef ASCIFY_TEXTURE_TYPES_HPP
#define ASCIFY_TEXTURE_TYPES_HPP
#include <ascify/ascify_cuda_compat.hpp>

namespace ascify {

namespace detail {
template<typename T> struct textureRGBAElementType : std::false_type {};
template<typename T> struct textureRGBAFloatType : std::false_type {};
// CCEC supplies these native vector types in both halves of pure-SIMT
// compilation. Merely finding vector_functions.h in a host include directory
// does not prove that they exist. Plain C++ and the public-8.5 family keep the
// scalar-only surface, without synthetic vector declarations.
#if defined(ASCIFY_SIMT_LEGACY_HAS_VECTOR_CONSTRUCTORS) && \
    defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)
static_assert(sizeof(uchar4) == 4 && alignof(uchar4) == 4,
              "RGBA8 texture requires native four-byte uchar4 ABI");
static_assert(sizeof(float4) == 16 && alignof(float4) == 16,
              "RGBA8 normalized texture requires native float4 ABI");
template<> struct textureRGBAElementType<uchar4> : std::true_type {};
template<> struct textureRGBAFloatType<float4> : std::true_type {};
#endif
template<typename T> struct textureFetchType : std::integral_constant<bool,
    std::is_same<T,float>::value || std::is_same<T,unsigned int>::value ||
    textureRGBAElementType<T>::value || textureRGBAFloatType<T>::value> {};
} // namespace detail

// Explicit, bounded adapter types. The converter must prove SDK ownership
// before publishing these names. No arbitrary SDK record-layout reflection
// is admitted by declaring these structurally similar types.
enum cudaChannelFormatKind {
  cudaChannelFormatKindSigned = 0,
  cudaChannelFormatKindUnsigned = 1,
  cudaChannelFormatKindFloat = 2,
  cudaChannelFormatKindNone = 3
};
struct cudaChannelFormatDesc {
  int x, y, z, w;
  cudaChannelFormatKind f;
};
struct cudaArray;
using cudaArray_t = cudaArray *;
using cudaArray_const_t = const cudaArray *;
using cudaTextureObject_t = unsigned long long;
inline constexpr unsigned cudaArrayLayered = 1U;

enum cudaResourceType {
  cudaResourceTypeArray = 0,
  cudaResourceTypeMipmappedArray = 1,
  cudaResourceTypeLinear = 2,
  cudaResourceTypePitch2D = 3
};
struct cudaResourceDesc {
  cudaResourceType resType;
  union {
    struct { cudaArray_t array; } array;
    struct { void *mipmap; } mipmap;
    struct { void *devPtr; cudaChannelFormatDesc desc; size_t sizeInBytes; } linear;
    struct {
      void *devPtr;
      cudaChannelFormatDesc desc;
      size_t width, height, pitchInBytes;
    } pitch2D;
    struct { int reserved[32]; } reserved;
  } res;
  unsigned flags;
};
struct cudaResourceViewDesc;
enum cudaTextureAddressMode {
  cudaAddressModeWrap = 0, cudaAddressModeClamp = 1,
  cudaAddressModeMirror = 2, cudaAddressModeBorder = 3
};
enum cudaTextureFilterMode { cudaFilterModePoint = 0, cudaFilterModeLinear = 1 };
enum cudaTextureReadMode { cudaReadModeElementType = 0, cudaReadModeNormalizedFloat = 1 };
struct cudaTextureDesc {
  cudaTextureAddressMode addressMode[3];
  cudaTextureFilterMode filterMode;
  cudaTextureReadMode readMode;
  int sRGB;
  float borderColor[4];
  int normalizedCoords;
  unsigned maxAnisotropy;
  cudaTextureFilterMode mipmapFilterMode;
  float mipmapLevelBias, minMipmapLevelClamp, maxMipmapLevelClamp;
  int disableTrilinearOptimization, seamlessCubemap;
};
struct cudaPitchedPtr { void *ptr; size_t pitch, xsize, ysize; };
struct cudaExtent { size_t width, height, depth; };
struct cudaPos { size_t x, y, z; };
struct cudaMemcpy3DParms {
  cudaArray_t srcArray;
  cudaPos srcPos;
  cudaPitchedPtr srcPtr;
  cudaArray_t dstArray;
  cudaPos dstPos;
  cudaPitchedPtr dstPtr;
  cudaExtent extent;
  aclrtMemcpyKind kind;
};

inline cudaExtent make_cudaExtent(size_t width, size_t height, size_t depth) {
  return {width, height, depth};
}
inline cudaPos make_cudaPos(size_t x, size_t y, size_t z) { return {x, y, z}; }
inline cudaPitchedPtr make_cudaPitchedPtr(void *pointer, size_t pitch,
                                        size_t width, size_t height) {
  return {pointer, pitch, width, height};
}
inline cudaChannelFormatDesc cudaCreateChannelDesc(
    int x, int y, int z, int w, cudaChannelFormatKind kind) {
  return {x, y, z, w, kind};
}
template<typename T>
inline cudaChannelFormatDesc cudaCreateChannelDesc() {
  static_assert(std::is_same<T, float>::value ||
                std::is_same<T, unsigned int>::value ||
                detail::textureRGBAElementType<T>::value,
                "Ascify texture channels require float32, unsigned int32 or native uchar4");
  static_assert(sizeof(unsigned int) == 4, "texture unsigned int channel width");
  if constexpr (detail::textureRGBAElementType<T>::value)
    return {8, 8, 8, 8, cudaChannelFormatKindUnsigned};
  return {32, 0, 0, 0, std::is_same<T, unsigned int>::value
      ? cudaChannelFormatKindUnsigned : cudaChannelFormatKindFloat};
}

} // namespace ascify
#endif
