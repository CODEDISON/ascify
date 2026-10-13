#ifndef ASCIFY_TEXTURE_TYPES_HPP
#define ASCIFY_TEXTURE_TYPES_HPP
#include <ascify/ascify_cuda_compat.hpp>

namespace ascify {

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
  static_assert(std::is_same<T, float>::value,
                "Ascify texture channels currently admit only float32");
  return {32, 0, 0, 0, cudaChannelFormatKindFloat};
}

} // namespace ascify
#endif
