#ifndef ASCIFY_TEXTURE_COMPAT_HPP
#define ASCIFY_TEXTURE_COMPAT_HPP
#include <ascify/texture_types.hpp>
#include <ascify/texture_sampling.hpp>
#include <cstring>

namespace ascify {
namespace detail {
enum TextureChannelKind { TextureChannelInvalid, TextureChannelFloat,
                          TextureChannelUnsigned, TextureChannelRGBA };

// Host bookkeeping uses fixed slots and owns no process-exit ACL callback.
// The runtime manager resets owned devices before finalizing ACL. Explicit
// successful resets discard the corresponding bookkeeping through the hook.
struct TextureAllocation {
  void *storage = nullptr;
  size_t bytes = 0, pitch = 0, width = 0, height = 0, layers = 0;
  int32_t device = -1;
  void *context = nullptr;
  unsigned references = 0;
  bool array = false, layered = false;
  TextureChannelKind channel_kind = TextureChannelInvalid;
};
struct TextureHandle {
  void *descriptor = nullptr;
  TextureAllocation *allocation = nullptr;
};
struct TextureRegistry {
  std::atomic_flag lock = ATOMIC_FLAG_INIT;
  TextureAllocation allocations[128] = {};
  TextureHandle handles[128] = {};
};
inline TextureRegistry &textureRegistry() { static TextureRegistry registry; return registry; }
struct TextureLock {
  TextureLock() { while (textureRegistry().lock.test_and_set(std::memory_order_acquire)) {} }
  ~TextureLock() { textureRegistry().lock.clear(std::memory_order_release); }
};
inline bool textureMultiply(size_t a, size_t b, size_t &result) {
  if (b && a > SIZE_MAX / b) return false;
  result = a * b; return true;
}
inline bool textureScalar32Channel(const cudaChannelFormatDesc &channel) {
  return channel.x == 32 && channel.y == 0 && channel.z == 0 && channel.w == 0 &&
         (channel.f == cudaChannelFormatKindFloat || channel.f == cudaChannelFormatKindUnsigned);
}
inline TextureChannelKind textureChannelKind(const cudaChannelFormatDesc &channel) {
  if (textureScalar32Channel(channel)) return channel.f == cudaChannelFormatKindFloat
      ? TextureChannelFloat : TextureChannelUnsigned;
  if (channel.x == 8 && channel.y == 8 && channel.z == 8 && channel.w == 8 &&
      channel.f == cudaChannelFormatKindUnsigned) return TextureChannelRGBA;
  return TextureChannelInvalid;
}
inline aclError textureCurrent(int32_t &device, void *&context) {
  aclError status = cudaRuntimeEnsureReady();
  if (status != ACL_SUCCESS) return status;
  status = ::aclrtGetDevice(&device);
  if (status != ACL_SUCCESS) return status;
  status = ::aclrtGetCurrentContext(&context);
  if (status != ACL_SUCCESS) return status;
  return context == nullptr ? ACL_ERROR_RT_PARAM_INVALID : ACL_SUCCESS;
}
inline TextureAllocation *textureArray(cudaArray_const_t array) {
  for (auto &record : textureRegistry().allocations)
    if (record.storage && record.array && reinterpret_cast<cudaArray_const_t>(&record) == array)
      return &record;
  return nullptr;
}
inline TextureAllocation *texturePointer(const void *pointer) {
  if (!pointer) return nullptr;
  for (auto &record : textureRegistry().allocations)
    if (record.storage == pointer && !record.array) return &record;
  return nullptr;
}
inline bool textureOwned(const TextureAllocation &record, int32_t device, void *context) {
  return record.device == device && record.context == context;
}

// Caller holds TextureLock and already proved the current device/context.
// Synchronize before reading the sticky error words; these are real GM copies.
inline aclError textureStatusLocked(int32_t device, void *context) {
  bool any = false;
  for (const auto &handle : textureRegistry().handles)
    if (handle.descriptor && textureOwned(*handle.allocation, device, context)) any = true;
  if (!any) return ACL_SUCCESS;
  aclError status = ::aclrtSynchronizeDevice();
  if (status != ACL_SUCCESS) return status;
  for (const auto &handle : textureRegistry().handles) {
    if (!handle.descriptor || !textureOwned(*handle.allocation, device, context)) continue;
    uint32_t flag = 0;
    status = ::aclrtMemcpy(&flag, sizeof(flag),
        static_cast<unsigned char *>(handle.descriptor) + sizeof(TextureSamplingDescriptor),
        sizeof(flag), ACL_MEMCPY_DEVICE_TO_HOST);
    if (status != ACL_SUCCESS) return status;
    if (flag != 0) return ACL_ERROR_FEATURE_UNSUPPORTED;
  }
  return ACL_SUCCESS;
}
inline aclError textureCheckPending() {
  TextureLock guard;
  bool any = false;
  for (const auto &handle : textureRegistry().handles) if (handle.descriptor) any = true;
  if (!any) return ACL_SUCCESS;
  int32_t device = -1; void *context = nullptr;
  aclError status = textureCurrent(device, context);
  return status == ACL_SUCCESS ? textureStatusLocked(device, context) : status;
}
// Freeing backing storage does not implicitly destroy its texture objects.
// Invalidate their device descriptors before the real free, keeping each error
// word alive until its object is destroyed. The retired host slot remains pinned
// by references so a later allocation cannot silently retarget an old object.
// Callers hold TextureLock and have proved the current device/context.
inline aclError textureReleaseStorage(TextureAllocation &record) {
  aclError status = ::aclrtSynchronizeDevice();
  if (status != ACL_SUCCESS) return status;
  const uint32_t invalidWidth = 0;
  for (const auto &handle : textureRegistry().handles) {
    if (!handle.descriptor || handle.allocation != &record) continue;
    status = ::aclrtMemcpy(static_cast<unsigned char *>(handle.descriptor) +
        offsetof(TextureSamplingDescriptor, width), sizeof(invalidWidth),
        &invalidWidth, sizeof(invalidWidth), ACL_MEMCPY_HOST_TO_DEVICE);
    if (status != ACL_SUCCESS) return status;
  }
  status = ::aclrtFree(record.storage);
  if (status == ACL_SUCCESS) {
    record.storage = nullptr;
    if (!record.references) record = {};
  }
  return status;
}
inline aclError textureFreeAllocation(void *pointer, bool *handled) {
  TextureLock guard;
  auto *record = texturePointer(pointer);
  *handled = record != nullptr;
  if (!record) return ACL_SUCCESS;
  int32_t device = -1; void *context = nullptr;
  aclError status = textureCurrent(device, context);
  if (status != ACL_SUCCESS) return status;
  if (!textureOwned(*record,device,context)) return ACL_ERROR_RT_PARAM_INVALID;
  return textureReleaseStorage(*record);
}
inline void textureAfterReset(int32_t device) {
  TextureLock guard;
  for (auto &handle : textureRegistry().handles)
    if (handle.descriptor && handle.allocation->device == device) handle = {};
  for (auto &record : textureRegistry().allocations)
    if (record.device == device) record = {};
}

inline aclError textureAllocate(TextureAllocation **result, size_t width,
    size_t height, size_t layers, bool array, bool layered,
    TextureChannelKind channel_kind = TextureChannelInvalid) {
  // Bounds also keep all sampler integer conversions and float extents exact.
  if (!width || !height || !layers || width > (1U<<20) || height > (1U<<20) ||
      layers > (1U<<20)) return ACL_ERROR_FEATURE_UNSUPPORTED;
  size_t pitch = width, plane = 0, bytes = 0;
  if (array && !textureMultiply(width,sizeof(float),pitch)) return ACL_ERROR_RT_PARAM_INVALID;
  if (!array) {
    if (pitch > SIZE_MAX - 31) return ACL_ERROR_RT_PARAM_INVALID;
    pitch = (pitch + 31) & ~size_t(31);
  }
  if (!textureMultiply(pitch,height,plane) || !textureMultiply(plane,layers,bytes))
    return ACL_ERROR_RT_PARAM_INVALID;
  int32_t device = -1; void *context = nullptr;
  aclError status = textureCurrent(device,context);
  if (status != ACL_SUCCESS) return status;
  static const RuntimeTextureHooks hooks={textureCheckPending,textureFreeAllocation,
                                          textureAfterReset};
  runtimeTextureHooks.store(&hooks,std::memory_order_release);
  TextureLock guard;
  TextureAllocation *slot = nullptr;
  for (auto &record : textureRegistry().allocations)
    if (!record.storage && !record.references) { slot=&record; break; }
  if (!slot) return ACL_ERROR_BAD_ALLOC;
  void *storage = nullptr;
  status = ::aclrtMalloc(&storage,bytes,ACL_MEM_MALLOC_HUGE_FIRST);
  if (status != ACL_SUCCESS) return status;
  if (!storage) return ACL_ERROR_RT_MEMORY_ALLOCATION;
  *slot = {storage,bytes,pitch,width,height,layers,device,context,0,array,layered,channel_kind};
  *result = slot;
  return ACL_SUCCESS;
}

inline bool textureRowsRange(size_t pitch, size_t width, size_t height, size_t &bytes) {
  size_t last = 0;
  if (width > pitch || !textureMultiply(height ? height-1 : 0,pitch,last) ||
      width > SIZE_MAX-last) return false;
  bytes = height ? last+width : 0;
  return true;
}
inline aclError textureCopyRows(void *dst, size_t dpitch, const void *src,
    size_t spitch, size_t width, size_t height, aclrtMemcpyKind kind) {
  size_t db = 0, sb = 0;
  if (!isSupportedMemcpyKind(kind) || !textureRowsRange(dpitch,width,height,db) ||
      !textureRowsRange(spitch,width,height,sb) ||
      !nonnullMemoryRange(dst,db) || !nonnullMemoryRange(src,sb)) return ACL_ERROR_RT_PARAM_INVALID;
  if (!width || !height) return ACL_SUCCESS;
  if (reinterpret_cast<uintptr_t>(dst) > UINTPTR_MAX-db ||
      reinterpret_cast<uintptr_t>(src) > UINTPTR_MAX-sb) return ACL_ERROR_RT_PARAM_INVALID;
  // Blocking immediate ACL copies do not themselves order earlier default
  // stream kernels. This is needed even when no texture object exists yet.
  if (kind != ACL_MEMCPY_HOST_TO_HOST) {
    const aclError status = ::aclrtSynchronizeStream(nullptr);
    if (status != ACL_SUCCESS) return status;
  }
  for (size_t row=0;row<height;++row) {
    aclError status = ::aclrtMemcpy(static_cast<unsigned char *>(dst)+row*dpitch,
        db-row*dpitch,static_cast<const unsigned char *>(src)+row*spitch,width,kind);
    if (status != ACL_SUCCESS) return status;
  }
  return ACL_SUCCESS;
}
} // namespace detail

inline aclError cudaMallocPitch(void **pointer, size_t *pitch, size_t width, size_t height) {
  if (!pointer || !pitch) return ACL_ERROR_RT_PARAM_INVALID;
  detail::TextureAllocation *record = nullptr;
  aclError status = detail::textureAllocate(&record,width,height,1,false,false);
  if (status == ACL_SUCCESS) { *pointer=record->storage; *pitch=record->pitch; }
  return status;
}
template<typename T>
inline aclError cudaMallocPitch(T **pointer, size_t *pitch, size_t width, size_t height) {
  if (!pointer) return ACL_ERROR_RT_PARAM_INVALID;
  void *storage=nullptr;
  aclError status=cudaMallocPitch(&storage,pitch,width,height);
  if(status==ACL_SUCCESS) *pointer=static_cast<T*>(storage);
  return status;
}
inline aclError cudaMallocArray(cudaArray_t *array, const cudaChannelFormatDesc *channel,
    size_t width, size_t height=0, unsigned flags=0) {
  if (!array || !channel) return ACL_ERROR_RT_PARAM_INVALID;
  const auto channel_kind = detail::textureChannelKind(*channel);
  if (flags || channel_kind == detail::TextureChannelInvalid) return ACL_ERROR_FEATURE_UNSUPPORTED;
  detail::TextureAllocation *record=nullptr;
  aclError status=detail::textureAllocate(&record,width,height,1,true,false,
      channel_kind);
  if(status==ACL_SUCCESS) *array=reinterpret_cast<cudaArray_t>(record);
  return status;
}
inline aclError cudaMalloc3DArray(cudaArray_t *array, const cudaChannelFormatDesc *channel,
    cudaExtent extent, unsigned flags=0) {
  if (!array || !channel) return ACL_ERROR_RT_PARAM_INVALID;
  const auto channel_kind = detail::textureChannelKind(*channel);
  if (flags != cudaArrayLayered || channel_kind == detail::TextureChannelInvalid)
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  detail::TextureAllocation *record=nullptr;
  aclError status=detail::textureAllocate(&record,extent.width,extent.height,extent.depth,true,true,
      channel_kind);
  if(status==ACL_SUCCESS) *array=reinterpret_cast<cudaArray_t>(record);
  return status;
}
inline aclError cudaFreeArray(cudaArray_t array) {
  if (!array) return ACL_SUCCESS;
  int32_t device=-1; void *context=nullptr;
  aclError status=detail::textureCurrent(device,context);
  if(status!=ACL_SUCCESS) return status;
  detail::TextureLock guard;
  auto *record=detail::textureArray(array);
  if(!record || !detail::textureOwned(*record,device,context))
    return ACL_ERROR_RT_PARAM_INVALID;
  return detail::textureReleaseStorage(*record);
}
inline aclError cudaMemcpy2D(void *dst,size_t dpitch,const void *src,size_t spitch,
    size_t width,size_t height,aclrtMemcpyKind kind) {
  aclError status=cudaRuntimeEnsureReady();
  if(status!=ACL_SUCCESS) return status;
  status=detail::textureCheckPending();
  if(status!=ACL_SUCCESS) return status;
  return detail::textureCopyRows(dst,dpitch,src,spitch,width,height,kind);
}
inline aclError cudaMemset2D(void *dst,size_t pitch,int value,size_t width,size_t height) {
  size_t bytes=0;
  if(!detail::textureRowsRange(pitch,width,height,bytes) || !detail::nonnullMemoryRange(dst,bytes))
    return ACL_ERROR_RT_PARAM_INVALID;
  if(!bytes) return ACL_SUCCESS;
  if(reinterpret_cast<uintptr_t>(dst)>UINTPTR_MAX-bytes) return ACL_ERROR_RT_PARAM_INVALID;
  aclError status=cudaRuntimeEnsureReady();
  if(status!=ACL_SUCCESS) return status;
  status=detail::textureCheckPending();
  if(status!=ACL_SUCCESS) return status;
  status=::aclrtSynchronizeStream(nullptr);
  if(status!=ACL_SUCCESS) return status;
  for(size_t row=0;row<height;++row) {
    status=::aclrtMemset(static_cast<unsigned char*>(dst)+row*pitch,bytes-row*pitch,value,width);
    if(status!=ACL_SUCCESS) return status;
  }
  return ACL_SUCCESS;
}
inline aclError cudaMemcpyToArray(cudaArray_t array,size_t x,size_t y,const void *src,
    size_t bytes,aclrtMemcpyKind kind) {
  if(kind!=ACL_MEMCPY_HOST_TO_DEVICE && kind!=ACL_MEMCPY_DEVICE_TO_DEVICE)
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  if(!detail::nonnullMemoryRange(src,bytes)) return ACL_ERROR_RT_PARAM_INVALID;
  int32_t device=-1; void *context=nullptr;
  aclError status=detail::textureCurrent(device,context);
  if(status!=ACL_SUCCESS) return status;
  detail::TextureLock guard;
  auto *record=detail::textureArray(array);
  if(!record || record->layered || !detail::textureOwned(*record,device,context) ||
      y>=record->height || x>record->pitch) return ACL_ERROR_RT_PARAM_INVALID;
  const size_t offset=y*record->pitch+x;
  if(bytes>record->bytes-offset) return ACL_ERROR_RT_PARAM_INVALID;
  status=detail::textureStatusLocked(device,context);
  if(status!=ACL_SUCCESS || !bytes) return status;
  status=::aclrtSynchronizeStream(nullptr);
  if(status!=ACL_SUCCESS) return status;
  return ::aclrtMemcpy(static_cast<unsigned char*>(record->storage)+offset,
      record->bytes-offset,src,bytes,kind);
}

inline aclError cudaMemcpy3D(const cudaMemcpy3DParms *parameters) {
  if(!parameters) return ACL_ERROR_RT_PARAM_INVALID;
  const auto &p=*parameters;
  if(!detail::isSupportedMemcpyKind(p.kind)) return ACL_ERROR_FEATURE_UNSUPPORTED;
  if((p.srcArray!=nullptr)==(p.srcPtr.ptr!=nullptr) ||
      (p.dstArray!=nullptr)==(p.dstPtr.ptr!=nullptr)) return ACL_ERROR_RT_PARAM_INVALID;
  // This bounded adapter requires an explicit direction consistent with the
  // array endpoints. No UVA inference or silent direction correction is used.
  if((p.srcArray && p.kind!=ACL_MEMCPY_DEVICE_TO_HOST && p.kind!=ACL_MEMCPY_DEVICE_TO_DEVICE) ||
      (p.dstArray && p.kind!=ACL_MEMCPY_HOST_TO_DEVICE && p.kind!=ACL_MEMCPY_DEVICE_TO_DEVICE))
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  size_t width=p.extent.width;
  if((p.srcArray || p.dstArray) && !detail::textureMultiply(width,sizeof(float),width))
    return ACL_ERROR_RT_PARAM_INVALID;
  if(!width || !p.extent.height || !p.extent.depth) return ACL_ERROR_RT_PARAM_INVALID;
  int32_t device=-1;void *context=nullptr;
  aclError status=detail::textureCurrent(device,context);
  if(status!=ACL_SUCCESS) return status;
  detail::TextureLock guard;
  struct Endpoint { unsigned char *base; size_t pitch,plane,offset; };
  auto endpoint=[&](cudaArray_t array,const cudaPitchedPtr &pointer,
                    cudaPos position,Endpoint &out)->bool {
    size_t pitch=pointer.pitch,height=pointer.ysize,x=position.x;
    unsigned char *base=static_cast<unsigned char*>(pointer.ptr);
    if(array) {
      const auto *record=detail::textureArray(array);
      if(!record || !detail::textureOwned(*record,device,context) ||
          position.z>record->layers || p.extent.depth>record->layers-position.z ||
          !detail::textureMultiply(x,sizeof(float),x)) return false;
      pitch=record->pitch;height=record->height;base=static_cast<unsigned char*>(record->storage);
    }
    if(!base || x>pitch || width>pitch-x || position.y>height ||
        p.extent.height>height-position.y) return false;
    size_t plane=0,zbytes=0,ybytes=0,last_plane=0,last_row=0;
    if(!detail::textureMultiply(pitch,height,plane) ||
        !detail::textureMultiply(position.z,plane,zbytes) ||
        !detail::textureMultiply(position.y,pitch,ybytes) ||
        ybytes>SIZE_MAX-zbytes || x>SIZE_MAX-zbytes-ybytes ||
        !detail::textureMultiply(p.extent.depth-1,plane,last_plane) ||
        !detail::textureMultiply(p.extent.height-1,pitch,last_row) ||
        last_row>SIZE_MAX-last_plane || width>SIZE_MAX-last_plane-last_row) return false;
    const size_t offset=zbytes+ybytes+x, span=last_plane+last_row+width;
    if(span>SIZE_MAX-offset || reinterpret_cast<uintptr_t>(base)>UINTPTR_MAX-offset-span)
      return false;
    out={base,pitch,plane,offset};return true;
  };
  Endpoint source={},destination={};
  if(!endpoint(p.srcArray,p.srcPtr,p.srcPos,source) ||
      !endpoint(p.dstArray,p.dstPtr,p.dstPos,destination)) return ACL_ERROR_RT_PARAM_INVALID;
  status=detail::textureStatusLocked(device,context);
  if(status!=ACL_SUCCESS) return status;
  if(p.kind!=ACL_MEMCPY_HOST_TO_HOST) {
    status=::aclrtSynchronizeStream(nullptr);
    if(status!=ACL_SUCCESS) return status;
  }
  for(size_t layer=0;layer<p.extent.depth;++layer)
    for(size_t row=0;row<p.extent.height;++row) {
      status=::aclrtMemcpy(destination.base+destination.offset+layer*destination.plane+row*destination.pitch,
          width,source.base+source.offset+layer*source.plane+row*source.pitch,width,p.kind);
      if(status!=ACL_SUCCESS) return status;
    }
  return ACL_SUCCESS;
}

inline aclError cudaCreateTextureObject(cudaTextureObject_t *output,
    const cudaResourceDesc *resource,const cudaTextureDesc *texture,
    const cudaResourceViewDesc *view) {
  if(!output || !resource || !texture) return ACL_ERROR_RT_PARAM_INVALID;
  if(view || resource->flags || (texture->readMode!=cudaReadModeElementType &&
      texture->readMode!=cudaReadModeNormalizedFloat) ||
      texture->sRGB || (texture->normalizedCoords!=0 && texture->normalizedCoords!=1) ||
      texture->maxAnisotropy>1 || texture->mipmapFilterMode!=cudaFilterModePoint ||
      texture->mipmapLevelBias!=0 || texture->minMipmapLevelClamp!=0 ||
      texture->maxMipmapLevelClamp!=0 || texture->disableTrilinearOptimization ||
      texture->seamlessCubemap || (texture->filterMode!=cudaFilterModePoint &&
                                 texture->filterMode!=cudaFilterModeLinear))
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  for(unsigned i=0;i<4;++i) if(texture->borderColor[i]!=0) return ACL_ERROR_FEATURE_UNSUPPORTED;
  for(unsigned i=0;i<3;++i) if(texture->addressMode[i]!=cudaAddressModeWrap &&
      texture->addressMode[i]!=cudaAddressModeClamp) return ACL_ERROR_FEATURE_UNSUPPORTED;
  int32_t device=-1; void *context=nullptr;
  aclError status=detail::textureCurrent(device,context);
  if(status!=ACL_SUCCESS) return status;
  detail::TextureLock guard;
  detail::TextureAllocation *record=nullptr;
  size_t width=0,height=0,pitch=0;
  detail::TextureChannelKind channel_kind=detail::TextureChannelInvalid;
  if(resource->resType==cudaResourceTypeArray) {
    record=detail::textureArray(resource->res.array.array);
    if(record) {
      width=record->width; height=record->height; pitch=record->pitch;
      channel_kind=record->channel_kind;
    }
  } else if(resource->resType==cudaResourceTypePitch2D) {
    const auto &input=resource->res.pitch2D;
    channel_kind=detail::textureChannelKind(input.desc);
    if(channel_kind==detail::TextureChannelInvalid) return ACL_ERROR_FEATURE_UNSUPPORTED;
    record=detail::texturePointer(input.devPtr);
    width=input.width; height=input.height; pitch=input.pitchInBytes;
    size_t row_bytes=0,bytes=0;
    if(!width || !height || width>(1U<<20) || height>(1U<<20) ||
        !detail::textureMultiply(width,sizeof(float),row_bytes) ||
        !detail::textureRowsRange(pitch,row_bytes,height,bytes) ||
        !record || pitch!=record->pitch || bytes>record->bytes) return ACL_ERROR_RT_PARAM_INVALID;
  } else return ACL_ERROR_FEATURE_UNSUPPORTED;
  if(!record || !detail::textureOwned(*record,device,context)) return ACL_ERROR_RT_PARAM_INVALID;
  detail::TextureSampleKind sample_kind=detail::TextureSampleFloat;
  if(channel_kind==detail::TextureChannelRGBA) {
    sample_kind=texture->readMode==cudaReadModeNormalizedFloat
        ? detail::TextureSampleRGBANormalized : detail::TextureSampleRGBAElement;
    if(sample_kind==detail::TextureSampleRGBAElement && texture->filterMode!=cudaFilterModePoint)
      return ACL_ERROR_FEATURE_UNSUPPORTED;
  } else {
    if(texture->readMode!=cudaReadModeElementType || channel_kind==detail::TextureChannelInvalid)
      return ACL_ERROR_FEATURE_UNSUPPORTED;
    if(channel_kind==detail::TextureChannelUnsigned) {
      if(texture->filterMode!=cudaFilterModePoint) return ACL_ERROR_FEATURE_UNSUPPORTED;
      sample_kind=detail::TextureSampleUnsigned;
    }
  }
  detail::TextureHandle *slot=nullptr;
  for(auto &handle:detail::textureRegistry().handles) if(!handle.descriptor) {slot=&handle;break;}
  if(!slot) return ACL_ERROR_BAD_ALLOC;
  void *descriptor=nullptr;
  status=::aclrtMalloc(&descriptor,96,ACL_MEM_MALLOC_HUGE_FIRST);
  if(status!=ACL_SUCCESS) return status;
  if(!descriptor) return ACL_ERROR_RT_MEMORY_ALLOCATION;
  detail::TextureSamplingDescriptor value={};
  value.data=reinterpret_cast<uint64_t>(record->storage);
  value.pitch_bytes=pitch; value.layer_bytes=record->pitch*record->height;
  value.width=static_cast<uint32_t>(width);value.height=static_cast<uint32_t>(height);
  value.layers=static_cast<uint32_t>(record->layers);value.layered=record->layered;
  value.normalized=texture->normalizedCoords;
  value.wrap_x=value.normalized && texture->addressMode[0]==cudaAddressModeWrap;
  value.wrap_y=value.normalized && texture->addressMode[1]==cudaAddressModeWrap;
  value.linear=texture->filterMode==cudaFilterModeLinear
      ? detail::TextureFilterLinear : detail::TextureFilterPoint;
  value.unsigned_element=sample_kind;
  value.abi_version=detail::TextureDescriptorVersion;
  value.error=reinterpret_cast<uint64_t>(descriptor)+sizeof(value);
  unsigned char image[96]={};std::memcpy(image,&value,sizeof(value));
  status=::aclrtMemcpy(descriptor,sizeof(image),image,sizeof(image),ACL_MEMCPY_HOST_TO_DEVICE);
  if(status!=ACL_SUCCESS) {
    const aclError cleanup=::aclrtFree(descriptor);
    return cleanup==ACL_SUCCESS ? status : cleanup;
  }
  *slot={descriptor,record};++record->references;
  *output=reinterpret_cast<cudaTextureObject_t>(descriptor);
  return ACL_SUCCESS;
}
inline aclError cudaDestroyTextureObject(cudaTextureObject_t texture) {
  if(!texture) return ACL_ERROR_RT_PARAM_INVALID;
  int32_t device=-1;void *context=nullptr;
  aclError status=detail::textureCurrent(device,context);
  if(status!=ACL_SUCCESS) return status;
  detail::TextureLock guard;
  for(auto &handle:detail::textureRegistry().handles) {
    if(reinterpret_cast<cudaTextureObject_t>(handle.descriptor)!=texture) continue;
    if(!detail::textureOwned(*handle.allocation,device,context)) return ACL_ERROR_RT_PARAM_INVALID;
    status=::aclrtSynchronizeDevice();
    if(status!=ACL_SUCCESS) return status;
    uint32_t flag=0;
    status=::aclrtMemcpy(&flag,sizeof(flag),static_cast<unsigned char*>(handle.descriptor)+
        sizeof(detail::TextureSamplingDescriptor),sizeof(flag),ACL_MEMCPY_DEVICE_TO_HOST);
    if(status!=ACL_SUCCESS) return status;
    status=::aclrtFree(handle.descriptor);
    if(status!=ACL_SUCCESS) return status;
    auto *record=handle.allocation;
    --record->references;handle={};
    if(!record->storage && !record->references) *record={};
    return flag ? ACL_ERROR_FEATURE_UNSUPPORTED : ACL_SUCCESS;
  }
  return ACL_ERROR_RT_PARAM_INVALID;
}

template<typename T>
__aicore__ ASCIFY_FORCEINLINE T tex2D(cudaTextureObject_t texture,float x,float y) {
  static_assert(detail::textureFetchType<T>::value,
                "Ascify texture fetch requires float32, unsigned int32 or admitted native RGBA vectors");
#if defined(ASCIFY_SIMT_LEGACY_HAS_GM_ATOMICS) && defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)
  if constexpr (std::is_same<T,unsigned int>::value)
    return detail::textureSampleUnsigned(texture,x,y,0,false);
#if defined(ASCIFY_SIMT_LEGACY_HAS_VECTOR_CONSTRUCTORS)
  else if constexpr (detail::textureRGBAElementType<T>::value)
    return detail::textureSampleRGBAElement(texture,x,y,0,false);
  else if constexpr (detail::textureRGBAFloatType<T>::value)
    return detail::textureSampleRGBANormalized(texture,x,y,0,false);
#endif
  else
    return detail::textureSampleFloat(texture,x,y,0,false);
#else
  static_assert(detail::dependentFalse<T>::value,"Ascify texture sampling requires the admitted pure-SIMT API");
  return T{};
#endif
}
template<typename T>
__aicore__ ASCIFY_FORCEINLINE T tex2DLayered(cudaTextureObject_t texture,float x,float y,int layer) {
  static_assert(detail::textureFetchType<T>::value,
                "Ascify texture fetch requires float32, unsigned int32 or admitted native RGBA vectors");
#if defined(ASCIFY_SIMT_LEGACY_HAS_GM_ATOMICS) && defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)
  if constexpr (std::is_same<T,unsigned int>::value)
    return detail::textureSampleUnsigned(texture,x,y,layer,true);
#if defined(ASCIFY_SIMT_LEGACY_HAS_VECTOR_CONSTRUCTORS)
  else if constexpr (detail::textureRGBAElementType<T>::value)
    return detail::textureSampleRGBAElement(texture,x,y,layer,true);
  else if constexpr (detail::textureRGBAFloatType<T>::value)
    return detail::textureSampleRGBANormalized(texture,x,y,layer,true);
#endif
  else
    return detail::textureSampleFloat(texture,x,y,layer,true);
#else
  static_assert(detail::dependentFalse<T>::value,"Ascify texture sampling requires the admitted pure-SIMT API");
  return T{};
#endif
}
template<typename T>
__aicore__ ASCIFY_FORCEINLINE T tex2DFast(cudaTextureObject_t texture,float x,float y) {
  static_assert(detail::textureFetchType<T>::value,
                "Ascify texture fetch requires float32, unsigned int32 or admitted native RGBA vectors");
#if defined(ASCIFY_SIMT_LEGACY_HAS_GM_ATOMICS) && defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)
  if constexpr (std::is_same<T,unsigned int>::value)
    return detail::textureSampleUnsigned(texture,x,y,0,false);
#if defined(ASCIFY_SIMT_LEGACY_HAS_VECTOR_CONSTRUCTORS)
  else if constexpr (detail::textureRGBAElementType<T>::value)
    return detail::textureSampleRGBAElement(texture,x,y,0,false);
  else if constexpr (detail::textureRGBAFloatType<T>::value)
    return detail::textureSampleRGBANormalized<true>(texture,x,y,0,false);
#endif
  else
    return detail::textureSampleFloat<true>(texture,x,y,0,false);
#else
  static_assert(detail::dependentFalse<T>::value,"Ascify texture sampling requires the admitted pure-SIMT API");
  return T{};
#endif
}
template<typename T>
__aicore__ ASCIFY_FORCEINLINE T tex2DLayeredFast(cudaTextureObject_t texture,float x,float y,int layer) {
  static_assert(detail::textureFetchType<T>::value,
                "Ascify texture fetch requires float32, unsigned int32 or admitted native RGBA vectors");
#if defined(ASCIFY_SIMT_LEGACY_HAS_GM_ATOMICS) && defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)
  if constexpr (std::is_same<T,unsigned int>::value)
    return detail::textureSampleUnsigned(texture,x,y,layer,true);
#if defined(ASCIFY_SIMT_LEGACY_HAS_VECTOR_CONSTRUCTORS)
  else if constexpr (detail::textureRGBAElementType<T>::value)
    return detail::textureSampleRGBAElement(texture,x,y,layer,true);
  else if constexpr (detail::textureRGBAFloatType<T>::value)
    return detail::textureSampleRGBANormalized<true>(texture,x,y,layer,true);
#endif
  else
    return detail::textureSampleFloat<true>(texture,x,y,layer,true);
#else
  static_assert(detail::dependentFalse<T>::value,"Ascify texture sampling requires the admitted pure-SIMT API");
  return T{};
#endif
}
} // namespace ascify
#endif
