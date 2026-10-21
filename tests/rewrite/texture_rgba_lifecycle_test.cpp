// RGBA host model for the actual public texture lifecycle. The ACL calls
// below are test doubles, not real ACL/GM or target execution evidence.
#define aclrtMalloc ascify_unused_stub_malloc
#define aclrtFree ascify_unused_stub_free
#define aclrtMemcpy ascify_unused_stub_memcpy
#define aclrtGetDevice ascify_unused_stub_device
#include <acl/acl.h>
#undef aclrtMalloc
#undef aclrtFree
#undef aclrtMemcpy
#undef aclrtGetDevice

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>

static std::map<void *, size_t> allocations;
static unsigned allocation_calls = 0, free_calls = 0;
static int32_t model_device = 0;
static unsigned model_context = 0;
static unsigned other_context = 0;
static void *active_context = &model_context;
static aclError provider_status = ACL_SUCCESS;
inline aclError aclrtGetDevice(int32_t *device) { *device=model_device; return ACL_SUCCESS; }
inline aclError aclrtGetCurrentContext(void **context) {
  *context = active_context;
  return ACL_SUCCESS;
}
static bool device_range(const void *pointer, size_t count) {
  const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
  for (const auto &allocation : allocations) {
    const uintptr_t base = reinterpret_cast<uintptr_t>(allocation.first);
    if (address >= base && address - base <= allocation.second &&
        count <= allocation.second - (address - base)) return true;
  }
  return false;
}
inline aclError aclrtMalloc(void **pointer, size_t bytes, aclrtMemMallocPolicy) {
  if (provider_status!=ACL_SUCCESS) return provider_status;
  assert(pointer && bytes);
  void *storage = std::malloc(bytes);
  assert(storage);
  // Nonzero initial contents catch a missing descriptor/sticky upload.
  std::memset(storage, 0xa5, bytes);
  assert(allocations.emplace(storage, bytes).second);
  ++allocation_calls;
  *pointer = storage;
  return ACL_SUCCESS;
}
inline aclError aclrtFree(void *pointer) {
  assert(allocations.erase(pointer) == 1);
  ++free_calls;
  std::free(pointer);
  return ACL_SUCCESS;
}
inline aclError aclrtMemcpy(void *destination, size_t capacity,
                          const void *source, size_t count, aclrtMemcpyKind kind) {
  assert(destination && source && capacity >= count);
  if (kind == ACL_MEMCPY_HOST_TO_DEVICE || kind == ACL_MEMCPY_DEVICE_TO_DEVICE)
    assert(device_range(destination, count));
  if (kind == ACL_MEMCPY_DEVICE_TO_HOST || kind == ACL_MEMCPY_DEVICE_TO_DEVICE)
    assert(device_range(source, count));
  std::memmove(destination, source, count);
  return ACL_SUCCESS;
}

#define __aicore__
#include <ascify/ascify_cuda_compat.hpp>
#define __NPU_COMPILER_INTERNAL_PURE_SIMT__
#include <ascify/texture_compat.hpp>

static ascify::cudaTextureDesc texture(bool normalized, bool linear) {
  ascify::cudaTextureDesc t{};
  t.readMode=normalized?ascify::cudaReadModeNormalizedFloat:ascify::cudaReadModeElementType;
  t.filterMode=linear?ascify::cudaFilterModeLinear:ascify::cudaFilterModePoint;
  return t; // Original CUDA zero descriptor uses Wrap; non-normalized clamps.
}
static void check(ascify::cudaTextureObject_t token, unsigned red, unsigned alpha, bool normalized) {
  const auto *d=reinterpret_cast<const ascify::detail::TextureSamplingDescriptor *>(token);
  assert(d->abi_version==ascify::detail::TextureDescriptorVersion && d->error==token+72);
  assert(d->wrap_x==0 && d->wrap_y==0 && d->normalized==0);
  assert(d->unsigned_element==(normalized?ascify::detail::TextureSampleRGBANormalized:
                                          ascify::detail::TextureSampleRGBAElement));
  if(normalized) {
    const auto v=ascify::tex2DFast<float4>(token,.5f,.5f);
    assert(std::fabs(double(v.x)-double(red)/255)<0x1p-21 &&
           std::fabs(double(v.w)-double(alpha)/255)<0x1p-21);
  } else { const auto v=ascify::tex2D<uchar4>(token,0,0);assert(v.x==red && v.w==alpha); }
}
int main() {
  using namespace ascify;
  const auto channel=cudaCreateChannelDesc<uchar4>();
  const unsigned char pixels[24]={0,17,33,255, 255,0,11,37, 1,2,3,4,
                                  4,3,2,1, 10,20,30,40, 91,92,93,94};
  cudaArray_t array=nullptr;
  assert(cudaMallocArray(&array,&channel,3,2)==ACL_SUCCESS);
  assert(cudaMemcpyToArray(array,0,0,pixels,sizeof(pixels),ACL_MEMCPY_HOST_TO_DEVICE)==ACL_SUCCESS);
  cudaResourceDesc resource{};resource.resType=cudaResourceTypeArray;resource.res.array.array=array;
  for(bool normalized:{false,true}) for(bool linear:{false,true}) {
    auto t=texture(normalized,linear);cudaTextureObject_t token=99;
    const unsigned before=allocation_calls;
    const auto status=cudaCreateTextureObject(&token,&resource,&t,nullptr);
    if(!normalized&&linear) { assert(status==ACL_ERROR_FEATURE_UNSUPPORTED && token==99 && allocation_calls==before);continue; }
    assert(status==ACL_SUCCESS);check(token,0,255,normalized);
    assert(cudaDestroyTextureObject(token)==ACL_SUCCESS);
  }
  auto t=texture(true,false);cudaTextureObject_t token=0,second=0;
  assert(cudaCreateTextureObject(&token,&resource,&t,nullptr)==ACL_SUCCESS);
  assert(cudaFreeArray(array)==ACL_SUCCESS);
  const auto invalid=tex2D<float4>(token,0,0);
  assert(std::isnan(invalid.x) && std::isnan(invalid.w));
  assert(cudaDeviceSynchronize()==ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(cudaDestroyTextureObject(token)==ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(allocations.empty());

  // Pitch storage can have two differently interpreted views. Creation must
  // not mutate its shared allocation format or the first live descriptor.
  void *storage=nullptr;size_t pitch=0;
  assert(cudaMallocPitch(&storage,&pitch,12,2)==ACL_SUCCESS);
  assert(cudaMemcpy2D(storage,pitch,pixels,12,12,2,ACL_MEMCPY_HOST_TO_DEVICE)==ACL_SUCCESS);
  resource={};resource.resType=cudaResourceTypePitch2D;
  resource.res.pitch2D={storage,channel,3,2,pitch};
  t=texture(false,false);
  assert(cudaCreateTextureObject(&token,&resource,&t,nullptr)==ACL_SUCCESS);
  resource.res.pitch2D.desc=cudaCreateChannelDesc<unsigned int>();
  assert(cudaCreateTextureObject(&second,&resource,&t,nullptr)==ACL_SUCCESS);
  check(token,0,255,false);
  unsigned int expected_word=0;std::memcpy(&expected_word,pixels,4);
  assert(tex2D<unsigned int>(second,0,0)==expected_word);
  assert(cudaDestroyTextureObject(second)==ACL_SUCCESS);
  assert(cudaDestroyTextureObject(token)==ACL_SUCCESS);
  assert(cudaFree(storage)==ACL_SUCCESS && allocations.empty());

  // All invalid descriptions fail before allocation and preserve outputs.
  const auto before_invalid=allocation_calls;
  for(auto bad:{cudaCreateChannelDesc(8,8,8,0,cudaChannelFormatKindUnsigned),
                cudaCreateChannelDesc(8,8,8,8,cudaChannelFormatKindSigned),
                cudaCreateChannelDesc(16,16,16,16,cudaChannelFormatKindUnsigned),
                cudaCreateChannelDesc(32,32,32,32,cudaChannelFormatKindFloat)}) {
    array=reinterpret_cast<cudaArray_t>(uintptr_t(123));
    assert(cudaMallocArray(&array,&bad,3,2)==ACL_ERROR_FEATURE_UNSUPPORTED);
    assert(array==reinterpret_cast<cudaArray_t>(uintptr_t(123)) && allocation_calls==before_invalid);
  }
  array=nullptr;
  assert(cudaMallocArray(&array,&channel,SIZE_MAX,2)==ACL_ERROR_FEATURE_UNSUPPORTED && !array);
  assert(cudaMallocArray(&array,&channel,3,2)==ACL_SUCCESS);
  resource={};resource.resType=cudaResourceTypeArray;resource.res.array.array=array;
  t=texture(true,true);token=123;
  provider_status=ACL_ERROR_BAD_ALLOC;
  assert(cudaCreateTextureObject(&token,&resource,&t,nullptr)==ACL_ERROR_BAD_ALLOC && token==123);
  provider_status=ACL_SUCCESS;model_device=1;
  assert(cudaCreateTextureObject(&token,&resource,&t,nullptr)==ACL_ERROR_RT_PARAM_INVALID && token==123);
  model_device=0;
  active_context=&other_context;
  assert(cudaCreateTextureObject(&token,&resource,&t,nullptr)==ACL_ERROR_RT_PARAM_INVALID && token==123);
  active_context=&model_context;
  assert(cudaFreeArray(array)==ACL_SUCCESS && allocations.empty());

  // Layered 1x1 storage, full copy and channel preservation, including clamp
  // at arbitrary finite coordinates. Point and linear both return same texel.
  assert(cudaMalloc3DArray(&array,&channel,make_cudaExtent(1,1,2),cudaArrayLayered)==ACL_SUCCESS);
  cudaMemcpy3DParms copy{};copy.srcPtr=make_cudaPitchedPtr(const_cast<unsigned char*>(pixels),4,4,1);
  copy.dstArray=array;copy.extent=make_cudaExtent(1,1,2);copy.kind=ACL_MEMCPY_HOST_TO_DEVICE;
  assert(cudaMemcpy3D(&copy)==ACL_SUCCESS);
  resource.res.array.array=array;
  for(bool linear:{false,true}) {
    t=texture(true,linear);assert(cudaCreateTextureObject(&token,&resource,&t,nullptr)==ACL_SUCCESS);
    const auto first=tex2DLayeredFast<float4>(token,-100,100,0);
    const auto last=tex2DLayeredFast<float4>(token,100,-100,1);
    assert(first.x==0 && first.w==1 && last.x==1 && std::fabs(last.w-37.0f/255)<0x1p-21);
    assert(cudaDestroyTextureObject(token)==ACL_SUCCESS);
  }
  // Reset callback discards descriptors/registry only after the provider has
  // actually released all allocations. This is a CPU provider model.
  t=texture(false,false);assert(cudaCreateTextureObject(&token,&resource,&t,nullptr)==ACL_SUCCESS);
  while(!allocations.empty())assert(aclrtFree(allocations.begin()->first)==ACL_SUCCESS);
  detail::textureAfterReset(0);
  assert(cudaDestroyTextureObject(token)==ACL_ERROR_RT_PARAM_INVALID);
  assert(cudaMallocArray(&array,&channel,1,1)==ACL_SUCCESS);
  assert(cudaFreeArray(array)==ACL_SUCCESS);
  assert(allocations.empty() && allocation_calls==free_calls);
}

