// Host memory model for the actual public texture lifecycle. The ACL calls
// below are test doubles, not real ACL/GM or target execution evidence.
#define ASCIFY_TEST_CUSTOM_CURRENT_CONTEXT
#define aclrtMalloc ascify_unused_stub_malloc
#define aclrtFree ascify_unused_stub_free
#define aclrtMemcpy ascify_unused_stub_memcpy
#include <acl/acl.h>
#undef aclrtMalloc
#undef aclrtFree
#undef aclrtMemcpy

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>

static std::map<void *, size_t> allocations;
static unsigned allocation_calls = 0, free_calls = 0;
static unsigned model_contexts[2];
static unsigned context_index=0;
inline aclError aclrtGetCurrentContext(void **context) {
  *context = &model_contexts[context_index];
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


#include <cstdio>
#include <vector>
static ascify::cudaTextureDesc cube_texture() {
  ascify::cudaTextureDesc t{};t.normalizedCoords=1;t.filterMode=ascify::cudaFilterModeLinear;
  t.addressMode[0]=t.addressMode[1]=t.addressMode[2]=ascify::cudaAddressModeWrap;
  return t;
}
int main() {
  using namespace ascify;using namespace ascify::detail;
  const auto channel=cudaCreateChannelDesc<float>();
  cudaArray_t array=reinterpret_cast<cudaArray_t>(0x1234);
  const auto prior=array;const unsigned before=allocation_calls;
  for(auto extent:{make_cudaExtent(0,0,6),make_cudaExtent(2,3,6),make_cudaExtent(2,2,5),make_cudaExtent(2,2,12)}) {
    assert(cudaMalloc3DArray(&array,&channel,extent,cudaArrayCubemap)!=ACL_SUCCESS&&array==prior);
  }
  for(unsigned flag:{0U,cudaArrayCubemap|cudaArrayLayered,cudaArrayCubemap|2U})
    assert(cudaMalloc3DArray(&array,&channel,make_cudaExtent(2,2,6),flag)==ACL_ERROR_FEATURE_UNSUPPORTED&&array==prior);
  const auto u=cudaCreateChannelDesc<unsigned int>();
  assert(cudaMalloc3DArray(&array,&u,make_cudaExtent(2,2,6),cudaArrayCubemap)==ACL_ERROR_FEATURE_UNSUPPORTED&&array==prior);
  assert(allocation_calls==before&&allocations.empty());
  unsigned checks=0;
  for(unsigned n:{1U,2U,4U,64U}) {
    assert(cudaMalloc3DArray(&array,&channel,make_cudaExtent(n,n,6),cudaArrayCubemap)==ACL_SUCCESS);
    auto* owner=textureArray(array);assert(owner&&owner->topology==TextureTopologyCube&&owner->references==0);
    std::vector<float> input(6*n*n);for(unsigned i=0;i<input.size();++i)input[i]=float(i);
    cudaMemcpy3DParms p{};p.srcPtr=make_cudaPitchedPtr(input.data(),n*sizeof(float),n,n);
    p.dstArray=array;p.extent=make_cudaExtent(n,n,6);p.kind=ACL_MEMCPY_HOST_TO_DEVICE;
    assert(cudaMemcpy3D(&p)==ACL_SUCCESS);
    cudaResourceDesc resource{};resource.resType=cudaResourceTypeArray;resource.res.array.array=array;
    auto texture=cube_texture();cudaTextureObject_t token=0x1234;
    for(unsigned mutation=0;mutation<4;++mutation) {
      auto bad=texture;switch(mutation){case 0:bad.normalizedCoords=0;break;case 1:bad.seamlessCubemap=1;break;
        case 2:bad.addressMode[2]=cudaAddressModeBorder;break;case 3:bad.readMode=cudaReadModeNormalizedFloat;break;}
      const auto count=allocation_calls;assert(cudaCreateTextureObject(&token,&resource,&bad,nullptr)==ACL_ERROR_FEATURE_UNSUPPORTED);
      assert(token==0x1234&&allocation_calls==count&&owner->references==0);
    }
    assert(cudaCreateTextureObject(&token,&resource,&texture,nullptr)==ACL_SUCCESS);
    auto* d=reinterpret_cast<TextureSamplingDescriptor*>(token);
    assert(d->layered==TextureTopologyCube&&d->unsigned_element==TextureSampleCubeFloat&&d->layers==6);
    assert(d->error==token+72&&allocations.at(d)==96&&owner->references==1);
    for(unsigned face=0;face<6;++face)for(unsigned y=0;y<n;++y)for(unsigned x=0;x<n;++x) {
      const float s=float(2*int(x)+1-int(n))/float(n),t=float(2*int(y)+1-int(n))/float(n);
      float cx=0,cy=0,cz=0;
      switch(face){case 0:cx=1;cy=-t;cz=-s;break;case 1:cx=-1;cy=-t;cz=s;break;
        case 2:cx=s;cy=1;cz=t;break;case 3:cx=s;cy=-1;cz=-t;break;
        case 4:cx=s;cy=-t;cz=1;break;case 5:cx=-s;cy=-t;cz=-1;break;}
      assert(texCubemap<float>(token,cx,cy,cz)==input[face*n*n+y*n+x]);++checks;
    }
    const unsigned frees=free_calls;context_index=1;
    assert(cudaFreeArray(array)==ACL_ERROR_RT_PARAM_INVALID&&free_calls==frees);
    assert(cudaMemcpy3D(&p)==ACL_ERROR_RT_PARAM_INVALID);
    cudaTextureObject_t untouched=0x1234;
    assert(cudaCreateTextureObject(&untouched,&resource,&texture,nullptr)==ACL_ERROR_RT_PARAM_INVALID&&untouched==0x1234);
    context_index=0;
    assert(cudaMemcpyToArray(array,0,0,input.data(),4,ACL_MEMCPY_HOST_TO_DEVICE)==ACL_ERROR_RT_PARAM_INVALID);
    assert(cudaMemcpy2DToArray(array,0,0,input.data(),n*4,n*4,n,ACL_MEMCPY_HOST_TO_DEVICE)==ACL_ERROR_RT_PARAM_INVALID);
    if(n==2) {
      // Real backing free invalidates descriptors but pins owner generation.
      assert(cudaFreeArray(array)==ACL_SUCCESS&&d->width==0&&!owner->storage&&owner->references==1);
      cudaArray_t another=nullptr;
      assert(cudaMalloc3DArray(&another,&channel,make_cudaExtent(2,2,6),cudaArrayCubemap)==ACL_SUCCESS&&another!=array);
      assert(std::isnan(texCubemap<float>(token,1,.5f,.5f)));
      assert(cudaDestroyTextureObject(token)==ACL_ERROR_FEATURE_UNSUPPORTED);
      assert(cudaFreeArray(another)==ACL_SUCCESS);
    } else {
      assert(cudaDestroyTextureObject(token)==ACL_SUCCESS&&owner->references==0);
      assert(cudaFreeArray(array)==ACL_SUCCESS);
    }
    const unsigned after=free_calls;
    assert(cudaFreeArray(array)==ACL_ERROR_RT_PARAM_INVALID&&free_calls==after);
    assert(allocations.empty());
  }
  // Allocation/copy topology is independent from the narrower read domain.
  assert(cudaMalloc3DArray(&array,&channel,make_cudaExtent(3,3,6),cudaArrayCubemap)==ACL_SUCCESS);
  cudaResourceDesc resource{};resource.resType=cudaResourceTypeArray;resource.res.array.array=array;
  auto texture=cube_texture();cudaTextureObject_t token=0;
  assert(cudaCreateTextureObject(&token,&resource,&texture,nullptr)==ACL_SUCCESS);
  assert(std::isnan(texCubemap<float>(token,1,0,0)));
  assert(cudaDestroyTextureObject(token)==ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(cudaFreeArray(array)==ACL_SUCCESS&&allocations.empty());
  assert(allocation_calls==free_calls);
  std::printf("cube lifecycle CPU fetches=%u allocations=%u frees=%u; no ACL device execution\n",checks,allocation_calls,free_calls);
}
