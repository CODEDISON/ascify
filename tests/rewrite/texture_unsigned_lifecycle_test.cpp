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
inline aclError aclrtGetCurrentContext(void **context) {
  static unsigned model_context;
  *context = &model_context;
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

static ascify::cudaTextureDesc point_texture() {
  ascify::cudaTextureDesc texture{};
  texture.addressMode[0] = texture.addressMode[1] = texture.addressMode[2] =
      ascify::cudaAddressModeClamp;
  return texture;
}
static void check_descriptor(ascify::cudaTextureObject_t token, bool layered) {
  const auto *descriptor = reinterpret_cast<const ascify::detail::TextureSamplingDescriptor *>(token);
  assert(allocations.at(const_cast<ascify::detail::TextureSamplingDescriptor *>(descriptor)) == 96);
  assert(descriptor->unsigned_element == ascify::detail::TextureSampleUnsigned &&
         descriptor->layered == layered);
  assert(descriptor->abi_version == ascify::detail::TextureDescriptorVersion &&
         descriptor->linear == ascify::detail::TextureFilterPoint);
  assert(descriptor->error == token + 72);
  assert(*reinterpret_cast<const uint32_t *>(descriptor->error) == 0);
}

int main() {
  const auto channel = ascify::cudaCreateChannelDesc<unsigned int>();
  unsigned int values[] = {0U, 16777217U, 0xffffffffU, 0x80000000U,
                           0x7f800000U, 0x7fc00001U, 9U, 17U};
  const size_t row_bytes = 4 * sizeof(unsigned int);
  auto texture = point_texture();

  // Public array allocation, upload, descriptor creation and sticky copyback.
  ascify::cudaArray_t array = nullptr;
  assert(ascify::cudaMallocArray(&array, &channel, 4, 2) == ACL_SUCCESS);
  assert(ascify::cudaMemcpyToArray(array, 0, 0, values, sizeof(values),
                                 ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS);
  ascify::cudaResourceDesc resource{};
  resource.resType = ascify::cudaResourceTypeArray;
  resource.res.array.array = array;
  ascify::cudaTextureObject_t token = 0;
  assert(ascify::cudaCreateTextureObject(&token, &resource, &texture, nullptr) == ACL_SUCCESS);
  check_descriptor(token, false);
  for (unsigned y = 0; y != 2; ++y)
    for (unsigned x = 0; x != 4; ++x)
      assert(ascify::tex2D<unsigned int>(token, x, y) == values[y * 4 + x]);
  assert(ascify::cudaDeviceSynchronize() == ACL_SUCCESS);
  assert(std::isnan(ascify::tex2D<float>(token, 0, 0)));
  assert(ascify::cudaDeviceSynchronize() == ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(ascify::cudaMemcpy2DToArray(array,0,0,values,row_bytes,row_bytes,2,
      ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(ascify::cudaDeviceSynchronize() == ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(ascify::cudaDestroyTextureObject(token) == ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(ascify::cudaFreeArray(array) == ACL_SUCCESS);
  assert(allocations.empty());

  // Every failed public create leaves the output and allocation set intact.
  assert(ascify::cudaMallocArray(&array, &channel, 4, 2) == ACL_SUCCESS);
  resource.res.array.array = array;
  const unsigned before_invalid = allocation_calls;
  token = 123;
  texture.filterMode = ascify::cudaFilterModeLinear;
  assert(ascify::cudaCreateTextureObject(&token, &resource, &texture, nullptr) == ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(token == 123 && allocation_calls == before_invalid);
  texture = point_texture();
  texture.readMode = ascify::cudaReadModeNormalizedFloat;
  assert(ascify::cudaCreateTextureObject(&token, &resource, &texture, nullptr) == ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(token == 123 && allocation_calls == before_invalid);
  texture = point_texture();
  assert(ascify::cudaFreeArray(array) == ACL_SUCCESS);

  // The pitched allocation path uses the resource channel, retains padding,
  // and invalidates all live objects before backing storage is actually freed.
  unsigned int *pitched = nullptr;
  size_t pitch = 0;
  assert(ascify::cudaMallocPitch(&pitched, &pitch, row_bytes, 2) == ACL_SUCCESS);
  assert(pitch >= row_bytes);
  assert(ascify::cudaMemcpy2D(pitched, pitch, values, row_bytes, row_bytes, 2,
                            ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS);
  resource = {};
  resource.resType = ascify::cudaResourceTypePitch2D;
  resource.res.pitch2D = {pitched, channel, 4, 2, pitch};
  assert(ascify::cudaCreateTextureObject(&token, &resource, &texture, nullptr) == ACL_SUCCESS);
  check_descriptor(token, false);
  ascify::cudaTextureObject_t second = 0;
  assert(ascify::cudaCreateTextureObject(&second, &resource, &texture, nullptr) == ACL_SUCCESS);
  for (unsigned y = 0; y != 2; ++y)
    for (unsigned x = 0; x != 4; ++x)
      assert(ascify::tex2DFast<unsigned int>(token, x, y) == values[y * 4 + x]);
  assert(ascify::cudaFree(pitched) == ACL_SUCCESS);
  assert(allocations.size() == 2); // The two descriptors outlive the backing.
  assert(ascify::tex2D<unsigned int>(token, 0, 0) == 0U);
  assert(ascify::tex2D<unsigned int>(second, 0, 0) == 0U);
  assert(ascify::cudaDeviceSynchronize() == ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(ascify::cudaDestroyTextureObject(token) == ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(ascify::cudaDestroyTextureObject(second) == ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(allocations.empty());

  // Public layered allocation plus 3D copy uses element-width extent for an
  // array endpoint. Read both layers and copy back the complete payload.
  assert(ascify::cudaMalloc3DArray(&array, &channel,
      ascify::make_cudaExtent(2, 2, 2), ascify::cudaArrayLayered) == ACL_SUCCESS);
  ascify::cudaMemcpy3DParms copy{};
  copy.srcPtr = ascify::make_cudaPitchedPtr(values, 2 * sizeof(unsigned int),
                                          2 * sizeof(unsigned int), 2);
  copy.dstArray = array;
  copy.extent = ascify::make_cudaExtent(2, 2, 2);
  copy.kind = ACL_MEMCPY_HOST_TO_DEVICE;
  assert(ascify::cudaMemcpy3D(&copy) == ACL_SUCCESS);
  resource = {};
  resource.resType = ascify::cudaResourceTypeArray;
  resource.res.array.array = array;
  assert(ascify::cudaCreateTextureObject(&token, &resource, &texture, nullptr) == ACL_SUCCESS);
  check_descriptor(token, true);
  for (unsigned layer = 0; layer != 2; ++layer)
    for (unsigned y = 0; y != 2; ++y)
      for (unsigned x = 0; x != 2; ++x)
        assert(ascify::tex2DLayered<unsigned int>(token, x, y, layer) == values[layer * 4 + y * 2 + x]);
  unsigned int readback[8]{};
  copy = {};
  copy.srcArray = array;
  copy.dstPtr = ascify::make_cudaPitchedPtr(readback, 2 * sizeof(unsigned int),
                                          2 * sizeof(unsigned int), 2);
  copy.extent = ascify::make_cudaExtent(2, 2, 2);
  copy.kind = ACL_MEMCPY_DEVICE_TO_HOST;
  assert(ascify::cudaMemcpy3D(&copy) == ACL_SUCCESS);
  assert(std::memcmp(readback, values, sizeof(values)) == 0);
  assert(ascify::cudaDestroyTextureObject(token) == ACL_SUCCESS);
  assert(ascify::cudaFreeArray(array) == ACL_SUCCESS);

  // Actual host create publishes mode2 only for normalized mirror. The
  // unnormalized setting becomes clamp; third-axis mirror does not alter XY.
  assert(ascify::cudaMallocArray(&array,&channel,4,2)==ACL_SUCCESS);
  assert(ascify::cudaMemcpyToArray(array,0,0,values,sizeof(values),ACL_MEMCPY_HOST_TO_DEVICE)==ACL_SUCCESS);
  resource.res.array.array=array;
  for(bool norm:{false,true})for(unsigned axis=0;axis<3;++axis) {
    texture=point_texture();texture.normalizedCoords=norm;
    texture.addressMode[axis]=ascify::cudaAddressModeMirror;
    assert(ascify::cudaCreateTextureObject(&token,&resource,&texture,nullptr)==ACL_SUCCESS);
    const auto *d=reinterpret_cast<const ascify::detail::TextureSamplingDescriptor *>(token);
    assert(d->wrap_x==(norm&&axis==0?2U:0U)&&d->wrap_y==(norm&&axis==1?2U:0U));
    const float x=axis==0?-.25f:.25f,y=axis==1?-.75f:.25f;
    const unsigned expected=!norm?values[0]:axis==1?values[5]:values[1];
    assert(ascify::tex2D<unsigned int>(token,x,y)==expected);
    assert(ascify::cudaDestroyTextureObject(token)==ACL_SUCCESS);
  }
  texture=point_texture();texture.normalizedCoords=1;
  texture.addressMode[0]=ascify::cudaAddressModeMirror;
  texture.addressMode[1]=ascify::cudaAddressModeWrap;
  assert(ascify::cudaCreateTextureObject(&token,&resource,&texture,nullptr)==ACL_SUCCESS);
  assert(ascify::tex2D<unsigned int>(token,-.25f,-.25f)==values[5]);
  assert(ascify::cudaDestroyTextureObject(token)==ACL_SUCCESS);
  const unsigned before_border=allocation_calls;
  for(unsigned axis=0;axis<3;++axis) {
    texture=point_texture();texture.addressMode[axis]=ascify::cudaAddressModeBorder;token=123;
    assert(ascify::cudaCreateTextureObject(&token,&resource,&texture,nullptr)==ACL_ERROR_FEATURE_UNSUPPORTED);
    assert(token==123&&allocation_calls==before_border);
  }
  assert(ascify::cudaFreeArray(array)==ACL_SUCCESS);
  assert(allocations.empty() && allocation_calls == free_calls);
}
