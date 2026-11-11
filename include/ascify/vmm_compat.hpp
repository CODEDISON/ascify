#ifndef ASCIFY_VMM_COMPAT_HPP
#define ASCIFY_VMM_COMPAT_HPP

#include <acl/acl.h>
#include <cstddef>

// CUresult is the Driver-API status domain.
using CUresult = aclError;
#define CUDA_SUCCESS ACL_SUCCESS
#define CUDA_ERROR_INVALID_VALUE ACL_ERROR_RT_PARAM_INVALID

using CUdevice = int32_t;
using CUdeviceptr = void *;
using CUmemGenericAllocationHandle = void *;

enum CUmemAllocationLocationType { CU_MEM_LOCATION_TYPE_DEVICE = 1 };
enum CUmemAllocationGranularity_flags {
  CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0,
  CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1
};
enum CUmemHandleType { CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1 };

struct CUmemLocation {
  CUmemAllocationLocationType type;
  int id;
};
struct CUmemAllocationProp {
  CUmemAllocationLocationType locationType;
  CUmemLocation location;
  CUmemHandleType allocFlagsHandleType;
  unsigned compressionType;
  unsigned requestedHandleTypes;
};

enum CUdevice_attribute_vmm {
  CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
  CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 108
};

namespace ascify {

inline CUresult cuCtxGetDevice(CUdevice *device) {
  if (device == nullptr) return CUDA_ERROR_INVALID_VALUE;
  int32_t current = -1;
  const aclError status = aclrtGetDevice(&current);
  if (status != ACL_SUCCESS) return static_cast<CUresult>(status);
  *device = current;
  return CUDA_SUCCESS;
}

inline CUresult cuMemAddressReserve(void **ptr, size_t size,
                                    size_t alignment, void *addr,
                                    unsigned flags) {
  (void)alignment; (void)addr; (void)flags;
  if (ptr == nullptr || size == 0) return CUDA_ERROR_INVALID_VALUE;
  return static_cast<CUresult>(
      aclrtMalloc(ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
}

inline CUresult cuMemAddressFree(void *ptr, size_t size) {
  (void)size;
  if (ptr == nullptr) return CUDA_ERROR_INVALID_VALUE;
  return static_cast<CUresult>(aclrtFree(ptr));
}

inline CUresult cuMemCreate(CUmemGenericAllocationHandle *handle,
                            size_t size, const CUmemAllocationProp *prop,
                            unsigned flags) {
  (void)prop; (void)flags;
  if (handle == nullptr || size == 0) return CUDA_ERROR_INVALID_VALUE;
  return static_cast<CUresult>(
      aclrtMalloc(handle, size, ACL_MEM_MALLOC_HUGE_FIRST));
}

inline CUresult cuMemMap(void *ptr, size_t size, size_t offset,
                         CUmemGenericAllocationHandle handle,
                         unsigned flags) {
  (void)ptr; (void)size; (void)offset; (void)flags;
  if (handle == nullptr) return CUDA_ERROR_INVALID_VALUE;
  return CUDA_SUCCESS;
}

inline CUresult cuMemUnmap(void *ptr, size_t size) {
  (void)ptr; (void)size;
  return CUDA_SUCCESS;
}

inline CUresult cuMemSetAccess(void *ptr, size_t size,
                               const void *descs, size_t count) {
  (void)ptr; (void)size; (void)descs; (void)count;
  return CUDA_SUCCESS;
}

inline CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
  if (handle == nullptr) return CUDA_ERROR_INVALID_VALUE;
  return static_cast<CUresult>(aclrtFree(handle));
}

inline CUresult cuMemGetAllocationGranularity(
    size_t *granularity, const CUmemAllocationProp *prop,
    CUmemAllocationGranularity_flags flags) {
  (void)prop; (void)flags;
  if (granularity == nullptr) return CUDA_ERROR_INVALID_VALUE;
  *granularity = 2097152;
  return CUDA_SUCCESS;
}

inline CUresult cuMemGetAllocationPropertiesFromHandle(
    CUmemAllocationProp *prop, CUmemGenericAllocationHandle handle) {
  if (prop == nullptr || handle == nullptr) return CUDA_ERROR_INVALID_VALUE;
  prop->locationType = CU_MEM_LOCATION_TYPE_DEVICE;
  prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop->location.id = 0;
  prop->compressionType = 0;
  return CUDA_SUCCESS;
}

}  // namespace ascify
#endif
