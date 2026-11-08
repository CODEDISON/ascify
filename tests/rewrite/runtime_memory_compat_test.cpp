#define __aicore__
#define ASCIFY_TEST_ACL_CONTROLLABLE_RUNTIME_MEMORY
#define ASCIFY_TEST_ACL_COPY_ORDER

#include <ascify/ascify_cuda_compat.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

aclError ascify_test_runtime_status = ACL_SUCCESS;
void* ascify_test_allocation_result = nullptr;
const char* ascify_test_recent_error_message = "recorded ACL failure";
aclError ascify_test_last_error = ACL_SUCCESS;
int ascify_test_malloc_calls = 0;
int ascify_test_malloc_host_calls = 0;
int ascify_test_free_calls = 0;
int ascify_test_free_host_calls = 0;
int ascify_test_host_register_calls = 0;
int ascify_test_host_unregister_calls = 0;
int ascify_test_host_get_device_pointer_calls = 0;
void* ascify_test_host_register_pointer = nullptr;
void* ascify_test_host_unregister_pointer = nullptr;
aclrtHostRegisterType ascify_test_host_register_type = ACL_HOST_REGISTER_MAPPED;
int ascify_test_binary_load_calls = 0;
int ascify_test_binary_get_function_calls = 0;
const char* ascify_test_get_function_name = nullptr;
aclError ascify_test_binary_status = ACL_SUCCESS;
int ascify_test_memcpy_calls = 0;
int ascify_test_memcpy_async_calls = 0;
int ascify_test_memset_calls = 0;
int ascify_test_memset_async_calls = 0;
void** ascify_test_output_pointer = nullptr;
void* ascify_test_destination = nullptr;
const void* ascify_test_source = nullptr;
size_t ascify_test_destination_max = 0;
size_t ascify_test_count = 0;
int ascify_test_value = 0;
aclrtMemcpyKind ascify_test_memcpy_kind = ACL_MEMCPY_HOST_TO_HOST;
aclrtStream ascify_test_stream = nullptr;
aclrtMemMallocPolicy ascify_test_malloc_policy = ACL_MEM_MALLOC_HUGE_FIRST;
int ascify_test_copy_order = 0;
int ascify_test_copy_sequence = 0;
int ascify_test_sync_sequence = 0;
int ascify_test_sync_calls = 0;
aclrtStream ascify_test_sync_stream = nullptr;
aclError ascify_test_sync_status = ACL_SUCCESS;

namespace {

void resetRecordedCall() {
  ascify_test_host_register_pointer = nullptr;
  ascify_test_host_unregister_pointer = nullptr;
  ascify_test_destination = nullptr;
  ascify_test_source = nullptr;
  ascify_test_destination_max = 0;
  ascify_test_count = 0;
  ascify_test_value = 0;
  ascify_test_memcpy_kind = ACL_MEMCPY_HOST_TO_HOST;
  ascify_test_stream = nullptr;
}

void testAllocationContracts() {
  alignas(int) unsigned char allocation[sizeof(int)] = {};
  ascify_test_allocation_result = allocation;
  ascify_test_runtime_status = ACL_SUCCESS;

  int* device = reinterpret_cast<int*>(1);
  assert(ascify::cudaMalloc(&device, sizeof(allocation)) == ACL_SUCCESS);
  assert(device == reinterpret_cast<int*>(allocation));
  assert(ascify_test_malloc_calls == 1);
  assert(ascify_test_malloc_policy == ACL_MEM_MALLOC_HUGE_FIRST);

  int* host = reinterpret_cast<int*>(1);
  assert(ascify::cudaMallocHost(&host, sizeof(allocation)) == ACL_SUCCESS);
  assert(host == reinterpret_cast<int*>(allocation));
  assert(ascify_test_malloc_host_calls == 1);

  host = nullptr;
  assert(ascify::cudaMallocHost(&host, sizeof(allocation), 0) == ACL_SUCCESS);
  assert(host == reinterpret_cast<int*>(allocation));
  void* raw_host = nullptr;
  assert(ascify::cudaMallocHost(&raw_host, sizeof(allocation), 0) == ACL_SUCCESS);
  assert(raw_host == allocation);
  assert(ascify_test_malloc_host_calls == 3);
  for (unsigned flags : {1u, 2u, 4u, 7u, 0xffffffffu}) {
    host = reinterpret_cast<int*>(1);
    assert(ascify::cudaMallocHost(&host, 32, flags) == ACL_ERROR_FEATURE_UNSUPPORTED);
    assert(host == nullptr);
  }
  assert(ascify::cudaMallocHost(static_cast<int**>(nullptr), 1, 0) ==
         ACL_ERROR_RT_PARAM_INVALID);
  assert(ascify_test_malloc_host_calls == 3);

  assert(ascify::cudaMalloc(static_cast<void**>(nullptr), 1) ==
         ACL_ERROR_RT_PARAM_INVALID);
  assert(ascify::cudaMallocHost(static_cast<void**>(nullptr), 1) ==
         ACL_ERROR_RT_PARAM_INVALID);
  assert(ascify_test_malloc_calls == 1);
  assert(ascify_test_malloc_host_calls == 3);

  ascify_test_runtime_status = ACL_ERROR_RT_MEMORY_ALLOCATION;
  device = reinterpret_cast<int*>(1);
  host = reinterpret_cast<int*>(1);
  assert(ascify::cudaMalloc(&device, 32) == ACL_ERROR_RT_MEMORY_ALLOCATION);
  assert(ascify::cudaMallocHost(&host, 32) == ACL_ERROR_RT_MEMORY_ALLOCATION);
  assert(device == nullptr);
  assert(host == nullptr);
  host = reinterpret_cast<int*>(1);
  assert(ascify::cudaMallocHost(&host, 32, 0) == ACL_ERROR_RT_MEMORY_ALLOCATION);
  assert(host == nullptr);

  ascify_test_runtime_status = ACL_SUCCESS;
  assert(ascify::cudaFree(allocation) == ACL_SUCCESS);
  assert(ascify_test_free_calls == 1);
  assert(ascify_test_destination == allocation);
  assert(ascify::cudaFree(nullptr) == ACL_SUCCESS);
  assert(ascify_test_free_calls == 1);
  assert(ascify::cudaFreeHost(allocation) == ACL_SUCCESS);
  assert(ascify_test_free_host_calls == 1);
  assert(ascify_test_destination == allocation);
}

void testCopyContracts() {
  unsigned char source[16] = {};
  unsigned char destination[16] = {};
  aclrtStream stream = reinterpret_cast<aclrtStream>(0x1234);
  ascify_test_runtime_status = ACL_SUCCESS;
  resetRecordedCall();

  assert(ascify::cudaMemcpy(destination, source, sizeof(source),
                            ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS);
  assert(ascify_test_memcpy_calls == 1);
  assert(ascify_test_destination == destination);
  assert(ascify_test_source == source);
  assert(ascify_test_destination_max == sizeof(source));
  assert(ascify_test_count == sizeof(source));
  assert(ascify_test_memcpy_kind == ACL_MEMCPY_HOST_TO_DEVICE);

  assert(ascify::cudaMemcpyAsync(destination, source, 7,
                                 ACL_MEMCPY_DEVICE_TO_HOST,
                                 stream) == ACL_SUCCESS);
  assert(ascify_test_memcpy_async_calls == 1);
  assert(ascify_test_destination_max == 7);
  assert(ascify_test_count == 7);
  assert(ascify_test_memcpy_kind == ACL_MEMCPY_DEVICE_TO_HOST);
  assert(ascify_test_stream == stream);

  const int sync_calls = ascify_test_memcpy_calls;
  assert(ascify::cudaMemcpy(nullptr, source, 1,
                            ACL_MEMCPY_HOST_TO_DEVICE) ==
         ACL_ERROR_RT_PARAM_INVALID);
  assert(ascify::cudaMemcpy(destination, nullptr, 1,
                            ACL_MEMCPY_HOST_TO_DEVICE) ==
         ACL_ERROR_RT_PARAM_INVALID);
  assert(ascify::cudaMemcpy(destination, source, 1,
                            static_cast<aclrtMemcpyKind>(99)) ==
         ACL_ERROR_RT_PARAM_INVALID);
  assert(ascify_test_memcpy_calls == sync_calls);

  assert(ascify::cudaMemcpy(nullptr, nullptr, 0,
                            ACL_MEMCPY_HOST_TO_HOST) == ACL_SUCCESS);
  assert(ascify_test_memcpy_calls == sync_calls);
  const int async_calls = ascify_test_memcpy_async_calls;
  assert(ascify::cudaMemcpyAsync(nullptr, nullptr, 0,
                                 ACL_MEMCPY_DEVICE_TO_DEVICE,
                                 stream) == ACL_SUCCESS);
  assert(ascify_test_memcpy_async_calls == async_calls);
  assert(ascify::cudaMemcpy(nullptr, nullptr, 0,
                            static_cast<aclrtMemcpyKind>(99)) ==
         ACL_ERROR_RT_PARAM_INVALID);
  assert(ascify_test_memcpy_calls == sync_calls);

  ascify_test_runtime_status = ACL_ERROR_RT_PARAM_INVALID;
  assert(ascify::cudaMemcpy(destination, source, sizeof(source),
                            ACL_MEMCPY_DEVICE_TO_DEVICE) ==
         ACL_ERROR_RT_PARAM_INVALID);
}

void testDefaultStreamCopyOrdering() {
  unsigned char source[16] = {}, destination[16] = {};
  ascify_test_runtime_status = ACL_SUCCESS;
  for (aclrtMemcpyKind kind : {ACL_MEMCPY_HOST_TO_DEVICE,
                              ACL_MEMCPY_DEVICE_TO_HOST, ACL_MEMCPY_DEVICE_TO_DEVICE}) {
    const int copies = ascify_test_memcpy_calls;
    const int waits = ascify_test_sync_calls;
    ascify_test_copy_order = 0;
    ascify_test_sync_stream = reinterpret_cast<aclrtStream>(1);
    assert(ascify::cudaMemcpy(destination, source, sizeof(source), kind) == ACL_SUCCESS);
    assert(ascify_test_sync_calls == waits + 1);
    assert(ascify_test_sync_stream == nullptr);
    assert(ascify_test_sync_sequence == 1 && ascify_test_copy_sequence == 2);
    assert(ascify_test_memcpy_calls == copies + 1);
    ascify_test_sync_status = 98765;
    assert(ascify::cudaMemcpy(destination, source, sizeof(source), kind) == 98765);
    assert(ascify_test_memcpy_calls == copies + 1);
    ascify_test_sync_status = ACL_SUCCESS;
    ascify_test_runtime_status = 98766;
    assert(ascify::cudaMemcpy(destination, source, sizeof(source), kind) == 98766);
    ascify_test_runtime_status = ACL_SUCCESS;
  }
  const int waits = ascify_test_sync_calls;
  assert(ascify::cudaMemcpy(destination, source, sizeof(source), ACL_MEMCPY_HOST_TO_HOST) == ACL_SUCCESS);
  assert(ascify_test_sync_calls == waits);
  const int copies = ascify_test_memcpy_calls;
  const int queued = ascify_test_memcpy_async_calls;
  for (aclrtMemcpyKind kind : {ACL_MEMCPY_HOST_TO_DEVICE, ACL_MEMCPY_DEVICE_TO_HOST,
                              ACL_MEMCPY_DEVICE_TO_DEVICE, ACL_MEMCPY_HOST_TO_HOST}) {
    assert(ascify::cudaMemcpy(nullptr, nullptr, 0, kind) == ACL_SUCCESS);
    assert(ascify::cudaMemcpy(nullptr, source, 1, kind) == ACL_ERROR_RT_PARAM_INVALID);
    assert(ascify::cudaMemcpy(destination, nullptr, 1, kind) == ACL_ERROR_RT_PARAM_INVALID);
  }
  assert(ascify_test_sync_calls == waits && ascify_test_memcpy_calls == copies);
  assert(ascify_test_memcpy_async_calls == queued);
}

void testMemsetAndErrorContracts() {
  unsigned char destination[16] = {};
  aclrtStream stream = reinterpret_cast<aclrtStream>(0x5678);
  ascify_test_runtime_status = ACL_SUCCESS;
  resetRecordedCall();

  assert(ascify::cudaMemset(destination, 0x5a, 9) == ACL_SUCCESS);
  assert(ascify_test_memset_calls == 1);
  assert(ascify_test_destination == destination);
  assert(ascify_test_destination_max == 9);
  assert(ascify_test_count == 9);
  assert(ascify_test_value == 0x5a);

  assert(ascify::cudaMemsetAsync(destination, 3, 11, stream) == ACL_SUCCESS);
  assert(ascify_test_memset_async_calls == 1);
  assert(ascify_test_destination_max == 11);
  assert(ascify_test_count == 11);
  assert(ascify_test_value == 3);
  assert(ascify_test_stream == stream);

  const int sync_calls = ascify_test_memset_calls;
  assert(ascify::cudaMemset(nullptr, 0, 1) == ACL_ERROR_RT_PARAM_INVALID);
  assert(ascify_test_memset_calls == sync_calls);
  assert(ascify::cudaMemset(nullptr, 0, 0) == ACL_SUCCESS);
  assert(ascify_test_memset_calls == sync_calls);
  const int async_calls = ascify_test_memset_async_calls;
  assert(ascify::cudaMemsetAsync(nullptr, 0, 0, stream) == ACL_SUCCESS);
  assert(ascify_test_memset_async_calls == async_calls);

  ascify_test_last_error = ACL_ERROR_RT_MEMORY_ALLOCATION;
  assert(ascify::cudaGetLastError() == ACL_ERROR_RT_MEMORY_ALLOCATION);
  assert(std::strcmp(ascify::cudaGetErrorString(ACL_SUCCESS),
                     "ACL_SUCCESS") == 0);
  assert(std::strcmp(ascify::cudaGetErrorString(
                         ACL_ERROR_RT_MEMORY_ALLOCATION),
                     "recorded ACL failure") == 0);
  ascify_test_recent_error_message = nullptr;
  assert(std::strcmp(ascify::cudaGetErrorString(
                         ACL_ERROR_RT_MEMORY_ALLOCATION),
                     "ACL runtime error") == 0);
}

void testMappedHostContracts() {
  alignas(int) unsigned char allocation[64] = {};
  ascify_test_allocation_result = allocation;
  ascify_test_runtime_status = ACL_SUCCESS;

  // Default and Portable allocations stay pinned-only: no registration.
  void* pinned = nullptr;
  assert(ascify::cudaHostAlloc(&pinned, 32, ascify::cudaHostAllocPortable) ==
         ACL_SUCCESS);
  assert(pinned == allocation);
  assert(ascify_test_host_register_calls == 0);
  assert(ascify::cudaFreeHost(pinned) == ACL_SUCCESS);
  assert(ascify_test_host_unregister_calls == 0);

  // Mapped allocation registers the range; FreeHost unregisters it first.
  void* mapped = nullptr;
  assert(ascify::cudaHostAlloc(&mapped, 64, ascify::cudaHostAllocMapped) ==
         ACL_SUCCESS);
  assert(mapped == allocation);
  assert(ascify_test_host_register_calls == 1);
  assert(ascify_test_host_register_type == ACL_HOST_REGISTER_MAPPED);
  void* device_alias = nullptr;
  assert(ascify::cudaHostGetDevicePointer(&device_alias, mapped, 0) ==
         ACL_SUCCESS);
  assert(device_alias == mapped);
  assert(ascify_test_host_get_device_pointer_calls == 1);
  assert(ascify::cudaHostGetDevicePointer(&device_alias, mapped, 1) ==
         ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(ascify::cudaFreeHost(mapped) == ACL_SUCCESS);
  assert(ascify_test_host_unregister_calls == 1);
  assert(ascify_test_host_unregister_pointer == mapped);

  // Explicit registration ledger, unmapped lookup passthrough, and flags.
  int buffer[16] = {};
  void* alias = nullptr;
  assert(ascify::cudaHostGetDevicePointer(&alias, buffer, 0) == ACL_SUCCESS);
  assert(alias == buffer);
  assert(ascify::cudaHostRegister(buffer, sizeof(buffer),
                                  ascify::cudaHostRegisterMapped) ==
         ACL_SUCCESS);
  assert(ascify_test_host_register_calls == 2);
  assert(ascify::cudaHostRegister(buffer, sizeof(buffer), 0xffffffffu) ==
         ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(ascify::cudaHostUnregister(buffer) == ACL_SUCCESS);
  assert(ascify_test_host_unregister_pointer == buffer);

  // WriteCombined and unknown host-alloc flags stay unsupported; null and
  // zero-size arguments fail closed without reaching the runtime.
  const int registers = ascify_test_host_register_calls;
  void* hostile = reinterpret_cast<void*>(1);
  assert(ascify::cudaHostAlloc(&hostile, 32,
                               ascify::cudaHostAllocWriteCombined) ==
         ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(hostile == nullptr);
  assert(ascify::cudaHostAlloc(&hostile, 32, 0xffffffffu) ==
         ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(ascify_test_host_register_calls == registers);
  assert(ascify::cudaHostAlloc(static_cast<void**>(nullptr), 1, 0) ==
         ACL_ERROR_RT_PARAM_INVALID);
  assert(ascify::cudaHostRegister(nullptr, 8, 0) ==
         ACL_ERROR_RT_PARAM_INVALID);
  assert(ascify::cudaHostRegister(buffer, 0, 0) == ACL_ERROR_RT_PARAM_INVALID);

  // Device flags: schedule hints and MapHost are accepted epoch-wide; any
  // unknown bit is refused instead of being silently reinterpreted.
  assert(ascify::cudaSetDeviceFlags(ascify::cudaDeviceMapHost) == ACL_SUCCESS);
  assert(ascify::cudaSetDeviceFlags(ascify::cudaDeviceBlockingSync |
                                    ascify::cudaDeviceScheduleSpin) ==
         ACL_SUCCESS);
  assert(ascify::cudaSetDeviceFlags(ascify::cudaDeviceScheduleAuto) ==
         ACL_SUCCESS);
  assert(ascify::cudaSetDeviceFlags(0x10u) == ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(ascify::cudaSetDeviceFlags(0xffffffffu) ==
         ACL_ERROR_FEATURE_UNSUPPORTED);
}

void testDriverContracts() {
  // Module image: a minimal synthetic ELF64 header whose section-header
  // table ends at a known offset (the loader derives the image size).
  unsigned char elf[64 + 2 * 64];
  memset(elf, 0, sizeof(elf));
  elf[0] = 0x7f; elf[1] = 'E'; elf[2] = 'L'; elf[3] = 'F';
  elf[4] = 2;  // ELFCLASS64
  const uint64_t shoff = 64;
  const uint16_t shentsize = 64, shnum = 2;
  memcpy(elf + 0x28, &shoff, 8);
  memcpy(elf + 0x3a, &shentsize, 2);
  memcpy(elf + 0x3c, &shnum, 2);

  ascify_test_binary_status = ACL_SUCCESS;
  ascify::CUmodule module = nullptr;
  assert(ascify::cuModuleLoadData(&module, elf) == ACL_SUCCESS);
  assert(module != nullptr);
  assert(ascify_test_binary_load_calls == 1);

  // Registry: one synthetic entry with a matching mangled name cross-check.
  static bool launched = false;
  struct Local {
    static void thunk(unsigned, unsigned, unsigned, unsigned, unsigned,
                      unsigned, unsigned, aclrtStream, void**) { launched = true; }
  };
  static const ascify::drv::KernelEntry entry = {"VecAdd_kernel",
                                                 "_Z13VecAdd_kernelPKfS0_Pfi",
                                                 &Local::thunk};
  assert(ascify::drv::registerKernel(&entry));
  assert(ascify::drv::findKernel("VecAdd_kernel") == &entry);
  assert(ascify::drv::findKernel("missing") == nullptr);

  ascify::CUfunction function = nullptr;
  assert(ascify::cuModuleGetFunction(&function, module, "VecAdd_kernel") ==
         ACL_SUCCESS);
  assert(function == &entry);
  assert(ascify_test_binary_get_function_calls == 1);
  ascify_test_get_function_name = nullptr;
  function = nullptr;
  assert(ascify::cuModuleGetFunction(&function, module, "nope") ==
         ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(function == nullptr);

  // Launch dispatches through the typed thunk with validated geometry.
  int one = 1; void* args[] = {&one};
  assert(ascify::cuLaunchKernel(&entry, 1, 1, 1, 64, 1, 1, 0, nullptr, args,
                                nullptr) == ACL_SUCCESS);
  assert(launched);
  assert(ascify::cuLaunchKernel(nullptr, 1, 1, 1, 64, 1, 1, 0, nullptr, args,
                                nullptr) == ACL_ERROR_RT_PARAM_INVALID);
  assert(ascify::cuLaunchKernel(&entry, 0, 1, 1, 64, 1, 1, 0, nullptr, args,
                                nullptr) == ACL_ERROR_RT_PARAM_INVALID);

  // Device surface: count/ordinal/name and the epoch attribute contract.
  int count = 0;
  assert(ascify::cuDeviceGetCount(&count) == ACL_SUCCESS && count == 1);
  ascify::CUdevice device = -1;
  assert(ascify::cuDeviceGet(&device, 0) == ACL_SUCCESS && device == 0);
  assert(ascify::cuDeviceGet(&device, 5) == ACL_ERROR_RT_INVALID_DEVICEID);
  char name[100] = {};
  assert(ascify::cuDeviceGetName(name, sizeof(name), 0) == ACL_SUCCESS);
  int value = 0;
  assert(ascify::cuDeviceGetAttribute(
             &value, ascify::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0) ==
         ACL_SUCCESS);
  assert(value == 7);
  assert(ascify::cuDeviceGetAttribute(
             &value, ascify::CU_DEVICE_ATTRIBUTE_INTEGRATED, 0) ==
         ACL_SUCCESS);
  assert(value == 0);
  assert(ascify::cuDeviceGetAttribute(&value, 999, 0) ==
         ACL_ERROR_FEATURE_UNSUPPORTED);

  // Context epoch binding and error-string surface.
  ascify::CUcontext context = nullptr;
  assert(ascify::cuCtxCreate(&context, nullptr, 0, 0) == ACL_SUCCESS);
  assert(context != nullptr);
  assert(ascify::cuCtxDestroy(context) == ACL_SUCCESS);
  const char* text = nullptr;
  assert(ascify::cuGetErrorString(ACL_SUCCESS, &text) == ACL_SUCCESS);
  assert(text != nullptr);

  // Non-ELF images are refused without touching the loader.
  const int loads = ascify_test_binary_load_calls;
  ascify::CUmodule hostile = nullptr;
  assert(ascify::cuModuleLoadData(&hostile, "not-an-elf") ==
         ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(hostile == nullptr);
  assert(ascify_test_binary_load_calls == loads);
  assert(ascify::cuModuleUnload(module) == ACL_SUCCESS);
}

}  // namespace

int main() {
  testAllocationContracts();
  testMappedHostContracts();
  testDriverContracts();
  testCopyContracts();
  testDefaultStreamCopyOrdering();
  testMemsetAndErrorContracts();
  return 0;
}
