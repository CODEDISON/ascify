#define __aicore__
#define ASCIFY_TEST_ACL_CONTROLLABLE_RUNTIME_MEMORY

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

namespace {

void resetRecordedCall() {
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

  assert(ascify::cudaMalloc(static_cast<void**>(nullptr), 1) ==
         ACL_ERROR_RT_PARAM_INVALID);
  assert(ascify::cudaMallocHost(static_cast<void**>(nullptr), 1) ==
         ACL_ERROR_RT_PARAM_INVALID);
  assert(ascify_test_malloc_calls == 1);
  assert(ascify_test_malloc_host_calls == 1);

  ascify_test_runtime_status = ACL_ERROR_RT_MEMORY_ALLOCATION;
  device = reinterpret_cast<int*>(1);
  host = reinterpret_cast<int*>(1);
  assert(ascify::cudaMalloc(&device, 32) == ACL_ERROR_RT_MEMORY_ALLOCATION);
  assert(ascify::cudaMallocHost(&host, 32) == ACL_ERROR_RT_MEMORY_ALLOCATION);
  assert(device == nullptr);
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

}  // namespace

int main() {
  testAllocationContracts();
  testCopyContracts();
  testMemsetAndErrorContracts();
  return 0;
}
