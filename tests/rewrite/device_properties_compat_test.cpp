#define __aicore__
#define ASCIFY_TEST_ACL_CONTROLLABLE_DEVICE_QUERY
#define ASCIFY_TEST_ACL_CONTROLLABLE_MEMORY_PROPERTY
#include <ascify/ascify_cuda_compat.hpp>
#include <cassert>
#include <cstring>
#include <initializer_list>

aclError ascify_test_get_device_status = ACL_SUCCESS;
aclError ascify_test_vector_core_info_status = ACL_SUCCESS;
aclError ascify_test_max_thread_info_status = ACL_SUCCESS;
aclError ascify_test_memory_info_status = ACL_SUCCESS;
int64_t ascify_test_total_global_memory = int64_t{1} << 40;
aclError ascify_test_block_limit_status = ACL_SUCCESS;
int64_t ascify_test_block_limit = 2048;
int32_t ascify_test_device = 2;
int64_t ascify_test_vector_core_count = 48;
int64_t ascify_test_max_threads_per_core = 1024;
int ascify_test_get_device_calls = 0;
int ascify_test_get_device_info_calls = 0;
const char* ascify_test_soc_name = "Ascend test product";

void test_kernel(unsigned *) {}

int main() {
  int unavailable = 12345;
  assert(ascify::cudaDeviceGetAttribute(&unavailable,
      ascify::cudaDevAttrComputeCapabilityMajor, 2) == ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(ascify::cudaDeviceGetAttribute(&unavailable,
      ascify::cudaDevAttrComputeCapabilityMinor, 2) == ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(unavailable == 12345 && ascify_test_get_device_info_calls == 0);
  assert(ascify::cudaDeviceGetAttribute(nullptr,
      ascify::cudaDevAttrComputeCapabilityMajor, 2) == ACL_ERROR_RT_PARAM_INVALID);
  assert(ascify::cudaDeviceGetAttribute(&unavailable,
      ascify::cudaDevAttrComputeCapabilityMinor, -1) == ACL_ERROR_RT_PARAM_INVALID);
  assert(unavailable == 12345 && ascify_test_get_device_info_calls == 0);
  ascify::cudaDeviceProp properties = {};
  assert(ascify::cudaGetDeviceProperties(&properties, 2) == ACL_SUCCESS);
  assert(properties.multiProcessorCount == 48);
  assert(properties.totalGlobalMem == (uint64_t{1} << 40));
  assert(std::strcmp(properties.name, ascify_test_soc_name) == 0);
  const ascify::cudaDeviceProp saved = properties;
  assert(ascify::cudaGetDeviceProperties(nullptr, 2) == ACL_ERROR_RT_PARAM_INVALID);
  assert(ascify::cudaGetDeviceProperties(&properties, -1) == ACL_ERROR_RT_PARAM_INVALID);
  assert(ascify::cudaGetDeviceProperties(&properties, 0) == ACL_ERROR_FEATURE_UNSUPPORTED);
  ascify_test_get_device_status = 71;
  assert(ascify::cudaGetDeviceProperties(&properties, 2) == 71);
  ascify_test_get_device_status = ACL_SUCCESS;
  ascify_test_vector_core_info_status = 72;
  assert(ascify::cudaGetDeviceProperties(&properties, 2) == 72);
  ascify_test_vector_core_info_status = ACL_SUCCESS;
  ascify_test_memory_info_status = 77;
  assert(ascify::cudaGetDeviceProperties(&properties, 2) == 77);
  ascify_test_memory_info_status = ACL_SUCCESS;
  for (int64_t memory : {int64_t{0}, int64_t{-1}}) {
    ascify_test_total_global_memory = memory;
    assert(ascify::cudaGetDeviceProperties(&properties, 2) == ACL_ERROR_FEATURE_UNSUPPORTED);
    assert(std::memcmp(&properties, &saved, sizeof(properties)) == 0);
  }
  ascify_test_total_global_memory = int64_t{1} << 40;
  for (int64_t count : {int64_t{0}, int64_t{-1}, int64_t{INT_MAX} + 1}) {
    ascify_test_vector_core_count = count;
    assert(ascify::cudaGetDeviceProperties(&properties, 2) == ACL_ERROR_FEATURE_UNSUPPORTED);
  }
  ascify_test_vector_core_count = 48;
  ascify_test_soc_name = nullptr;
  assert(ascify::cudaGetDeviceProperties(&properties, 2) == ACL_ERROR_FEATURE_UNSUPPORTED);
  ascify_test_soc_name = "";
  assert(ascify::cudaGetDeviceProperties(&properties, 2) == ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(std::memcmp(&properties, &saved, sizeof(properties)) == 0);
  char long_name[400];
  std::memset(long_name, 'N', sizeof(long_name));
  long_name[399] = '\0';
  ascify_test_soc_name = long_name;
  assert(ascify::cudaGetDeviceProperties(&properties, 2) == ACL_SUCCESS);
  assert(std::strlen(properties.name) == 255);
  ascify::cudaFuncAttributes attributes{};
  attributes.sharedSizeBytes = 791;  // The restricted helper never claims UB metadata.
  attributes.maxThreadsPerBlock = 37;
  const auto original_attributes = attributes;
#if defined(ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3)
  assert(ascify::cudaFuncGetAttributesMaxThreads<128>(&attributes, test_kernel) == ACL_SUCCESS);
  assert(attributes.maxThreadsPerBlock == 128 && attributes.sharedSizeBytes == 791);
  assert(ascify::cudaFuncGetAttributesMaxThreads<1024>(&attributes, test_kernel) == ACL_SUCCESS);
  assert(attributes.maxThreadsPerBlock == 1024);
  ascify_test_block_limit = 96;
  assert(ascify::cudaFuncGetAttributesMaxThreads<128>(&attributes, test_kernel) == ACL_SUCCESS);
  assert(attributes.maxThreadsPerBlock == 96);
  attributes = original_attributes;
  ascify_test_block_limit_status = 78;
  assert(ascify::cudaFuncGetAttributesMaxThreads<128>(&attributes, test_kernel) == 78);
  ascify_test_block_limit_status = ACL_SUCCESS;
  for (int64_t limit : {int64_t{0}, int64_t{-1}, int64_t{INT_MAX}+1}) {
    ascify_test_block_limit = limit;
    assert(ascify::cudaFuncGetAttributesMaxThreads<128>(&attributes, test_kernel) == ACL_ERROR_FEATURE_UNSUPPORTED);
    assert(std::memcmp(&attributes, &original_attributes, sizeof(attributes)) == 0);
  }
  ascify_test_block_limit = 2048;
  ascify_test_get_device_status = 79;
  assert(ascify::cudaFuncGetAttributesMaxThreads<128>(&attributes, test_kernel) == 79);
  ascify_test_get_device_status = ACL_SUCCESS;
#else
  assert(ascify::cudaFuncGetAttributesMaxThreads<128>(&attributes, test_kernel) == ACL_ERROR_FEATURE_UNSUPPORTED);
#endif
  assert(ascify::cudaFuncGetAttributesMaxThreads<128>(nullptr, test_kernel) == ACL_ERROR_RT_PARAM_INVALID);
  assert(ascify::cudaFuncGetAttributesMaxThreads<128>(&attributes, static_cast<void(*)(unsigned *)>(nullptr)) == ACL_ERROR_RT_PARAM_INVALID);
  assert(std::memcmp(&attributes, &original_attributes, sizeof(attributes)) == 0);
  for (int predicate : {0, 1, -2, INT_MIN, INT_MAX}) {
    // Host stubs represent a uniform warp; mixed warps need the DT probe.
    assert(ascify::any_sync(UINT32_MAX, predicate) == (predicate != 0));
    assert(ascify::all_sync(UINT32_MAX, predicate) == (predicate != 0));
  }
}
