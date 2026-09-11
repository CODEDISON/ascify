#define __aicore__
#define ASCIFY_TEST_ACL_CONTROLLABLE_DEVICE_QUERY
#include <ascify/ascify_cuda_compat.hpp>
#include <cassert>
#include <cstring>

aclError ascify_test_get_device_status = ACL_SUCCESS;
aclError ascify_test_vector_core_info_status = ACL_SUCCESS;
aclError ascify_test_max_thread_info_status = ACL_SUCCESS;
int32_t ascify_test_device = 2;
int64_t ascify_test_vector_core_count = 48;
int64_t ascify_test_max_threads_per_core = 1024;
int ascify_test_get_device_calls = 0;
int ascify_test_get_device_info_calls = 0;
const char* ascify_test_soc_name = "Ascend test product";

int main() {
  ascify::cudaDeviceProp properties = {};
  assert(ascify::cudaGetDeviceProperties(&properties, 2) == ACL_SUCCESS);
  assert(properties.multiProcessorCount == 48);
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
  for (int predicate : {0, 1, -2, INT_MIN, INT_MAX}) {
    // Host stubs represent a uniform warp; mixed warps need the DT probe.
    assert(ascify::any_sync(UINT32_MAX, predicate) == (predicate != 0));
    assert(ascify::all_sync(UINT32_MAX, predicate) == (predicate != 0));
  }
}
