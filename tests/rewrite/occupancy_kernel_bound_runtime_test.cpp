// Pure CPU stubs: exercises bounded adapters, not kernel metadata or NPU.
#define __aicore__
#define ASCIFY_TEST_ACL_CONTROLLABLE_DEVICE_QUERY
#define ASCIFY_TEST_ACL_CONTROLLABLE_MEMORY_PROPERTY
#define ASCIFY_TEST_ACL_CONTROLLABLE_OCCUPANCY
#include <ascify/ascify_cuda_compat.hpp>
#include <cassert>
aclError ascify_test_get_device_status = ACL_SUCCESS;
aclError ascify_test_vector_core_info_status = ACL_SUCCESS;
aclError ascify_test_max_thread_info_status = ACL_SUCCESS;
int32_t ascify_test_device = 0;
int64_t ascify_test_vector_core_count = 2;
int64_t ascify_test_max_threads_per_core = 2048;
int ascify_test_get_device_calls = 0;
int ascify_test_get_device_info_calls = 0;
aclError ascify_test_memory_info_status = ACL_SUCCESS;
int64_t ascify_test_total_global_memory = 1LL << 30;
aclError ascify_test_block_limit_status = ACL_SUCCESS;
int64_t ascify_test_block_limit = 2048;
aclError ascify_test_grid_limit_status[3] = {ACL_SUCCESS, ACL_SUCCESS, ACL_SUCCESS};
int64_t ascify_test_grid_limit[3] = {65535, 65535, 65535};
aclError ascify_test_warp_info_status = ACL_SUCCESS;
int64_t ascify_test_warp_size = 32;
aclError ascify_test_ub_status = ACL_SUCCESS;
int64_t ascify_test_ub_bytes = 65536;
int64_t ascify_test_grid_blocks = 4096;
int main() {
  int token_storage = 0;
  void *token = &token_storage; // Only a CPU fixture token, no frontend proof.
  int grid = -1, block = -2, user_limit = 1000000;
  assert(ascify::cudaOccupancyMaxPotentialBlockSizeMaxThreads<1024>(
      &grid, &block, token, size_t(0), user_limit) == ACL_SUCCESS);
  assert(block == 1024 && grid == 4 && user_limit == 1000000);
  assert(ascify::cudaOccupancyMaxPotentialBlockSizeMaxThreads<512>(
      &grid, &block, token, size_t(0), 0) == ACL_SUCCESS);
  assert(block == 512 && grid == 8);
  assert(ascify::cudaOccupancyMaxPotentialBlockSizeMaxThreads<1024>(
      &grid, &block, token, 0, 128) == ACL_SUCCESS); // int0 uses fixed-UB overload.
  assert(block == 128 && grid == 32);
  grid = -1; block = -2;
  assert(ascify::cudaOccupancyMaxPotentialBlockSizeMaxThreads<1024>(
      &grid, &block, token, 0, 16) == ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(grid == -1 && block == -2);
  int active = -3, requested = 2048;
  int queries = ascify_test_get_device_info_calls;
  assert(ascify::cudaOccupancyMaxActiveBlocksPerMultiprocessorMaxThreads<1024>(
      &active, token, requested, 0) == ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(active == -3 && requested == 2048 && ascify_test_get_device_info_calls == queries);
  assert(ascify::cudaOccupancyMaxActiveBlocksPerMultiprocessorMaxThreads<1024>(
      &active, token, 32, 0) == ACL_SUCCESS && active == 64);
  assert(ascify::cudaOccupancyMaxActiveBlocksPerMultiprocessorMaxThreads<1024>(
      &active, token, 1024, 0) == ACL_SUCCESS && active == 2);
  ascify_test_block_limit = 512; active = -3;
  assert(ascify::cudaOccupancyMaxActiveBlocksPerMultiprocessorMaxThreads<1024>(
      &active, token, 1024, 0) == ACL_ERROR_FEATURE_UNSUPPORTED && active == -3);
  ascify_test_block_limit = 2048;
  ascify_test_block_limit_status = ACL_ERROR_RT_PARAM_INVALID;
  assert(ascify::cudaOccupancyMaxActiveBlocksPerMultiprocessorMaxThreads<1024>(
      &active, token, 32, 0) == ACL_ERROR_RT_PARAM_INVALID && active == -3);
  ascify_test_block_limit_status = ACL_SUCCESS;
  ascify_test_warp_info_status = ACL_ERROR_RT_PARAM_INVALID;
  assert(ascify::cudaOccupancyMaxPotentialBlockSizeMaxThreads<1024>(
      &grid, &block, token, 0, 0) == ACL_ERROR_RT_PARAM_INVALID);
  assert(grid == -1 && block == -2);
  ascify_test_warp_info_status = ACL_SUCCESS;
  int candidates[64] = {}, calls = 0;
  auto model = [&](int candidate) -> size_t {
    assert(calls < 64); candidates[calls++] = candidate;
    return candidate > 256 ? size_t(131072) : size_t(65536);
  };
  assert(ascify::cudaOccupancyMaxPotentialBlockSizeMaxThreads<1024>(
      &grid, &block, token, model, 0) == ACL_SUCCESS);
  assert(block == 256 && calls == 25);
  for (int i = 0; i < calls; ++i) assert(candidates[i] == 1024 - 32 * i);
  ascify::cudaDeviceProp properties{};
  assert(ascify::cudaGetDeviceProperties(&properties, 0) == ACL_SUCCESS);
  assert(properties.maxThreadsPerBlock == 2048 && properties.maxThreadsPerMultiProcessor == 2048);
}
