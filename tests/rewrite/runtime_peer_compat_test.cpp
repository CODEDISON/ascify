#define __aicore__
#define ASCIFY_TEST_ACL_CONTROLLABLE_RUNTIME_LIFECYCLE
#define ASCIFY_TEST_ACL_THREAD_LOCAL_DEVICE
#include <ascify/ascify_cuda_compat.hpp>
#include <cassert>
#include <cstring>
#include <thread>
#include <type_traits>

static_assert(std::is_same<decltype(&ascify::cudaDeviceCanAccessPeer),
                          aclError (*)(int*, int, int)>::value, "CUDA query ABI");
static_assert(std::is_same<decltype(&ascify::cudaDeviceEnablePeerAccess),
                          aclError (*)(int, unsigned int)>::value, "CUDA enable ABI");
static_assert(std::is_same<decltype(&ascify::cudaDeviceDisablePeerAccess),
                          aclError (*)(int)>::value, "CUDA disable ABI");

aclError ascify_test_init_status = ACL_SUCCESS;
aclError ascify_test_set_device_status = ACL_SUCCESS;
aclError ascify_test_reset_device_status = ACL_SUCCESS;
aclError ascify_test_device_count_status = ACL_SUCCESS;
aclError ascify_test_synchronize_status = ACL_SUCCESS;
aclError ascify_test_finalize_status = ACL_SUCCESS;
bool ascify_test_acl_initialized = false;
thread_local bool ascify_test_context_bound = false;
thread_local int32_t ascify_test_current_device = -1;
uint32_t ascify_test_device_count = 8;
int ascify_test_init_calls = 0;
int ascify_test_finalize_calls = 0;
int ascify_test_set_device_calls = 0;
int ascify_test_get_device_calls = 0;
int ascify_test_reset_device_calls = 0;
int ascify_test_get_device_count_calls = 0;
int ascify_test_synchronize_device_calls = 0;
int ascify_test_get_default_stream_calls = 0;
int ascify_test_create_stream_calls = 0;
int ascify_test_record_event_calls = 0;
aclrtStream ascify_test_default_stream = nullptr;
aclrtStream ascify_test_recorded_event_stream = nullptr;
int32_t ascify_test_reset_devices[80] = {};

namespace {
aclError peer_status = ACL_SUCCESS;
int32_t support = 0;
int query_calls = 0, enable_calls = 0, disable_calls = 0;
int32_t seen_device = -1, seen_peer = -1;
uint32_t seen_flags = 99;
}

aclError aclrtDeviceCanAccessPeer(int32_t* value, int32_t device,
                                int32_t peer) {
  ++query_calls;
  assert(ascify_test_acl_initialized);
  seen_device = device;
  seen_peer = peer;
  // Deliberately write even on failure: the shim must not publish it then.
  *value = support;
  return peer_status;
}
aclError aclrtDeviceEnablePeerAccess(int32_t peer, uint32_t flags) {
  ++enable_calls;
  assert(ascify_test_acl_initialized && ascify_test_context_bound);
  seen_device = ascify_test_current_device;
  seen_peer = peer;
  seen_flags = flags;
  return peer_status;
}
aclError aclrtDeviceDisablePeerAccess(int32_t peer) {
  ++disable_calls;
  assert(ascify_test_acl_initialized && ascify_test_context_bound);
  seen_device = ascify_test_current_device;
  seen_peer = peer;
  return peer_status;
}

int main(int argc, char** argv) {
  assert(argc == 2);
  const char* mode = argv[1];
  if (std::strcmp(mode, "invalid") == 0) {
    int value = 73;
    assert(ascify::cudaDeviceCanAccessPeer(nullptr, 0, 1) == ACL_ERROR_RT_PARAM_INVALID);
    assert(ascify::cudaDeviceCanAccessPeer(&value, -1, 1) == ACL_ERROR_RT_PARAM_INVALID);
    assert(ascify::cudaDeviceCanAccessPeer(&value, 0, -1) == ACL_ERROR_RT_PARAM_INVALID);
    assert(ascify::cudaDeviceEnablePeerAccess(-1) == ACL_ERROR_RT_PARAM_INVALID);
    assert(ascify::cudaDeviceEnablePeerAccess(1, 1) == ACL_ERROR_RT_PARAM_INVALID);
    assert(ascify::cudaDeviceEnablePeerAccess(1, ~0U) == ACL_ERROR_RT_PARAM_INVALID);
    assert(ascify::cudaDeviceDisablePeerAccess(-1) == ACL_ERROR_RT_PARAM_INVALID);
    assert(value == 73 && ascify_test_init_calls == 0);
    assert(query_calls == 0 && enable_calls == 0 && disable_calls == 0);
  } else if (std::strcmp(mode, "query") == 0) {
    int value = 73;
    assert(ascify::cudaDeviceCanAccessPeer(&value, 3, 6) == ACL_SUCCESS);
    assert(value == 0 && seen_device == 3 && seen_peer == 6);
    support = 1;
    assert(ascify::cudaDeviceCanAccessPeer(&value, 6, 3) == ACL_SUCCESS);
    assert(value == 1 && seen_device == 6 && seen_peer == 3);
    peer_status = ACL_ERROR_FEATURE_UNSUPPORTED;
    support = 99;
    assert(ascify::cudaDeviceCanAccessPeer(&value, 6, 3) == peer_status);
    assert(value == 1 && query_calls == 3);
    assert(ascify::cudaPeekAtLastError() == peer_status);
    assert(ascify::cudaGetLastError() == peer_status);
    assert(ascify::cudaPeekAtLastError() == ACL_SUCCESS);
    assert(ascify_test_init_calls == 1);
    assert(ascify_test_get_device_calls == 0 && ascify_test_set_device_calls == 0);
  } else if (std::strcmp(mode, "lazy") == 0) {
    assert(ascify::cudaDeviceEnablePeerAccess(2) == ACL_SUCCESS);
    assert(seen_device == 0 && seen_peer == 2 && seen_flags == 0);
    assert(ascify_test_init_calls == 1 && ascify_test_set_device_calls == 1);
    assert(ascify::cudaDeviceDisablePeerAccess(2) == ACL_SUCCESS);
    assert(disable_calls == 1 && seen_device == 0 && seen_peer == 2);
    assert(ascify::cudaSetDevice(3) == ACL_SUCCESS);
    assert(ascify::cudaDeviceReset() == ACL_SUCCESS);
    assert(!ascify_test_context_bound);
    assert(ascify::cudaDeviceEnablePeerAccess(4, 0) == ACL_SUCCESS);
    assert(seen_device == 3 && seen_peer == 4);
  } else if (std::strcmp(mode, "borrowed") == 0) {
    ascify_test_init_status = ACL_ERROR_REPEAT_INITIALIZE;
    ascify_test_acl_initialized = true;
    ascify_test_context_bound = true;
    ascify_test_current_device = 4;
    int value = 73;
    assert(ascify::cudaDeviceCanAccessPeer(&value, 2, 6) == ACL_SUCCESS);
    assert(ascify_test_current_device == 4 && ascify_test_get_device_calls == 0);
    assert(ascify::cudaDeviceEnablePeerAccess(2) == ACL_SUCCESS);
    assert(seen_device == 4 && seen_peer == 2);
    assert(ascify::cudaDeviceDisablePeerAccess(2) == ACL_SUCCESS);
    assert(ascify_test_set_device_calls == 0);
    ascify::detail::runtime_manager.shutdown();
    assert(ascify_test_reset_device_calls == 0 && ascify_test_finalize_calls == 0);
    return 0;
  } else if (std::strcmp(mode, "errors") == 0) {
    assert(ascify::cudaSetDevice(3) == ACL_SUCCESS);
    peer_status = ACL_ERROR_FEATURE_UNSUPPORTED;
    assert(ascify::cudaDeviceEnablePeerAccess(2) == peer_status);
    assert(ascify::cudaPeekAtLastError() == peer_status);
    assert(ascify::cudaGetLastError() == peer_status);
    peer_status = ACL_ERROR_RT_INVALID_DEVICEID;
    assert(ascify::cudaDeviceDisablePeerAccess(7) == peer_status);
    assert(ascify::cudaGetLastError() == peer_status);
    assert(seen_device == 3 && seen_peer == 7);
    assert(enable_calls == 1 && disable_calls == 1);
    assert(ascify_test_set_device_calls == 1);
  } else if (std::strcmp(mode, "init-failure") == 0) {
    ascify_test_init_status = ACL_ERROR_FEATURE_UNSUPPORTED;
    int value = 73;
    assert(ascify::cudaDeviceCanAccessPeer(&value, 0, 1) == ascify_test_init_status);
    assert(ascify::cudaDeviceEnablePeerAccess(1) == ascify_test_init_status);
    assert(ascify::cudaDeviceDisablePeerAccess(1) == ascify_test_init_status);
    assert(value == 73 && ascify_test_init_calls == 1);
    assert(ascify_test_set_device_calls == 0);
    assert(query_calls == 0 && enable_calls == 0 && disable_calls == 0);
  } else {
    assert(std::strcmp(mode, "threads") == 0);
    assert(ascify::cudaSetDevice(3) == ACL_SUCCESS);
    std::thread selected([] {
      assert(ascify::cudaSetDevice(5) == ACL_SUCCESS);
      assert(ascify::cudaDeviceEnablePeerAccess(1) == ACL_SUCCESS);
      assert(seen_device == 5 && seen_peer == 1);
      assert(ascify::cudaDeviceReset() == ACL_SUCCESS);
      assert(ascify::cudaDeviceDisablePeerAccess(1) == ACL_SUCCESS);
      assert(seen_device == 5);
    });
    selected.join();
    assert(ascify_test_current_device == 3 && ascify_test_context_bound);
    assert(ascify::detail::selectedDeviceForThread() == 3);
    std::thread fresh([] {
      assert(ascify::cudaDeviceEnablePeerAccess(1) == ACL_SUCCESS);
      assert(seen_device == 0 && seen_peer == 1);
    });
    fresh.join();
    assert(ascify_test_current_device == 3);
    assert(ascify::cudaDeviceDisablePeerAccess(2) == ACL_SUCCESS);
    assert(seen_device == 3 && seen_peer == 2);
  }
  ascify::detail::runtime_manager.shutdown();
  if (ascify_test_init_calls != 0 && ascify_test_init_status == ACL_SUCCESS) {
    assert(ascify_test_finalize_calls == 1);
  } else {
    assert(ascify_test_finalize_calls == 0);
  }
}
