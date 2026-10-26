#define __aicore__
#define ASCIFY_TEST_ACL_CONTROLLABLE_RUNTIME_LIFECYCLE
#define ASCIFY_TEST_ACL_THREAD_LOCAL_DEVICE
#define ASCIFY_TEST_ACL_CONTROLLABLE_STREAM_WAIT_QUERY
#include <ascify/ascify_cuda_compat.hpp>
#include <cassert>
#include <cstring>
#include <thread>
#include <type_traits>

static_assert(std::is_same<decltype(&ascify::cudaStreamQuery),
                          aclError (*)(aclrtStream)>::value, "CUDA query ABI");
static_assert(std::is_same<decltype(&ascify::cudaStreamWaitEvent),
                          aclError (*)(aclrtStream, aclrtEvent, unsigned int)>::value,
              "CUDA wait ABI");
static_assert(ascify::cudaEventWaitDefault == 0U, "CUDA default wait flag");
static_assert(ascify::cudaEventWaitExternal == 1U, "CUDA external wait flag");

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
int ascify_test_synchronize_stream_calls = 0;
int ascify_test_synchronize_event_calls = 0;
aclrtStream ascify_test_stream_create_result = nullptr;

namespace {
struct Stream { int32_t device; };
aclError backend_status = ACL_SUCCESS;
aclrtStreamStatus backend_state = ACL_STREAM_STATUS_COMPLETE;
int query_calls = 0, wait_calls = 0;
aclrtStream seen_stream = nullptr;
aclrtEvent seen_event = nullptr;
int32_t seen_device = -1;
}

aclError aclrtStreamQuery(aclrtStream stream, aclrtStreamStatus* state) {
  ++query_calls;
  assert(ascify_test_acl_initialized && ascify_test_context_bound);
  seen_stream = stream;
  seen_device = ascify_test_current_device;
  assert(stream != nullptr);
  if (static_cast<Stream*>(stream)->device != seen_device) {
    return ACL_ERROR_RT_CONTEXT_NULL;
  }
  // A backend may write an output even when failing; the wrapper must use
  // the aclError result before interpreting this state.
  *state = backend_state;
  return backend_status;
}
aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event) {
  ++wait_calls;
  assert(ascify_test_acl_initialized && ascify_test_context_bound);
  seen_stream = stream;
  seen_event = event;
  seen_device = ascify_test_current_device;
  assert(stream != nullptr && event != nullptr);
  if (static_cast<Stream*>(stream)->device != seen_device) {
    return ACL_ERROR_RT_CONTEXT_NULL;
  }
  return backend_status;
}

int main(int argc, char** argv) {
  assert(argc == 2);
  const char* mode = argv[1];
  Stream stream{0};
  int event = 17;
  if (std::strcmp(mode, "invalid") == 0) {
    const aclrtStream implicit_streams[] = {nullptr, ascify::cudaStreamLegacy,
                                           ascify::cudaStreamPerThread};
    for (aclrtStream special : implicit_streams) {
      assert(ascify::cudaStreamQuery(special) == ACL_ERROR_FEATURE_UNSUPPORTED);
      assert(ascify::cudaStreamWaitEvent(special, &event) == ACL_ERROR_FEATURE_UNSUPPORTED);
    }
    assert(ascify::cudaStreamWaitEvent(&stream, nullptr) == ACL_ERROR_RT_PARAM_INVALID);
    assert(ascify::cudaStreamWaitEvent(&stream, &event, ascify::cudaEventWaitExternal) ==
           ACL_ERROR_FEATURE_UNSUPPORTED);
    assert(ascify::cudaStreamWaitEvent(&stream, &event, 2U) == ACL_ERROR_RT_PARAM_INVALID);
    assert(ascify::cudaStreamWaitEvent(&stream, &event, ~0U) == ACL_ERROR_RT_PARAM_INVALID);
    assert(ascify_test_init_calls == 0 && query_calls == 0 && wait_calls == 0);
    assert(ascify_test_get_device_calls == 0 && ascify_test_set_device_calls == 0);
  } else if (std::strcmp(mode, "query") == 0) {
    assert(ascify::cudaStreamQuery(&stream) == ACL_SUCCESS);
    assert(seen_stream == &stream && seen_device == 0 && query_calls == 1);
    backend_state = ACL_STREAM_STATUS_NOT_READY;
    assert(ascify::cudaStreamQuery(&stream) == ACL_ERROR_RT_EVENT_NOT_COMPLETE);
    backend_state = ACL_STREAM_STATUS_RESERVED;
    assert(ascify::cudaStreamQuery(&stream) == ACL_ERROR_FEATURE_UNSUPPORTED);
    backend_state = static_cast<aclrtStreamStatus>(67);
    assert(ascify::cudaStreamQuery(&stream) == ACL_ERROR_FEATURE_UNSUPPORTED);
    backend_status = ACL_ERROR_RT_MEMORY_ALLOCATION;
    backend_state = ACL_STREAM_STATUS_COMPLETE;
    assert(ascify::cudaStreamQuery(&stream) == backend_status);
    assert(query_calls == 5 && wait_calls == 0);
    backend_status = ACL_ERROR_RT_STREAM_NOT_COMPLETE;
    assert(ascify::cudaStreamQuery(&stream) == ACL_ERROR_RT_EVENT_NOT_COMPLETE);
    backend_status = ACL_SUCCESS;
    // Query result translation must not consume an earlier adapter error.
    ascify::detail::rememberLifecycleError(ACL_ERROR_RT_INVALID_DEVICEID);
    backend_state = ACL_STREAM_STATUS_NOT_READY;
    assert(ascify::cudaStreamQuery(&stream) == ACL_ERROR_RT_EVENT_NOT_COMPLETE);
    backend_state = ACL_STREAM_STATUS_COMPLETE;
    assert(ascify::cudaStreamQuery(&stream) == ACL_SUCCESS);
    assert(ascify::cudaPeekAtLastError() == ACL_ERROR_RT_INVALID_DEVICEID);
    assert(ascify::cudaGetLastError() == ACL_ERROR_RT_INVALID_DEVICEID);
  } else if (std::strcmp(mode, "wait") == 0) {
    unsigned int flags = 0;
    assert(ascify::cudaStreamWaitEvent(&stream, &event, flags++) == ACL_SUCCESS);
    assert(flags == 1 && wait_calls == 1 && query_calls == 0);
    assert(seen_stream == &stream && seen_event == &event && event == 17);
    backend_status = ACL_ERROR_RT_MEMORY_ALLOCATION;
    assert(ascify::cudaStreamWaitEvent(&stream, &event) == backend_status);
    assert(wait_calls == 2);
  } else if (std::strcmp(mode, "lazy") == 0) {
    ascify_test_stream_create_result = &stream;
    aclrtStream created = nullptr;
    assert(ascify::cudaStreamCreate(&created) == ACL_SUCCESS);
    assert(created == &stream && ascify_test_create_stream_calls == 1);
    assert(ascify::cudaStreamQuery(created) == ACL_SUCCESS);
    assert(ascify::cudaStreamWaitEvent(created, &event) == ACL_SUCCESS);
    assert(ascify_test_init_calls == 1 && ascify_test_set_device_calls == 1);
    assert(ascify::cudaSetDevice(3) == ACL_SUCCESS);
    assert(ascify::cudaDeviceReset() == ACL_SUCCESS);
    Stream after_reset{3};
    ascify_test_stream_create_result = &after_reset;
    assert(ascify::cudaStreamCreate(&created) == ACL_SUCCESS);
    assert(ascify::cudaStreamQuery(created) == ACL_SUCCESS);
    assert(ascify::cudaStreamWaitEvent(created, &event) == ACL_SUCCESS);
    assert(seen_device == 3 && seen_stream == &after_reset);
  } else if (std::strcmp(mode, "borrowed") == 0) {
    ascify_test_init_status = ACL_ERROR_REPEAT_INITIALIZE;
    ascify_test_acl_initialized = true;
    ascify_test_context_bound = true;
    ascify_test_current_device = 4;
    stream.device = 4;
    assert(ascify::cudaStreamQuery(&stream) == ACL_SUCCESS);
    assert(ascify::cudaStreamWaitEvent(&stream, &event) == ACL_SUCCESS);
    assert(seen_device == 4 && ascify_test_set_device_calls == 0);
  } else if (std::strcmp(mode, "wrong-context") == 0) {
    assert(ascify::cudaSetDevice(3) == ACL_SUCCESS);
    stream.device = 4;
    assert(ascify::cudaStreamQuery(&stream) == ACL_ERROR_RT_CONTEXT_NULL);
    assert(ascify::cudaStreamWaitEvent(&stream, &event) == ACL_ERROR_RT_CONTEXT_NULL);
    assert(seen_device == 3 && ascify_test_current_device == 3);
    assert(ascify_test_set_device_calls == 1 && query_calls == 1 && wait_calls == 1);
  } else if (std::strcmp(mode, "init-failure") == 0) {
    ascify_test_init_status = ACL_ERROR_FEATURE_UNSUPPORTED;
    assert(ascify::cudaStreamQuery(&stream) == ascify_test_init_status);
    assert(ascify::cudaStreamWaitEvent(&stream, &event) == ascify_test_init_status);
    assert(ascify_test_init_calls == 1 && ascify_test_set_device_calls == 0);
    assert(query_calls == 0 && wait_calls == 0);
  } else if (std::strcmp(mode, "bind-failure") == 0) {
    ascify_test_set_device_status = ACL_ERROR_RT_CONTEXT_NULL;
    assert(ascify::cudaStreamQuery(&stream) == ascify_test_set_device_status);
    assert(ascify::cudaStreamWaitEvent(&stream, &event) == ascify_test_set_device_status);
    assert(ascify_test_init_calls == 1 && ascify_test_set_device_calls == 2);
    assert(query_calls == 0 && wait_calls == 0);
    assert(ascify::cudaGetLastError() == ascify_test_set_device_status);
  } else {
    assert(std::strcmp(mode, "threads") == 0);
    assert(ascify::cudaSetDevice(3) == ACL_SUCCESS);
    std::thread selected([] {
      assert(ascify::cudaSetDevice(5) == ACL_SUCCESS);
      Stream worker{5}; int worker_event = 7;
      assert(ascify::cudaStreamQuery(&worker) == ACL_SUCCESS);
      assert(ascify::cudaStreamWaitEvent(&worker, &worker_event) == ACL_SUCCESS);
      assert(seen_device == 5);
    });
    selected.join();
    assert(ascify_test_current_device == 3 && ascify_test_context_bound);
    std::thread fresh([] {
      Stream worker{0}; int worker_event = 7;
      assert(ascify::cudaStreamQuery(&worker) == ACL_SUCCESS);
      assert(ascify::cudaStreamWaitEvent(&worker, &worker_event) == ACL_SUCCESS);
      assert(seen_device == 0);
    });
    fresh.join();
    stream.device = 3;
    assert(ascify::cudaStreamQuery(&stream) == ACL_SUCCESS);
    assert(seen_device == 3 && ascify_test_current_device == 3);
  }
  // Neither API may turn polling or device waits into host synchronization,
  // create streams, or look up a default stream behind the caller's back.
  assert(ascify_test_synchronize_device_calls == 0);
  assert(ascify_test_synchronize_stream_calls == 0);
  assert(ascify_test_synchronize_event_calls == 0);
  assert(ascify_test_get_default_stream_calls == 0);
  assert(ascify_test_record_event_calls == 0);
  assert(ascify_test_create_stream_calls == (std::strcmp(mode, "lazy") == 0 ? 2 : 0));
  ascify::detail::runtime_manager.shutdown();
  if (ascify_test_init_calls != 0 && ascify_test_init_status == ACL_SUCCESS) {
    assert(ascify_test_finalize_calls == 1);
  } else {
    assert(ascify_test_finalize_calls == 0);
    assert(ascify_test_reset_device_calls == 0);
  }
}
