#define __aicore__
#define ASCIFY_TEST_ACL_CONTROLLABLE_RUNTIME_LIFECYCLE
#define ASCIFY_TEST_ACL_CONTROLLABLE_RUNTIME_MEMORY
#define ASCIFY_TEST_CONTROLLABLE_EXIT_REGISTRATION
#define ASCIFY_TEST_ACL_LIFECYCLE_EXIT_ORDER

#include <ascify/ascify_cuda_compat.hpp>

#include <cassert>
#include <cstring>

aclError ascify_test_init_status = ACL_SUCCESS;
aclError ascify_test_set_device_status = ACL_SUCCESS;
aclError ascify_test_reset_device_status = ACL_SUCCESS;
aclError ascify_test_device_count_status = ACL_SUCCESS;
aclError ascify_test_synchronize_status = ACL_SUCCESS;
aclError ascify_test_finalize_status = ACL_SUCCESS;
bool ascify_test_acl_initialized = false;
bool ascify_test_context_bound = false;
int32_t ascify_test_current_device = -1;
uint32_t ascify_test_device_count = 7;
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
aclrtStream ascify_test_default_stream = reinterpret_cast<aclrtStream>(0x55);
aclrtStream ascify_test_recorded_event_stream = nullptr;
int32_t ascify_test_reset_devices[80] = {};
int ascify_test_exit_registration_status = 0;
int ascify_test_exit_registration_calls = 0;
int ascify_test_registration_observed_init_calls = 0;
bool ascify_test_registration_observed_initialized = false;
void (*ascify_test_exit_cleanup)() = nullptr;
int ascify_test_exit_registration_fail_at = 0;
bool ascify_test_real_exit = false;
bool ascify_test_sdk_init_alive = false;
bool ascify_test_sdk_lazy_alive[2] = {};
int ascify_test_sdk_lazy_count = 0;
bool ascify_test_second_lazy_phase = false;
int ascify_test_expected_exit_resets = 0;
int ascify_test_expected_exit_finalizes = 1;

void ascify_test_verify_exit_cleanup() {
  assert(ascify_test_reset_device_calls == ascify_test_expected_exit_resets);
  assert(ascify_test_finalize_calls == ascify_test_expected_exit_finalizes);
}

void ascify_test_destroy_sdk_init() {
  ascify_test_verify_exit_cleanup();
  ascify_test_sdk_init_alive = false;
}

void ascify_test_destroy_sdk_lazy_first() {
  ascify_test_verify_exit_cleanup();
  ascify_test_sdk_lazy_alive[0] = false;
}

void ascify_test_destroy_sdk_lazy_second() {
  ascify_test_verify_exit_cleanup();
  ascify_test_sdk_lazy_alive[1] = false;
}

void ascify_test_lifecycle_operation(const char* operation) {
  if (!ascify_test_real_exit) { return; }
  if (std::strcmp(operation, "init") == 0) {
    ascify_test_sdk_init_alive = true;
    assert(std::atexit(ascify_test_destroy_sdk_init) == 0);
  } else if (std::strcmp(operation, "set") == 0) {
    const int phase = ascify_test_sdk_lazy_count;
    if (phase == 0 || (phase == 1 && ascify_test_second_lazy_phase)) {
      ascify_test_sdk_lazy_alive[phase] = true;
      ++ascify_test_sdk_lazy_count;
      assert(std::atexit(phase == 0 ? ascify_test_destroy_sdk_lazy_first :
                                      ascify_test_destroy_sdk_lazy_second) == 0);
    }
  } else {
    assert(ascify_test_sdk_init_alive);
    for (int phase = 0; phase < ascify_test_sdk_lazy_count; ++phase) {
      assert(ascify_test_sdk_lazy_alive[phase]);
    }
  }
}

int ascify_test_register_exit_cleanup(void (*callback)()) {
  ++ascify_test_exit_registration_calls;
  ascify_test_registration_observed_init_calls = ascify_test_init_calls;
  ascify_test_registration_observed_initialized =
      ascify_test_acl_initialized;
  if (ascify_test_exit_registration_calls ==
      ascify_test_exit_registration_fail_at) { return 1; }
  if (ascify_test_exit_registration_status == 0) {
    ascify_test_exit_cleanup = callback;
    if (ascify_test_real_exit) { return std::atexit(callback); }
  }
  return ascify_test_exit_registration_status;
}

aclError ascify_test_runtime_status = ACL_SUCCESS;
void* ascify_test_allocation_result = reinterpret_cast<void*>(0x1234);
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

void testOwnedLifecycle() {
  void* first = nullptr;
  assert(ascify::cudaMalloc(&first, 64) == ACL_SUCCESS);
  assert(first == ascify_test_allocation_result);
  assert(ascify_test_init_calls == 1);
  assert(ascify_test_exit_registration_calls == 2);
  assert(ascify_test_registration_observed_init_calls == 1);
  assert(ascify_test_registration_observed_initialized);
  assert(ascify_test_exit_cleanup != nullptr);
  assert(ascify_test_set_device_calls == 1);
  assert(ascify_test_current_device == 0);
  assert(ascify_test_malloc_calls == 1);

  void* second = nullptr;
  assert(ascify::cudaMalloc(&second, 32) == ACL_SUCCESS);
  assert(ascify_test_init_calls == 1);
  assert(ascify_test_set_device_calls == 1);
  assert(ascify_test_malloc_calls == 2);

  assert(ascify::cudaSetDevice(3) == ACL_SUCCESS);
  assert(ascify::cudaSetDevice(3) == ACL_SUCCESS);
  assert(ascify_test_set_device_calls == 3);
  int device = -1;
  assert(ascify::cudaGetDevice(&device) == ACL_SUCCESS);
  assert(device == 3);

  assert(ascify::cudaDeviceReset() == ACL_SUCCESS);
  assert(ascify_test_reset_device_calls == 1);
  assert(ascify_test_reset_devices[0] == 3);
  assert(!ascify_test_context_bound);

  // CUDA keeps the selected device across cudaDeviceReset. The next runtime
  // operation recreates that device's default ACL context, not device zero.
  assert(ascify::cudaMalloc(&second, 16) == ACL_SUCCESS);
  assert(ascify_test_current_device == 3);
  assert(ascify_test_set_device_calls == 4);
  assert(ascify_test_exit_registration_calls == 2);

  int count = -1;
  assert(ascify::cudaGetDeviceCount(&count) == ACL_SUCCESS);
  assert(count == 7);
  assert(ascify_test_get_device_count_calls == 1);
  assert(ascify::cudaDeviceSynchronize() == ACL_SUCCESS);
  assert(ascify_test_synchronize_device_calls == 1);

  aclrtStream stream = nullptr;
  aclrtEvent start = nullptr;
  aclrtEvent end = nullptr;
  assert(ascify::cudaStreamCreate(&stream) == ACL_SUCCESS);
  assert(ascify_test_create_stream_calls == 1);
  aclrtStream flagged_stream = nullptr;
  assert(ascify::cudaStreamCreateWithFlags(
             &flagged_stream, ascify::cudaStreamDefault) == ACL_SUCCESS);
  assert(flagged_stream != nullptr);
  assert(ascify_test_create_stream_calls == 2);
  aclrtStream rejected_stream = reinterpret_cast<aclrtStream>(0x99);
  assert(ascify::cudaStreamCreateWithFlags(
             &rejected_stream, ascify::cudaStreamNonBlocking) ==
         ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(rejected_stream == nullptr);
  assert(ascify_test_create_stream_calls == 2);
  rejected_stream = reinterpret_cast<aclrtStream>(0x99);
  assert(ascify::cudaStreamCreateWithFlags(&rejected_stream, 0x80U) ==
         ACL_ERROR_RT_PARAM_INVALID);
  assert(rejected_stream == nullptr);
  assert(ascify_test_create_stream_calls == 2);
  assert(ascify::cudaEventCreate(&start) == ACL_SUCCESS);
  assert(ascify::cudaEventCreate(&end) == ACL_SUCCESS);
  assert(ascify::cudaEventRecord(start, stream) == ACL_SUCCESS);
  assert(ascify_test_get_default_stream_calls == 0);
  assert(ascify_test_recorded_event_stream == stream);
  assert(ascify::cudaEventRecord(start) == ACL_SUCCESS);
  assert(ascify_test_get_default_stream_calls == 1);
  assert(ascify_test_recorded_event_stream == ascify_test_default_stream);
  assert(ascify_test_record_event_calls == 2);
  assert(ascify::cudaEventSynchronize(start) == ACL_SUCCESS);
  float milliseconds = 0.0f;
  assert(ascify::cudaEventElapsedTime(&milliseconds, start, end) ==
         ACL_SUCCESS);
  assert(milliseconds == 1.0f);
  assert(ascify::cudaEventDestroy(start) == ACL_SUCCESS);
  assert(ascify::cudaEventDestroy(end) == ACL_SUCCESS);
  assert(ascify::cudaStreamSynchronize(stream) == ACL_SUCCESS);
  assert(ascify::cudaStreamDestroy(stream) == ACL_SUCCESS);
  assert(ascify::cudaStreamDestroy(flagged_stream) == ACL_SUCCESS);
  assert(ascify::cudaFree(first) == ACL_SUCCESS);
  assert(ascify::cudaFree(second) == ACL_SUCCESS);

  ascify_test_exit_cleanup();
  assert(ascify_test_finalize_calls == 1);
  assert(ascify_test_reset_device_calls == 3);
  assert(ascify_test_reset_devices[1] == 3);
  assert(ascify_test_reset_devices[2] == 0);

  // Cleanup is idempotent, and a call made after finalization fails closed.
  ascify_test_exit_cleanup();
  ascify::detail::runtime_manager.shutdown();
  assert(ascify_test_finalize_calls == 1);
  assert(ascify::cudaRuntimeEnsureReady() == ACL_ERROR_REPEAT_FINALIZE);
}

void testBorrowedInitialization() {
  ascify_test_init_status = ACL_ERROR_REPEAT_INITIALIZE;
  ascify_test_acl_initialized = true;

  void* device = nullptr;
  assert(ascify::cudaMalloc(&device, 64) == ACL_SUCCESS);
  assert(ascify_test_init_calls == 1);
  assert(ascify_test_set_device_calls == 1);
  assert(ascify_test_exit_registration_calls == 2);
  assert(ascify_test_registration_observed_init_calls == 1);
  assert(ascify_test_registration_observed_initialized);
  assert(ascify_test_exit_cleanup != nullptr);
  ascify_test_exit_cleanup();

  // The translated path owns the context it created, but not an embedding
  // application's earlier successful aclInit.
  assert(ascify_test_reset_device_calls == 1);
  assert(ascify_test_reset_devices[0] == 0);
  assert(ascify_test_finalize_calls == 0);
}

void testInitializationFailure() {
  ascify_test_init_status = ACL_ERROR_RT_PARAM_INVALID;

  void* device = reinterpret_cast<void*>(1);
  assert(ascify::cudaMalloc(&device, 64) == ACL_ERROR_RT_PARAM_INVALID);
  assert(device == nullptr);
  assert(ascify_test_init_calls == 1);
  assert(ascify_test_exit_registration_calls == 0);
  assert(ascify_test_set_device_calls == 0);
  assert(ascify_test_malloc_calls == 0);

  // Lifecycle errors participate in CUDA's peek/get-last-error shape even
  // though ACL initialization failed before an ACL thread error existed.
  assert(ascify::cudaPeekAtLastError() == ACL_ERROR_RT_PARAM_INVALID);
  assert(ascify::cudaGetLastError() == ACL_ERROR_RT_PARAM_INVALID);
  assert(ascify::cudaRuntimeEnsureReady() == ACL_ERROR_RT_PARAM_INVALID);
  assert(ascify_test_init_calls == 1);

  ascify::detail::runtime_manager.shutdown();
  assert(ascify_test_reset_device_calls == 0);
  assert(ascify_test_finalize_calls == 0);
}

void testFixedRegistryCapacity() {
  const size_t capacity =
      ascify::detail::RuntimeManager::kTrackedDeviceCapacity;
  assert(capacity == 64U);
  for (size_t index = 0; index < capacity; ++index) {
    assert(ascify::cudaSetDevice(static_cast<int>(index)) == ACL_SUCCESS);
  }

  // The 65th distinct binding is rolled back and reported. No dynamic
  // allocation or exception is permitted in the target host path.
  assert(ascify::cudaSetDevice(static_cast<int>(capacity)) ==
         ACL_ERROR_BAD_ALLOC);
  assert(ascify_test_set_device_calls == static_cast<int>(capacity + 1));
  assert(ascify_test_reset_device_calls == 1);
  assert(ascify_test_reset_devices[0] == static_cast<int32_t>(capacity));

  assert(ascify_test_exit_registration_calls == 2);
  assert(ascify_test_exit_cleanup != nullptr);
  ascify_test_exit_cleanup();
  assert(ascify_test_finalize_calls == 1);
  assert(ascify_test_reset_device_calls == static_cast<int>(capacity + 1));
  for (size_t index = 0; index < capacity; ++index) {
    assert(ascify_test_reset_devices[index + 1] ==
           static_cast<int32_t>(capacity - index - 1));
  }
}

void testExplicitResetThenExitCleanup() {
  assert(ascify::cudaSetDevice(5) == ACL_SUCCESS);
  assert(ascify_test_exit_registration_calls == 2);
  assert(ascify_test_exit_cleanup != nullptr);
  assert(ascify::cudaDeviceReset() == ACL_SUCCESS);
  assert(ascify_test_reset_device_calls == 1);
  assert(ascify_test_reset_devices[0] == 5);

  ascify_test_exit_cleanup();
  assert(ascify_test_reset_device_calls == 1);
  assert(ascify_test_finalize_calls == 1);

  // The registered callback and later global destructor are both idempotent.
  ascify_test_exit_cleanup();
  ascify::detail::runtime_manager.shutdown();
  assert(ascify_test_reset_device_calls == 1);
  assert(ascify_test_finalize_calls == 1);
}

void testOwnedExitRegistrationFailure() {
  ascify_test_exit_registration_status = 1;
  void* device = reinterpret_cast<void*>(1);
  assert(ascify::cudaMalloc(&device, 64) == ACL_ERROR_BAD_ALLOC);
  assert(device == nullptr);
  assert(ascify_test_init_calls == 1);
  assert(ascify_test_exit_registration_calls == 1);
  assert(ascify_test_exit_cleanup == nullptr);
  assert(ascify_test_finalize_calls == 1);
  assert(ascify_test_set_device_calls == 0);
  assert(ascify_test_malloc_calls == 0);

  // A failed registration is sticky; do not repeat aclInit or registration.
  assert(ascify::cudaRuntimeEnsureReady() == ACL_ERROR_BAD_ALLOC);
  assert(ascify_test_init_calls == 1);
  assert(ascify_test_exit_registration_calls == 1);
  ascify::detail::runtime_manager.shutdown();
  assert(ascify_test_finalize_calls == 1);
}

void testBorrowedExitRegistrationFailure() {
  ascify_test_init_status = ACL_ERROR_REPEAT_INITIALIZE;
  ascify_test_acl_initialized = true;
  ascify_test_exit_registration_status = 1;
  assert(ascify::cudaRuntimeEnsureReady() == ACL_ERROR_BAD_ALLOC);
  assert(ascify_test_init_calls == 1);
  assert(ascify_test_exit_registration_calls == 1);
  assert(ascify_test_exit_cleanup == nullptr);
  assert(ascify_test_finalize_calls == 0);
  assert(ascify_test_set_device_calls == 0);
  ascify::detail::runtime_manager.shutdown();
  assert(ascify_test_finalize_calls == 0);
}

void testRegistrationRollbackErrorPropagation() {
  ascify_test_exit_registration_status = 1;
  ascify_test_finalize_status = ACL_ERROR_RT_PARAM_INVALID;
  assert(ascify::cudaRuntimeEnsureReady() == ACL_ERROR_RT_PARAM_INVALID);
  assert(ascify_test_init_calls == 1);
  assert(ascify_test_exit_registration_calls == 1);
  assert(ascify_test_finalize_calls == 1);
  assert(ascify_test_exit_cleanup == nullptr);
  ascify::detail::runtime_manager.shutdown();
  assert(ascify_test_finalize_calls == 1);
}

void testLateExitRegistrationFailure(bool borrowed, bool bind_fails,
                                     bool reset_fails, bool finalize_fails) {
  ascify_test_exit_registration_fail_at = 2;
  if (borrowed) {
    ascify_test_init_status = ACL_ERROR_REPEAT_INITIALIZE;
    ascify_test_acl_initialized = true;
  }
  if (bind_fails) {
    ascify_test_set_device_status = ACL_ERROR_RT_NO_DEVICE;
  }
  if (reset_fails) {
    ascify_test_reset_device_status = ACL_ERROR_RT_PARAM_INVALID;
  }
  if (finalize_fails) {
    ascify_test_finalize_status = ACL_ERROR_RT_CONTEXT_NULL;
  }
  const aclError expected =
      reset_fails ? ACL_ERROR_RT_PARAM_INVALID :
      finalize_fails ? ACL_ERROR_RT_CONTEXT_NULL :
      bind_fails ? ACL_ERROR_RT_NO_DEVICE : ACL_ERROR_BAD_ALLOC;
  void* device = reinterpret_cast<void*>(1);
  assert(ascify::cudaMalloc(&device, 64) == expected);
  assert(device == nullptr);
  assert(ascify_test_malloc_calls == 0);
  assert(ascify_test_set_device_calls == 1);
  assert(ascify_test_exit_registration_calls == 2);
  assert(ascify_test_reset_device_calls == (bind_fails ? 0 : 1));
  assert(ascify_test_finalize_calls == (borrowed ? 0 : 1));
  assert(ascify::cudaRuntimeEnsureReady() == expected);
  assert(ascify::cudaSetDevice(3) == expected);
  assert(ascify::cudaPeekAtLastError() == expected);
  assert(ascify_test_set_device_calls == 1);
  ascify_test_exit_cleanup();
  ascify::detail::runtime_manager.shutdown();
  assert(ascify_test_reset_device_calls == (bind_fails ? 0 : 1));
  assert(ascify_test_finalize_calls == (borrowed ? 0 : 1));
}

// Use the real process atexit stack. A simulated SDK singleton registers its
// own teardown inside SetDevice, including before a failed call returns. An
// old after-aclInit-only callback fails these checks at normal process exit.
void testRealExitOrder(const char* scenario) {
  ascify_test_real_exit = true;
  const bool borrowed = std::strcmp(scenario, "exit-order-borrowed") == 0;
  const bool init_only = std::strcmp(scenario, "exit-order-init-only") == 0;
  const bool failed_only = std::strcmp(scenario, "exit-order-failed-bind") == 0;
  const bool retry = std::strcmp(scenario, "exit-order-retry") == 0;
  const bool late_failure = std::strcmp(scenario, "exit-order-late-failure") == 0;
  ascify_test_second_lazy_phase = retry;
  ascify_test_expected_exit_resets = init_only || failed_only ? 0 : 1;
  ascify_test_expected_exit_finalizes = borrowed ? 0 : 1;
  if (borrowed) {
    ascify_test_init_status = ACL_ERROR_REPEAT_INITIALIZE;
    ascify_test_acl_initialized = true;
  }
  assert(std::atexit(ascify_test_verify_exit_cleanup) == 0);
  if (init_only) {
    int count = 0;
    assert(ascify::cudaGetDeviceCount(&count) == ACL_SUCCESS);
    assert(count == 7);
    assert(ascify_test_exit_registration_calls == 1);
    return;
  }
  if (failed_only || retry) {
    ascify_test_set_device_status = ACL_ERROR_RT_NO_DEVICE;
    assert(ascify::cudaSetDevice(0) == ACL_ERROR_RT_NO_DEVICE);
    assert(ascify_test_reset_device_calls == 0);
    assert(ascify_test_exit_registration_calls == 2);
    if (failed_only) { return; }
    ascify_test_set_device_status = ACL_SUCCESS;
  }
  if (late_failure) { ascify_test_exit_registration_fail_at = 2; }
  assert(ascify::cudaSetDevice(0) ==
         (late_failure ? ACL_ERROR_BAD_ALLOC : ACL_SUCCESS));
  if (!late_failure) {
    assert(ascify::cudaSetDevice(0) == ACL_SUCCESS);
  }
  // Return normally. No manual manager shutdown or cleanup callback invocation.
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  if (std::strcmp(argv[1], "owned") == 0) {
    testOwnedLifecycle();
  } else if (std::strcmp(argv[1], "borrowed") == 0) {
    testBorrowedInitialization();
  } else if (std::strcmp(argv[1], "failure") == 0) {
    testInitializationFailure();
  } else if (std::strcmp(argv[1], "capacity") == 0) {
    testFixedRegistryCapacity();
  } else if (std::strcmp(argv[1], "reset-cleanup") == 0) {
    testExplicitResetThenExitCleanup();
  } else if (std::strcmp(argv[1], "registration-failure-owned") == 0) {
    testOwnedExitRegistrationFailure();
  } else if (std::strcmp(argv[1], "registration-failure-borrowed") == 0) {
    testBorrowedExitRegistrationFailure();
  } else if (std::strcmp(argv[1], "registration-rollback-error") == 0) {
    testRegistrationRollbackErrorPropagation();
  } else if (std::strcmp(argv[1], "late-registration-owned") == 0) {
    testLateExitRegistrationFailure(false, false, false, false);
  } else if (std::strcmp(argv[1], "late-registration-borrowed") == 0) {
    testLateExitRegistrationFailure(true, false, false, false);
  } else if (std::strcmp(argv[1], "late-registration-reset-error") == 0) {
    testLateExitRegistrationFailure(false, false, true, false);
  } else if (std::strcmp(argv[1], "late-registration-finalize-error") == 0) {
    testLateExitRegistrationFailure(false, false, false, true);
  } else if (std::strcmp(argv[1], "late-registration-bind-error") == 0) {
    testLateExitRegistrationFailure(false, true, false, false);
  } else if (std::strncmp(argv[1], "exit-order-", 11) == 0) {
    testRealExitOrder(argv[1]);
  } else {
    return 2;
  }
  return 0;
}
