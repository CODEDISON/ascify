#define __aicore__
#define ASCIFY_TEST_ACL_CONTROLLABLE_RUNTIME_LIFECYCLE

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
uint32_t ascify_test_device_count = 1;
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

void assertSuccessfulSelection(int selected) {
  assert(selected == 0);
  assert(ascify_test_get_device_count_calls == 1);
  assert(ascify_test_set_device_calls == 1);
  assert(ascify_test_current_device == 0);
  assert(ascify_test_context_bound);
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  if (std::strcmp(argv[1], "success-default") == 0) {
    const char* sample_argv[] = {"sample", "--size=64"};
    assertSuccessfulSelection(
        ascify::sampleFindCudaDevice(2, sample_argv));
    return 0;
  }
  if (std::strcmp(argv[1], "success-explicit-zero") == 0) {
    const char* sample_argv[] = {"sample", "--device=0"};
    const int selected = ascify::sampleFindCudaDevice(2, sample_argv);
    assertSuccessfulSelection(selected);
    return selected;
  }
  if (std::strcmp(argv[1], "zero-devices") == 0) {
    ascify_test_device_count = 0;
    const char* sample_argv[] = {"sample"};
    (void)ascify::sampleFindCudaDevice(1, sample_argv);
  }
  if (std::strcmp(argv[1], "multiple-devices") == 0) {
    // Multi-visible environments now select logical device zero (the epoch
    // performance tier) instead of failing; zero visible devices still does.
    ascify_test_device_count = 2;
    const char* sample_argv[] = {"sample"};
    const int selected = ascify::sampleFindCudaDevice(1, sample_argv);
    assertSuccessfulSelection(selected);
    return 0;
  }
  if (std::strcmp(argv[1], "other-device") == 0) {
    const char* sample_argv[] = {"sample", "--device=1"};
    (void)ascify::sampleFindCudaDevice(2, sample_argv);
  }
  if (std::strcmp(argv[1], "legacy-device-spelling") == 0) {
    const char* sample_argv[] = {"sample", "-device=0"};
    (void)ascify::sampleFindCudaDevice(2, sample_argv);
  }
  if (std::strcmp(argv[1], "duplicate-device") == 0) {
    const char* sample_argv[] = {"sample", "--device=0", "--device=0"};
    (void)ascify::sampleFindCudaDevice(3, sample_argv);
  }
  if (std::strcmp(argv[1], "negative-argc") == 0) {
    (void)ascify::sampleFindCudaDevice(-1, nullptr);
  }
  if (std::strcmp(argv[1], "null-argv") == 0) {
    (void)ascify::sampleFindCudaDevice(1, nullptr);
  }
  if (std::strcmp(argv[1], "null-argument") == 0) {
    const char* sample_argv[] = {"sample", nullptr};
    (void)ascify::sampleFindCudaDevice(2, sample_argv);
  }
  if (std::strcmp(argv[1], "query-failure") == 0) {
    ascify_test_device_count_status = ACL_ERROR_RT_PARAM_INVALID;
    const char* sample_argv[] = {"sample"};
    (void)ascify::sampleFindCudaDevice(1, sample_argv);
  }
  if (std::strcmp(argv[1], "bind-failure") == 0) {
    ascify_test_set_device_status = ACL_ERROR_RT_PARAM_INVALID;
    const char* sample_argv[] = {"sample"};
    (void)ascify::sampleFindCudaDevice(1, sample_argv);
  }
  return 90;
}
