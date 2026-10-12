#define __aicore__
#include <ascify/ascify_cuda_compat.hpp>
#include <cassert>

namespace {
int launch_calls = 0;
int callback_calls = 0;
aclError status = ACL_SUCCESS;
aclrtStream recorded_stream = nullptr;
aclrtHostFunc recorded_function = nullptr;
void* recorded_data = nullptr;
void callback(void*) { ++callback_calls; }
}
aclError aclrtLaunchHostFunc(aclrtStream stream, aclrtHostFunc function, void* data) {
  ++launch_calls;
  recorded_stream = stream;
  recorded_function = function;
  recorded_data = data;
  // Enqueue must not be replaced with a synchronous user-function invocation.
  return status;
}
int main() {
  int data = 42;
  aclrtStream stream = reinterpret_cast<aclrtStream>(0x1234);
  assert(ascify::cudaLaunchHostFunc(nullptr, callback, &data) == ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(ascify::cudaLaunchHostFunc(stream, nullptr, &data) == ACL_ERROR_RT_PARAM_INVALID);
  assert(launch_calls == 0 && callback_calls == 0);
  assert(ascify::cudaLaunchHostFunc(stream, callback, &data) == ACL_SUCCESS);
  assert(launch_calls == 1 && callback_calls == 0);
  assert(recorded_stream == stream && recorded_function == callback && recorded_data == &data);
  status = 507021;
  assert(ascify::cudaLaunchHostFunc(stream, callback, nullptr) == status);
  assert(launch_calls == 2 && callback_calls == 0 && recorded_data == nullptr);
}
