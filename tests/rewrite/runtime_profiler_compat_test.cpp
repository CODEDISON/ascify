#define __aicore__
#define ASCIFY_TEST_CONTROLLABLE_PROFILER_EXIT_REGISTRATION
int ascify_test_register_profiler_exit_cleanup(void (*)());
#include <ascify/ascify_cuda_compat.hpp>

#include <cassert>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {
std::vector<std::string> calls;
aclprofConfig config;
aclError init_status = ACL_SUCCESS;
aclError start_status = ACL_SUCCESS;
aclError stop_status = ACL_SUCCESS;
aclError destroy_status = ACL_SUCCESS;
aclError finalize_status = ACL_SUCCESS;
aclError context_status = ACL_SUCCESS;
bool create_fails = false;
int registration_status = 0;
void (*cleanup_callback)() = nullptr;
thread_local aclrtContext current_context = reinterpret_cast<aclrtContext>(0x42);
constexpr aclError failure = 507021;
void expect(std::initializer_list<const char*> expected) {
  assert(calls.size() == expected.size());
  size_t i = 0;
  for (const char* value : expected) { assert(calls[i++] == value); }
}
}

aclError aclrtGetCurrentContext(aclrtContext* context) {
  if (context_status == ACL_SUCCESS) { *context = current_context; }
  return context_status;
}
aclError aclprofInit(const char* path, size_t length) {
  calls.emplace_back("init");
  assert(std::string(path, length) == "/explicit/profile/path");
  return init_status;
}
aclprofConfig* aclprofCreateConfig(uint32_t* devices, uint32_t count,
    aclprofAicoreMetrics metric, const aclprofAicoreEvents* events, uint64_t flags) {
  calls.emplace_back("create");
  assert(count == 1 && devices[0] == 0 && metric == ACL_AICORE_NONE);
  assert(events == nullptr && flags == (ACL_PROF_ACL_API | ACL_PROF_TASK_TIME));
  return create_fails ? nullptr : &config;
}
aclError aclprofStart(const aclprofConfig* value) {
  assert(value == &config); calls.emplace_back("start"); return start_status;
}
aclError aclprofStop(const aclprofConfig* value) {
  assert(value == &config); calls.emplace_back("stop"); return stop_status;
}
aclError aclprofDestroyConfig(const aclprofConfig* value) {
  assert(value == &config); calls.emplace_back("destroy"); return destroy_status;
}
aclError aclprofFinalize() {
  calls.emplace_back("finalize"); return finalize_status;
}
int ascify_test_register_profiler_exit_cleanup(void (*callback)()) {
  calls.emplace_back("register"); cleanup_callback = callback;
  return registration_status;
}

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::string mode = argv[1];
  unsetenv("ASCIFY_PROFILE_DIR");
  if (mode == "disabled") {
    assert(ascify::cudaProfilerStart() == ACL_ERROR_FEATURE_UNSUPPORTED);
    assert(ascify::cudaProfilerStop() == ACL_ERROR_FEATURE_UNSUPPORTED);
    assert(calls.empty());
    return 0;
  }
  setenv("ASCIFY_PROFILE_DIR", "/explicit/profile/path", 1);
  if (mode == "context-error") {
    context_status = failure;
    assert(ascify::cudaProfilerStart() == failure);
    assert(calls.empty());
    return 0;
  }
  if (mode == "null-context") {
    current_context = nullptr;
    assert(ascify::cudaProfilerStart() == ACL_ERROR_RT_CONTEXT_NULL);
    assert(calls.empty());
    return 0;
  }
  if (mode == "init-error") {
    init_status = failure;
    assert(ascify::cudaProfilerStart() == failure);
    assert(ascify::cudaProfilerStop() == ACL_SUCCESS);
    expect({"init"});
    return 0;
  }
  if (mode == "create-error" || mode == "create-rollback-error") {
    create_fails = true;
    if (mode == "create-rollback-error") { finalize_status = failure; }
    assert(ascify::cudaProfilerStart() == ACL_ERROR_PROFILING_FAILURE);
    expect({"init", "create", "finalize"});
    if (mode == "create-rollback-error") {
      assert(ascify::cudaProfilerStart() == failure);
      assert(ascify::cudaProfilerStop() == failure);
      finalize_status = ACL_SUCCESS;
    }
    assert(ascify::cudaProfilerStop() == ACL_SUCCESS);
    return 0;
  }
  if (mode == "start-error") {
    start_status = failure;
    assert(ascify::cudaProfilerStart() == failure);
    expect({"init", "create", "start", "destroy", "finalize"});
    return 0;
  }
  if (mode == "registration-error") {
    registration_status = -1;
    assert(ascify::cudaProfilerStart() == ACL_ERROR_BAD_ALLOC);
    expect({"init", "create", "start", "register", "stop", "destroy", "finalize"});
    return 0;
  }
  assert(ascify::cudaProfilerStop() == ACL_SUCCESS);
  if (mode == "threads") {
    std::vector<std::thread> threads;
    for (int i = 0; i < 16; ++i) {
      threads.emplace_back([] { assert(ascify::cudaProfilerStart() == ACL_SUCCESS); });
    }
    for (auto& thread : threads) { thread.join(); }
  } else { assert(ascify::cudaProfilerStart() == ACL_SUCCESS); }
  expect({"init", "create", "start", "register"});
  assert(ascify::cudaProfilerStart() == ACL_SUCCESS);
  expect({"init", "create", "start", "register"});
  calls.clear();
  if (mode == "wrong-context") {
    current_context = reinterpret_cast<aclrtContext>(0x43);
    assert(ascify::cudaProfilerStart() == ACL_ERROR_FEATURE_UNSUPPORTED);
    assert(ascify::cudaProfilerStop() == ACL_ERROR_FEATURE_UNSUPPORTED);
    assert(calls.empty());
    current_context = reinterpret_cast<aclrtContext>(0x42);
  } else if (mode == "stop-error") {
    stop_status = failure;
    assert(ascify::cudaProfilerStop() == failure);
    assert(ascify::cudaProfilerStart() == failure);
    expect({"stop"}); calls.clear(); stop_status = ACL_SUCCESS;
  } else if (mode == "destroy-error") {
    destroy_status = failure;
    assert(ascify::cudaProfilerStop() == failure);
    assert(ascify::cudaProfilerStart() == failure);
    expect({"stop", "destroy"}); calls.clear(); destroy_status = ACL_SUCCESS;
    assert(ascify::cudaProfilerStop() == ACL_SUCCESS);
    expect({"destroy", "finalize"}); return 0;
  } else if (mode == "finalize-error") {
    finalize_status = failure;
    assert(ascify::cudaProfilerStop() == failure);
    assert(ascify::cudaProfilerStart() == failure);
    expect({"stop", "destroy", "finalize"}); calls.clear(); finalize_status = ACL_SUCCESS;
    assert(ascify::cudaProfilerStop() == ACL_SUCCESS);
    expect({"finalize"}); return 0;
  } else if (mode == "exit-cleanup") {
    assert(cleanup_callback != nullptr); cleanup_callback();
    expect({"stop", "destroy", "finalize"});
    cleanup_callback(); expect({"stop", "destroy", "finalize"}); return 0;
  }
  if (mode == "threads") {
    std::vector<std::thread> threads;
    for (int i = 0; i < 16; ++i) {
      threads.emplace_back([] { assert(ascify::cudaProfilerStop() == ACL_SUCCESS); });
    }
    for (auto& thread : threads) { thread.join(); }
  } else { assert(ascify::cudaProfilerStop() == ACL_SUCCESS); }
  assert(ascify::cudaProfilerStop() == ACL_SUCCESS);
  expect({"stop", "destroy", "finalize"});
  if (mode == "repeat") {
    calls.clear();
    assert(ascify::cudaProfilerStart() == ACL_SUCCESS);
    assert(ascify::cudaProfilerStop() == ACL_SUCCESS);
    expect({"init", "create", "start", "stop", "destroy", "finalize"});
  }
  return 0;
}
