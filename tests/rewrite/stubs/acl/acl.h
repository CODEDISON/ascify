#ifndef ASCIFY_TEST_STUB_ACL_H
#define ASCIFY_TEST_STUB_ACL_H

#include <stddef.h>
#include <stdint.h>

#if defined(ASCIFY_TEST_PUBLIC_85_ACL)
#define ACL_MAJOR_VERSION 1
#define ACL_MINOR_VERSION 16
#define ACL_PATCH_VERSION 0
#else
#define ACL_MAJOR_VERSION 1
#define ACL_MINOR_VERSION 17
#define ACL_PATCH_VERSION 0
#endif

typedef int aclError;
typedef void* aclrtStream;
typedef void* aclrtEvent;

enum aclrtMemcpyKind {
  ACL_MEMCPY_HOST_TO_HOST,
  ACL_MEMCPY_HOST_TO_DEVICE,
  ACL_MEMCPY_DEVICE_TO_HOST,
  ACL_MEMCPY_DEVICE_TO_DEVICE
};

enum aclrtMemMallocPolicy {
  ACL_MEM_MALLOC_HUGE_FIRST
};

enum aclrtDevAttr {
  ACL_DEV_ATTR_VECTOR_CORE_NUM = 201,
#if !defined(ASCIFY_TEST_PUBLIC_85_ACL)
  ACL_DEV_ATTR_WARP_SIZE = 202,
  ACL_DEV_ATTR_MAX_THREAD_PER_VECTOR_CORE = 203,
  ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE = 204,
  ACL_DEV_ATTR_LOCAL_MEM_PER_VECTOR_CORE = 204,
  ACL_DEV_ATTR_MAX_THREADS_PER_BLOCK = 209
#endif
};

enum aclrtLastErrLevel {
  ACL_RT_THREAD_LEVEL
};

static const aclError ACL_SUCCESS = 0;
static const aclError ACL_ERROR_RT_PARAM_INVALID = 1;
static const aclError ACL_ERROR_RT_MALLOC_FAILED = 2;
static const aclError ACL_ERROR_BAD_ALLOC = 200000;
static const aclError ACL_ERROR_FEATURE_UNSUPPORTED = 200006;
static const aclError ACL_ERROR_REPEAT_INITIALIZE = 100002;
static const aclError ACL_ERROR_REPEAT_FINALIZE = 100037;
static const aclError ACL_ERROR_RT_CONTEXT_NULL = 107002;
static const aclError ACL_ERROR_RT_NO_DEVICE = 207004;

#ifdef ASCIFY_TEST_ACL_CONTROLLABLE_RUNTIME_LIFECYCLE
extern aclError ascify_test_init_status;
extern aclError ascify_test_set_device_status;
extern aclError ascify_test_reset_device_status;
extern aclError ascify_test_device_count_status;
extern aclError ascify_test_synchronize_status;
extern aclError ascify_test_finalize_status;
extern bool ascify_test_acl_initialized;
extern bool ascify_test_context_bound;
extern int32_t ascify_test_current_device;
extern uint32_t ascify_test_device_count;
extern int ascify_test_init_calls;
extern int ascify_test_finalize_calls;
extern int ascify_test_set_device_calls;
extern int ascify_test_get_device_calls;
extern int ascify_test_reset_device_calls;
extern int ascify_test_get_device_count_calls;
extern int ascify_test_synchronize_device_calls;
extern int ascify_test_get_default_stream_calls;
extern int ascify_test_create_stream_calls;
extern int ascify_test_record_event_calls;
extern aclrtStream ascify_test_default_stream;
extern aclrtStream ascify_test_recorded_event_stream;
extern int32_t ascify_test_reset_devices[80];

inline aclError aclInit(const char*) {
  ++ascify_test_init_calls;
  if (ascify_test_init_status == ACL_SUCCESS) {
    ascify_test_acl_initialized = true;
  }
  return ascify_test_init_status;
}

inline aclError aclFinalize() {
  ++ascify_test_finalize_calls;
  if (ascify_test_finalize_status == ACL_SUCCESS) {
    ascify_test_acl_initialized = false;
  }
  return ascify_test_finalize_status;
}

inline aclError aclrtSetDevice(int32_t device) {
  ++ascify_test_set_device_calls;
  if (!ascify_test_acl_initialized) { return ACL_ERROR_RT_CONTEXT_NULL; }
  if (ascify_test_set_device_status == ACL_SUCCESS) {
    ascify_test_current_device = device;
    ascify_test_context_bound = true;
  }
  return ascify_test_set_device_status;
}

inline aclError aclrtGetDevice(int32_t* device) {
  ++ascify_test_get_device_calls;
  if (!ascify_test_context_bound) { return ACL_ERROR_RT_CONTEXT_NULL; }
  *device = ascify_test_current_device;
  return ACL_SUCCESS;
}

inline aclError aclrtResetDevice(int32_t device) {
  const int call = ascify_test_reset_device_calls++;
  if (call < 80) { ascify_test_reset_devices[call] = device; }
  if (ascify_test_reset_device_status == ACL_SUCCESS &&
      device == ascify_test_current_device) {
    ascify_test_context_bound = false;
  }
  return ascify_test_reset_device_status;
}

inline aclError aclrtGetDeviceCount(uint32_t* count) {
  ++ascify_test_get_device_count_calls;
  if (ascify_test_device_count_status == ACL_SUCCESS) {
    *count = ascify_test_device_count;
  }
  return ascify_test_device_count_status;
}

inline aclError aclrtSynchronizeDevice() {
  ++ascify_test_synchronize_device_calls;
  return ascify_test_synchronize_status;
}
#else
inline aclError aclInit(const char*) { return ACL_SUCCESS; }
inline aclError aclFinalize() { return ACL_SUCCESS; }
inline aclError aclrtSetDevice(int32_t) { return ACL_SUCCESS; }
inline aclError aclrtResetDevice(int32_t) { return ACL_SUCCESS; }
inline aclError aclrtGetDeviceCount(uint32_t* count) {
  *count = 1;
  return ACL_SUCCESS;
}
inline aclError aclrtSynchronizeDevice() { return ACL_SUCCESS; }
#endif

#ifdef ASCIFY_TEST_ACL_CONTROLLABLE_RUNTIME_MEMORY
extern aclError ascify_test_runtime_status;
extern void* ascify_test_allocation_result;
extern const char* ascify_test_recent_error_message;
extern aclError ascify_test_last_error;
#ifdef ASCIFY_TEST_ACL_CONSUMING_LAST_ERROR
extern int ascify_test_get_last_error_calls;
#endif
extern int ascify_test_malloc_calls;
extern int ascify_test_malloc_host_calls;
extern int ascify_test_free_calls;
extern int ascify_test_free_host_calls;
extern int ascify_test_memcpy_calls;
extern int ascify_test_memcpy_async_calls;
extern int ascify_test_memset_calls;
extern int ascify_test_memset_async_calls;
extern void** ascify_test_output_pointer;
extern void* ascify_test_destination;
extern const void* ascify_test_source;
extern size_t ascify_test_destination_max;
extern size_t ascify_test_count;
extern int ascify_test_value;
extern aclrtMemcpyKind ascify_test_memcpy_kind;
extern aclrtStream ascify_test_stream;
extern aclrtMemMallocPolicy ascify_test_malloc_policy;

inline aclError aclrtMalloc(
    void** pointer, size_t, aclrtMemMallocPolicy policy) {
  ++ascify_test_malloc_calls;
  ascify_test_output_pointer = pointer;
  ascify_test_malloc_policy = policy;
  if (ascify_test_runtime_status == ACL_SUCCESS) {
    *pointer = ascify_test_allocation_result;
  }
  return ascify_test_runtime_status;
}

inline aclError aclrtMallocHost(void** pointer, size_t) {
  ++ascify_test_malloc_host_calls;
  ascify_test_output_pointer = pointer;
  if (ascify_test_runtime_status == ACL_SUCCESS) {
    *pointer = ascify_test_allocation_result;
  }
  return ascify_test_runtime_status;
}

inline aclError aclrtFree(void* pointer) {
  ++ascify_test_free_calls;
  ascify_test_destination = pointer;
  return ascify_test_runtime_status;
}

inline aclError aclrtFreeHost(void* pointer) {
  ++ascify_test_free_host_calls;
  ascify_test_destination = pointer;
  return ascify_test_runtime_status;
}

inline aclError aclrtMemcpy(void* destination, size_t destination_max,
                            const void* source, size_t count,
                            aclrtMemcpyKind kind) {
  ++ascify_test_memcpy_calls;
  ascify_test_destination = destination;
  ascify_test_destination_max = destination_max;
  ascify_test_source = source;
  ascify_test_count = count;
  ascify_test_memcpy_kind = kind;
  return ascify_test_runtime_status;
}

inline aclError aclrtMemcpyAsync(void* destination, size_t destination_max,
                                 const void* source, size_t count,
                                 aclrtMemcpyKind kind, aclrtStream stream) {
  ++ascify_test_memcpy_async_calls;
  ascify_test_destination = destination;
  ascify_test_destination_max = destination_max;
  ascify_test_source = source;
  ascify_test_count = count;
  ascify_test_memcpy_kind = kind;
  ascify_test_stream = stream;
  return ascify_test_runtime_status;
}

inline aclError aclrtMemset(void* destination, size_t destination_max,
                            int value, size_t count) {
  ++ascify_test_memset_calls;
  ascify_test_destination = destination;
  ascify_test_destination_max = destination_max;
  ascify_test_value = value;
  ascify_test_count = count;
  return ascify_test_runtime_status;
}

inline aclError aclrtMemsetAsync(void* destination, size_t destination_max,
                                 int value, size_t count,
                                 aclrtStream stream) {
  ++ascify_test_memset_async_calls;
  ascify_test_destination = destination;
  ascify_test_destination_max = destination_max;
  ascify_test_value = value;
  ascify_test_count = count;
  ascify_test_stream = stream;
  return ascify_test_runtime_status;
}

inline aclError aclrtGetLastError(aclrtLastErrLevel) {
#ifdef ASCIFY_TEST_ACL_CONSUMING_LAST_ERROR
  ++ascify_test_get_last_error_calls;
  const aclError status = ascify_test_last_error;
  ascify_test_last_error = ACL_SUCCESS;
  return status;
#else
  return ascify_test_last_error;
#endif
}

inline const char* aclGetRecentErrMsg() {
  return ascify_test_recent_error_message;
}
#else
inline aclError aclrtMalloc(void** pointer, size_t,
                            aclrtMemMallocPolicy) {
  *pointer = reinterpret_cast<void*>(1);
  return ACL_SUCCESS;
}

inline aclError aclrtMallocHost(void** pointer, size_t) {
  *pointer = reinterpret_cast<void*>(1);
  return ACL_SUCCESS;
}

inline aclError aclrtFree(void*) { return ACL_SUCCESS; }
inline aclError aclrtFreeHost(void*) { return ACL_SUCCESS; }

inline aclError aclrtMemcpy(void*, size_t, const void*, size_t,
                            aclrtMemcpyKind) {
  return ACL_SUCCESS;
}

inline aclError aclrtMemcpyAsync(void*, size_t, const void*, size_t,
                                 aclrtMemcpyKind, aclrtStream) {
  return ACL_SUCCESS;
}

inline aclError aclrtMemset(void*, size_t, int, size_t) {
  return ACL_SUCCESS;
}

inline aclError aclrtMemsetAsync(void*, size_t, int, size_t, aclrtStream) {
  return ACL_SUCCESS;
}

inline aclError aclrtGetLastError(aclrtLastErrLevel) {
  return ACL_SUCCESS;
}

inline const char* aclGetRecentErrMsg() { return "ACL runtime error"; }
#endif

#if defined(ASCIFY_TEST_ACL_CONTROLLABLE_RUNTIME_LIFECYCLE)
// The lifecycle stub above owns aclrtGetDevice.
inline aclError aclrtGetDeviceInfo(uint32_t, aclrtDevAttr, int64_t* value) {
  *value = 1024;
  return ACL_SUCCESS;
}
#elif defined(ASCIFY_TEST_ACL_CONTROLLABLE_DEVICE_QUERY)
extern aclError ascify_test_get_device_status;
extern aclError ascify_test_vector_core_info_status;
extern aclError ascify_test_max_thread_info_status;
extern int32_t ascify_test_device;
extern int64_t ascify_test_vector_core_count;
extern int64_t ascify_test_max_threads_per_core;
extern int ascify_test_get_device_calls;
extern int ascify_test_get_device_info_calls;

inline aclError aclrtGetDevice(int32_t* device) {
  ++ascify_test_get_device_calls;
  if (ascify_test_get_device_status == ACL_SUCCESS) {
    *device = ascify_test_device;
  }
  return ascify_test_get_device_status;
}

inline aclError aclrtGetDeviceInfo(
    uint32_t device, aclrtDevAttr attribute, int64_t* value) {
  (void)device;
  ++ascify_test_get_device_info_calls;
  if (attribute == ACL_DEV_ATTR_VECTOR_CORE_NUM) {
    if (ascify_test_vector_core_info_status == ACL_SUCCESS) {
      *value = ascify_test_vector_core_count;
    }
    return ascify_test_vector_core_info_status;
  }
  if (static_cast<unsigned int>(attribute) == 203U) {
    if (ascify_test_max_thread_info_status == ACL_SUCCESS) {
      *value = ascify_test_max_threads_per_core;
    }
    return ascify_test_max_thread_info_status;
  }
  return ACL_ERROR_RT_PARAM_INVALID;
}
#else
inline aclError aclrtGetDevice(int32_t* device) {
  *device = 0;
  return ACL_SUCCESS;
}

inline aclError aclrtGetDeviceInfo(uint32_t, aclrtDevAttr, int64_t* value) {
  *value = 1024;
  return ACL_SUCCESS;
}
#endif

#if defined(ASCIFY_TEST_ACL_CONTROLLABLE_DEVICE_QUERY)
extern const char* ascify_test_soc_name;
inline const char* aclrtGetSocName() { return ascify_test_soc_name; }
#else
inline const char* aclrtGetSocName() { return "Ascify test device"; }
#endif

// Symbol-specific behavior is checked separately with controllable SDK calls.
inline aclError aclrtGetSymbolSize(const void*, size_t*) {
  return ACL_ERROR_FEATURE_UNSUPPORTED;
}
inline aclError aclrtMemcpyToSymbol(const void*, const void*, size_t, size_t,
                                    aclrtMemcpyKind) {
  return ACL_ERROR_FEATURE_UNSUPPORTED;
}

inline aclError aclrtPeekAtLastError(aclrtLastErrLevel) {
  return ACL_SUCCESS;
}

#if defined(ASCIFY_TEST_ACL_CONTROLLABLE_RUNTIME_LIFECYCLE)
inline aclError aclrtCtxGetCurrentDefaultStream(aclrtStream* stream) {
  ++ascify_test_get_default_stream_calls;
  *stream = ascify_test_default_stream;
  return ACL_SUCCESS;
}
#elif !defined(ASCIFY_TEST_PUBLIC_85_ACL)
inline aclError aclrtCtxGetCurrentDefaultStream(aclrtStream* stream) {
  *stream = reinterpret_cast<aclrtStream>(2);
  return ACL_SUCCESS;
}
#endif

inline aclError aclrtCreateStream(aclrtStream* stream) {
#if defined(ASCIFY_TEST_ACL_CONTROLLABLE_RUNTIME_LIFECYCLE)
  ++ascify_test_create_stream_calls;
#endif
  *stream = reinterpret_cast<aclrtStream>(1);
  return ACL_SUCCESS;
}

inline aclError aclrtDestroyStream(aclrtStream) { return ACL_SUCCESS; }
inline aclError aclrtSynchronizeStream(aclrtStream) { return ACL_SUCCESS; }

inline aclError aclrtCreateEvent(aclrtEvent* event) {
  *event = reinterpret_cast<aclrtEvent>(1);
  return ACL_SUCCESS;
}

inline aclError aclrtDestroyEvent(aclrtEvent) { return ACL_SUCCESS; }
inline aclError aclrtRecordEvent(aclrtEvent, aclrtStream stream) {
#if defined(ASCIFY_TEST_ACL_CONTROLLABLE_RUNTIME_LIFECYCLE)
  ++ascify_test_record_event_calls;
  ascify_test_recorded_event_stream = stream;
#else
  (void)stream;
#endif
  return ACL_SUCCESS;
}
inline aclError aclrtSynchronizeEvent(aclrtEvent) { return ACL_SUCCESS; }
inline aclError aclrtEventElapsedTime(float* milliseconds, aclrtEvent,
                                      aclrtEvent) {
  *milliseconds = 1.0f;
  return ACL_SUCCESS;
}

#endif
