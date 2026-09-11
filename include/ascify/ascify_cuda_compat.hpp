#ifndef ASCIFY_ASCIFY_CUDA_COMPAT_HPP
#define ASCIFY_ASCIFY_CUDA_COMPAT_HPP

#include <algorithm>
#include <atomic>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <acl/acl.h>

#if defined(ASCIFY_TEST_CONTROLLABLE_EXIT_REGISTRATION)
extern int ascify_test_register_exit_cleanup(void (*callback)());
#endif

// CANN exposes two incompatible SIMT header families on supported targets.
// Prefer the public 8.5 aggregate header when it is present; otherwise use the
// 9.1 beta3 legacy split headers.  The compatibility layer itself does not use
// bfloat16, so it must not make every translated source depend on an optional
// asc_bf16.h header.
#if defined(__has_include)
#if __has_include(<simt_api/kernel_simt_intf.h>)
#include <simt_api/kernel_simt_intf.h>
#define ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85 1
#elif __has_include(<simt_api/device_functions.h>) && \
      __has_include(<simt_api/device_sync_functions.h>) && \
      __has_include(<simt_api/device_warp_functions.h>) && \
      __has_include(<simt_api/math_functions.h>)
#include <simt_api/device_functions.h>
#include <simt_api/device_sync_functions.h>
#include <simt_api/device_warp_functions.h>
#include <simt_api/math_functions.h>
#define ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3 1
#if __has_include(<simt_api/vector_functions.h>)
// Coherent 9.1 beta3 already owns CUDA-compatible vector types and make_*
// constructors. Include that native surface instead of redeclaring helpers
// such as make_float4 in Ascify and risking an ABI/ODR conflict.
#include <simt_api/vector_functions.h>
#define ASCIFY_SIMT_LEGACY_HAS_VECTOR_CONSTRUCTORS 1
#endif
#if __has_include(<simt_api/device_atomic_functions.h>)
#include <simt_api/device_atomic_functions.h>
#define ASCIFY_SIMT_LEGACY_HAS_GM_ATOMICS 1
#endif
#else
#error "Ascify requires either public kernel_simt_intf.h or the legacy SIMT split headers"
#endif
#else
// Older beta3 CCEC revisions predate __has_include but use the split family.
#include <simt_api/device_functions.h>
#include <simt_api/device_sync_functions.h>
#include <simt_api/device_warp_functions.h>
#include <simt_api/math_functions.h>
#define ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3 1
#endif

#if defined(ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3)
#include <ascify/masked_warp_compat.hpp>
#endif

// CUDA spells alignment as `__align__(N)`. Keeping the argument in a macro
// preserves both dependent expressions such as sizeof(T) and the declaration
// position selected by the source.
#ifndef ASCIFY_ALIGN
#define ASCIFY_ALIGN(bytes) __attribute__((aligned(bytes)))
#endif

#ifndef ASCIFY_FORCEINLINE
#define ASCIFY_FORCEINLINE inline __attribute__((always_inline))
#endif

// Kernel qualifiers differ across the admitted compiler/header families.
// Public 8.5 requires an explicit aicore attribute, while coherent 9.1 beta3
// rejects the simultaneous device/global attributes. Generated kernels use
// this macro so the active target headers select the valid spelling.
#ifndef ASCIFY_GLOBAL
#if defined(ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85)
#define ASCIFY_GLOBAL __global__ __aicore__
#else
#define ASCIFY_GLOBAL __global__
#endif
#endif

namespace ascify {

inline constexpr unsigned int cudaStreamDefault = 0x00U;
inline constexpr unsigned int cudaStreamNonBlocking = 0x01U;

namespace detail {

class RuntimeLock {
 public:
  void lock() noexcept {
    while (locked_.test_and_set(std::memory_order_acquire)) {}
  }
  void unlock() noexcept { locked_.clear(std::memory_order_release); }

 private:
  std::atomic_flag locked_ = ATOMIC_FLAG_INIT;
};

class RuntimeLockGuard {
 public:
  explicit RuntimeLockGuard(RuntimeLock& lock) noexcept : lock_(lock) {
    lock_.lock();
  }
  ~RuntimeLockGuard() { lock_.unlock(); }
  RuntimeLockGuard(const RuntimeLockGuard&) = delete;
  RuntimeLockGuard& operator=(const RuntimeLockGuard&) = delete;

 private:
  RuntimeLock& lock_;
};

template<typename>
struct dependentFalse : std::false_type {};

}  // namespace detail

// Both admitted SIMT header families expose full-warp reductions for these
// scalar types, but under incompatible names.  Keep the supported overload set
// explicit so an unverified scalar type fails during target compilation.
__aicore__ ASCIFY_FORCEINLINE float warp_reduce_add(float value) {
#if defined(ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85)
  return AscendC::Simt::WarpReduceAddSync(value);
#else
  return asc_reduce_add(value);
#endif
}

__aicore__ ASCIFY_FORCEINLINE int32_t warp_reduce_add(int32_t value) {
#if defined(ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85)
  return AscendC::Simt::WarpReduceAddSync(value);
#else
  return asc_reduce_add(value);
#endif
}

__aicore__ ASCIFY_FORCEINLINE uint32_t warp_reduce_add(uint32_t value) {
#if defined(ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85)
  return AscendC::Simt::WarpReduceAddSync(value);
#else
  return asc_reduce_add(value);
#endif
}

__aicore__ ASCIFY_FORCEINLINE float warp_reduce_max(float value) {
#if defined(ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85)
  return AscendC::Simt::WarpReduceMaxSync(value);
#else
  return asc_reduce_max(value);
#endif
}

__aicore__ ASCIFY_FORCEINLINE int32_t warp_reduce_max(int32_t value) {
#if defined(ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85)
  return AscendC::Simt::WarpReduceMaxSync(value);
#else
  return asc_reduce_max(value);
#endif
}

__aicore__ ASCIFY_FORCEINLINE uint32_t warp_reduce_max(uint32_t value) {
#if defined(ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85)
  return AscendC::Simt::WarpReduceMaxSync(value);
#else
  return asc_reduce_max(value);
#endif
}

__aicore__ ASCIFY_FORCEINLINE float warp_reduce_min(float value) {
#if defined(ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85)
  return AscendC::Simt::WarpReduceMinSync(value);
#else
  return asc_reduce_min(value);
#endif
}

__aicore__ ASCIFY_FORCEINLINE int32_t warp_reduce_min(int32_t value) {
#if defined(ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85)
  return AscendC::Simt::WarpReduceMinSync(value);
#else
  return asc_reduce_min(value);
#endif
}

__aicore__ ASCIFY_FORCEINLINE uint32_t warp_reduce_min(uint32_t value) {
#if defined(ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85)
  return AscendC::Simt::WarpReduceMinSync(value);
#else
  return asc_reduce_min(value);
#endif
}

// These wrappers are internal targets for AST-proven global atomic calls.
// The frontend admits only addresses rooted at a parameter of the current
// CUDA global kernel. The explicit cast supplies the address-space type that
// coherent 9.1 device_atomic_functions.h requires; it must never be used as a
// name-only rewrite for shared/local pointers.
#if defined(ASCIFY_SIMT_LEGACY_HAS_GM_ATOMICS)
#define ASCIFY_DEFINE_GLOBAL_ATOMIC_BINARY(wrapper, target)                  \
  __aicore__ ASCIFY_FORCEINLINE int32_t wrapper(                             \
      int32_t* address, int32_t value) {                                     \
    return target(reinterpret_cast<__gm__ int32_t*>(address), value);        \
  }                                                                          \
  __aicore__ ASCIFY_FORCEINLINE uint32_t wrapper(                            \
      uint32_t* address, uint32_t value) {                                   \
    return target(reinterpret_cast<__gm__ uint32_t*>(address), value);       \
  }

#define ASCIFY_DEFINE_GLOBAL_ATOMIC_UNSIGNED(wrapper, target)                \
  __aicore__ ASCIFY_FORCEINLINE uint32_t wrapper(                            \
      uint32_t* address, uint32_t value) {                                   \
    return target(reinterpret_cast<__gm__ uint32_t*>(address), value);       \
  }

#define ASCIFY_DEFINE_GLOBAL_ATOMIC_CAS(wrapper, target)                     \
  __aicore__ ASCIFY_FORCEINLINE int32_t wrapper(                             \
      int32_t* address, int32_t compare, int32_t value) {                    \
    return target(                                                           \
        reinterpret_cast<__gm__ int32_t*>(address), compare, value);         \
  }                                                                          \
  __aicore__ ASCIFY_FORCEINLINE uint32_t wrapper(                            \
      uint32_t* address, uint32_t compare, uint32_t value) {                 \
    return target(                                                           \
        reinterpret_cast<__gm__ uint32_t*>(address), compare, value);        \
  }

ASCIFY_DEFINE_GLOBAL_ATOMIC_BINARY(atomic_add_global, asc_atomic_add)
ASCIFY_DEFINE_GLOBAL_ATOMIC_BINARY(atomic_sub_global, asc_atomic_sub)
ASCIFY_DEFINE_GLOBAL_ATOMIC_BINARY(atomic_exch_global, asc_atomic_exch)
ASCIFY_DEFINE_GLOBAL_ATOMIC_BINARY(atomic_max_global, asc_atomic_max)
ASCIFY_DEFINE_GLOBAL_ATOMIC_BINARY(atomic_min_global, asc_atomic_min)
ASCIFY_DEFINE_GLOBAL_ATOMIC_UNSIGNED(atomic_inc_global, asc_atomic_inc)
ASCIFY_DEFINE_GLOBAL_ATOMIC_UNSIGNED(atomic_dec_global, asc_atomic_dec)
ASCIFY_DEFINE_GLOBAL_ATOMIC_CAS(atomic_cas_global, asc_atomic_cas)
ASCIFY_DEFINE_GLOBAL_ATOMIC_BINARY(atomic_and_global, asc_atomic_and)
ASCIFY_DEFINE_GLOBAL_ATOMIC_BINARY(atomic_or_global, asc_atomic_or)
ASCIFY_DEFINE_GLOBAL_ATOMIC_BINARY(atomic_xor_global, asc_atomic_xor)

#undef ASCIFY_DEFINE_GLOBAL_ATOMIC_BINARY
#undef ASCIFY_DEFINE_GLOBAL_ATOMIC_UNSIGNED
#undef ASCIFY_DEFINE_GLOBAL_ATOMIC_CAS
#else
#define ASCIFY_DEFINE_UNAVAILABLE_GLOBAL_ATOMIC_BINARY(wrapper)              \
  template<typename T, typename V>                                           \
  __aicore__ ASCIFY_FORCEINLINE T wrapper(T*, V) {                           \
    static_assert(detail::dependentFalse<T>::value,                          \
                  "Ascify: no admitted global atomic API in this SIMT "     \
                  "header family");                                        \
    return T{};                                                              \
  }

#define ASCIFY_DEFINE_UNAVAILABLE_GLOBAL_ATOMIC_CAS(wrapper)                 \
  template<typename T, typename C, typename V>                               \
  __aicore__ ASCIFY_FORCEINLINE T wrapper(T*, C, V) {                        \
    static_assert(detail::dependentFalse<T>::value,                          \
                  "Ascify: no admitted global atomic API in this SIMT "     \
                  "header family");                                        \
    return T{};                                                              \
  }

ASCIFY_DEFINE_UNAVAILABLE_GLOBAL_ATOMIC_BINARY(atomic_add_global)
ASCIFY_DEFINE_UNAVAILABLE_GLOBAL_ATOMIC_BINARY(atomic_sub_global)
ASCIFY_DEFINE_UNAVAILABLE_GLOBAL_ATOMIC_BINARY(atomic_exch_global)
ASCIFY_DEFINE_UNAVAILABLE_GLOBAL_ATOMIC_BINARY(atomic_max_global)
ASCIFY_DEFINE_UNAVAILABLE_GLOBAL_ATOMIC_BINARY(atomic_min_global)
ASCIFY_DEFINE_UNAVAILABLE_GLOBAL_ATOMIC_BINARY(atomic_inc_global)
ASCIFY_DEFINE_UNAVAILABLE_GLOBAL_ATOMIC_BINARY(atomic_dec_global)
ASCIFY_DEFINE_UNAVAILABLE_GLOBAL_ATOMIC_CAS(atomic_cas_global)
ASCIFY_DEFINE_UNAVAILABLE_GLOBAL_ATOMIC_BINARY(atomic_and_global)
ASCIFY_DEFINE_UNAVAILABLE_GLOBAL_ATOMIC_BINARY(atomic_or_global)
ASCIFY_DEFINE_UNAVAILABLE_GLOBAL_ATOMIC_BINARY(atomic_xor_global)

#undef ASCIFY_DEFINE_UNAVAILABLE_GLOBAL_ATOMIC_BINARY
#undef ASCIFY_DEFINE_UNAVAILABLE_GLOBAL_ATOMIC_CAS
#endif

struct cudaFuncAttributes {
  size_t sharedSizeBytes = 0;
};

enum cudaFuncAttribute {
  cudaFuncAttributeMaxDynamicSharedMemorySize = 0
};

// Public 8.5 acl_rt.h stops at VECTOR_CORE_NUM=201 even though its frozen RTS
// ABI exposes 202/203/204.  Coherent 9.1 beta3 additionally exposes the block
// limit as 209 (with a plural THREADS spelling).  Isolate those ABI values from
// header spelling drift.  These are intentionally `const`, not `constexpr`:
// some host Clang revisions reject an out-of-enumerator constexpr enum cast.
// The active runtime remains authoritative and returns an error for a value it
// does not support; no portability claim is made for arbitrary CANN releases.
inline const aclrtDevAttr cudaDevAttrWarpSize =
    static_cast<aclrtDevAttr>(202U);
inline const aclrtDevAttr cudaDevAttrMaxThreadsPerVectorCore =
    static_cast<aclrtDevAttr>(203U);
inline const aclrtDevAttr cudaDevAttrLocalMemoryPerVectorCore =
    static_cast<aclrtDevAttr>(204U);
inline const aclrtDevAttr cudaDevAttrMaxThreadsPerBlock =
    static_cast<aclrtDevAttr>(209U);

namespace detail {

// CUDA initializes its runtime and a device's primary context lazily. ACL has
// two separate process/device operations instead: aclInit is process-wide and
// aclrtSetDevice creates the calling thread's default context. Keep that state
// in one header-owned manager so every translated translation unit observes
// the same initialization attempt and cleanup ownership.
class RuntimeManager {
 public:
  static constexpr size_t kTrackedDeviceCapacity = 64U;

  RuntimeManager() = default;
  ~RuntimeManager() { shutdown(); }
  RuntimeManager(const RuntimeManager&) = delete;
  RuntimeManager& operator=(const RuntimeManager&) = delete;

  aclError ensureInitialized() {
    RuntimeLockGuard guard(lock_);
    if (shutdown_) { return ACL_ERROR_REPEAT_FINALIZE; }
    if (initialization_attempted_) { return initialization_status_; }

    initialization_attempted_ = true;
    const aclError status = aclInit(nullptr);
    if (status == ACL_SUCCESS) {
      owns_acl_initialization_ = true;
    } else if (status == ACL_ERROR_REPEAT_INITIALIZE) {
      // An embedding application initialized ACL first. Its ownership also
      // includes finalization; Ascify only owns device bindings it creates.
      owns_acl_initialization_ = false;
    } else {
      initialization_status_ = status;
      return initialization_status_;
    }

    // The inline RuntimeManager object is constructed before main, so its
    // ordinary destructor is registered before aclInit creates CANN's own
    // process-exit state. Running ACL teardown from that late destructor can
    // therefore observe already-destroyed CANN internals. Register a dedicated
    // callback after aclInit for initialization-only paths. Device binding
    // can construct additional SDK exit state; bindDevice registers a later
    // callback after that operation. Both callbacks are idempotent.
    exit_cleanup_manager_ = this;
    if (registerExitCleanup(&RuntimeManager::runExitCleanup) == 0) {
      cleanup_registered_ = true;
      initialization_status_ = ACL_SUCCESS;
    } else {
      exit_cleanup_manager_ = nullptr;
      const bool finalize = owns_acl_initialization_;
      owns_acl_initialization_ = false;
      const aclError rollback = finalize ? aclFinalize() : ACL_SUCCESS;
      initialization_status_ =
          rollback == ACL_SUCCESS ? ACL_ERROR_BAD_ALLOC : rollback;
    }
    return initialization_status_;
  }

  aclError bindDevice(int32_t device) {
    const aclError initialized = ensureInitialized();
    if (initialized != ACL_SUCCESS) { return initialized; }

    RuntimeLockGuard guard(lock_);
    if (shutdown_) { return ACL_ERROR_REPEAT_FINALIZE; }
    if (initialization_status_ != ACL_SUCCESS) {
      return initialization_status_;
    }
    const aclError status = aclrtSetDevice(device);
    if (!device_cleanup_registered_) {
      // CANN's profiling singleton can be constructed lazily by SetDevice,
      // after our aclInit callback was registered. Register after the call so
      // LIFO teardown resets devices before that singleton is destroyed.
      // A failed attempt can also construct partial SDK state: protect it,
      // but do not assume that the next successful attempt creates no more.
      if (registerExitCleanup(&RuntimeManager::runExitCleanup) == 0) {
        device_cleanup_registered_ = status == ACL_SUCCESS;
      } else {
        // Before the first successful registration no binding is tracked.
        // Roll back now, while SDK state is live, and cache the failure. The
        // older callback must not retry ACL teardown after SDK destruction.
        const aclError reset =
            status == ACL_SUCCESS ? aclrtResetDevice(device) : ACL_SUCCESS;
        const bool finalize = owns_acl_initialization_;
        owns_acl_initialization_ = false;
        const aclError rollback = finalize ? aclFinalize() : ACL_SUCCESS;
        initialization_status_ =
            reset != ACL_SUCCESS ? reset :
            rollback != ACL_SUCCESS ? rollback :
            status != ACL_SUCCESS ? status : ACL_ERROR_BAD_ALLOC;
        return initialization_status_;
      }
    }
    if (status != ACL_SUCCESS) { return status; }
    for (size_t index = 0; index < active_device_count_; ++index) {
      if (active_devices_[index] == device) { return ACL_SUCCESS; }
    }

    if (active_device_count_ == kTrackedDeviceCapacity) {
      // Keep the registry allocation-free on CCEC's host path. If the fixed
      // admission bound is exhausted, undo the just-created ACL binding and
      // fail closed instead of leaking a device that teardown cannot track.
      const aclError rollback = aclrtResetDevice(device);
      return rollback == ACL_SUCCESS ? ACL_ERROR_BAD_ALLOC : rollback;
    }
    active_devices_[active_device_count_++] = device;
    return ACL_SUCCESS;
  }

  aclError resetDevice(int32_t device) {
    const aclError initialized = ensureInitialized();
    if (initialized != ACL_SUCCESS) { return initialized; }

    RuntimeLockGuard guard(lock_);
    if (shutdown_) { return ACL_ERROR_REPEAT_FINALIZE; }
    if (initialization_status_ != ACL_SUCCESS) {
      return initialization_status_;
    }
    const aclError status = aclrtResetDevice(device);
    if (status == ACL_SUCCESS) {
      for (size_t index = 0; index < active_device_count_; ++index) {
        if (active_devices_[index] == device) {
          for (size_t next = index + 1; next < active_device_count_; ++next) {
            active_devices_[next - 1] = active_devices_[next];
          }
          --active_device_count_;
          break;
        }
      }
    }
    return status;
  }

  void shutdown() noexcept {
    int32_t devices[kTrackedDeviceCapacity] = {};
    size_t device_count = 0;
    bool finalize = false;
    {
      RuntimeLockGuard guard(lock_);
      if (shutdown_) { return; }
      shutdown_ = true;
      device_count = active_device_count_;
      for (size_t index = 0; index < device_count; ++index) {
        devices[index] = active_devices_[index];
      }
      active_device_count_ = 0;
      finalize = owns_acl_initialization_;
      owns_acl_initialization_ = false;
      cleanup_registered_ = false;
    }

    // ACL documents one reset as sufficient even if a device was selected
    // repeatedly. Reset in reverse acquisition order, then finalize only an
    // aclInit that this compatibility layer successfully owned.
    while (device_count != 0) {
      --device_count;
      (void)aclrtResetDevice(devices[device_count]);
    }
    if (finalize) { (void)aclFinalize(); }
 }

 private:
  static int registerExitCleanup(void (*callback)()) noexcept {
#if defined(ASCIFY_TEST_CONTROLLABLE_EXIT_REGISTRATION)
    return ::ascify_test_register_exit_cleanup(callback);
#else
    return ::atexit(callback);
#endif
  }

  static void runExitCleanup() noexcept {
    RuntimeManager* const manager = exit_cleanup_manager_;
    if (manager != nullptr) { manager->shutdown(); }
  }

  inline static RuntimeManager* exit_cleanup_manager_ = nullptr;
  RuntimeLock lock_;
  bool initialization_attempted_ = false;
  bool owns_acl_initialization_ = false;
  bool shutdown_ = false;
  bool cleanup_registered_ = false;
  bool device_cleanup_registered_ = false;
  aclError initialization_status_ = ACL_SUCCESS;
  int32_t active_devices_[kTrackedDeviceCapacity] = {};
  size_t active_device_count_ = 0;
};

// This inline C++17 object is defined before declarations in every translated
// source that includes the compatibility header. Its destructor therefore
// provides process-exit cleanup without requiring a sample-specific main.
inline RuntimeManager runtime_manager;

inline int32_t& selectedDeviceForThread() {
  static thread_local int32_t device = 0;
  return device;
}

inline aclError& pendingLifecycleErrorForThread() {
  static thread_local aclError status = ACL_SUCCESS;
  return status;
}

inline aclError rememberLifecycleError(aclError status) {
  if (status != ACL_SUCCESS) { pendingLifecycleErrorForThread() = status; }
  return status;
}

inline bool isMissingCurrentContext(aclError status) {
  return status == ACL_ERROR_RT_CONTEXT_NULL ||
         status == ACL_ERROR_RT_NO_DEVICE;
}

inline aclError ensureCurrentDevice() {
  aclError status = runtime_manager.ensureInitialized();
  if (status != ACL_SUCCESS) { return rememberLifecycleError(status); }

  int32_t current = -1;
  status = aclrtGetDevice(&current);
  if (status == ACL_SUCCESS) {
    selectedDeviceForThread() = current;
    return ACL_SUCCESS;
  }
  if (!isMissingCurrentContext(status)) {
    return rememberLifecycleError(status);
  }

  status = runtime_manager.bindDevice(selectedDeviceForThread());
  return rememberLifecycleError(status);
}

}  // namespace detail

// A target compiler can use this before a raw <<<...>>> launch when the
// source contains no earlier CUDA Runtime API. Runtime shims below call the
// same function lazily, so ordinary CUDA programs need no explicit lifecycle
// calls of their own.
inline aclError cudaRuntimeEnsureReady() {
  return detail::ensureCurrentDevice();
}

inline aclError cudaSetDevice(int device) {
  if (device < 0) { return ACL_ERROR_RT_PARAM_INVALID; }
  const aclError status = detail::runtime_manager.bindDevice(device);
  if (status == ACL_SUCCESS) { detail::selectedDeviceForThread() = device; }
  return detail::rememberLifecycleError(status);
}

inline aclError cudaGetDevice(int* device) {
  if (device == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }

  int32_t current = -1;
  const aclError status = aclrtGetDevice(&current);
  if (status == ACL_SUCCESS) {
    *device = static_cast<int>(current);
    detail::selectedDeviceForThread() = current;
  }
  return detail::rememberLifecycleError(status);
}

inline aclError cudaGetDeviceCount(int* count) {
  if (count == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  *count = 0;
  const aclError initialized = detail::runtime_manager.ensureInitialized();
  if (initialized != ACL_SUCCESS) {
    return detail::rememberLifecycleError(initialized);
  }

  uint32_t raw_count = 0;
  const aclError status = aclrtGetDeviceCount(&raw_count);
  if (status == ACL_SUCCESS) { *count = static_cast<int>(raw_count); }
  return detail::rememberLifecycleError(status);
}

inline aclError cudaDeviceSynchronize() {
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  return aclrtSynchronizeDevice();
}

inline aclError cudaDeviceReset() {
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }

  int32_t current = -1;
  aclError status = aclrtGetDevice(&current);
  if (status != ACL_SUCCESS) {
    return detail::rememberLifecycleError(status);
  }
  detail::selectedDeviceForThread() = current;
  status = detail::runtime_manager.resetDevice(current);
  return detail::rememberLifecycleError(status);
}

// CUDA and ACL expose similar memory operations with observably different
// signatures.  Keep the CUDA call contract in generated code and adapt it in
// one typed shim instead of relying on token-level renames that compile only
// when both APIs happen to have the same arity.
inline aclError cudaMalloc(void** device_pointer, size_t bytes) {
  if (device_pointer == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  *device_pointer = nullptr;
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  return aclrtMalloc(device_pointer, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
}

template<typename T>
inline aclError cudaMalloc(T** device_pointer, size_t bytes) {
  if (device_pointer == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  *device_pointer = nullptr;
  void* raw_pointer = nullptr;
  const aclError status = cudaMalloc(&raw_pointer, bytes);
  if (status == ACL_SUCCESS) { *device_pointer = static_cast<T*>(raw_pointer); }
  return status;
}

inline aclError cudaFree(void* device_pointer) {
  // CUDA specifies cudaFree(0) as a successful no-op. Do not delegate that
  // case to runtimes that reject null pointers.
  if (device_pointer == nullptr) { return ACL_SUCCESS; }
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  return aclrtFree(device_pointer);
}

inline aclError cudaMallocHost(void** host_pointer, size_t bytes) {
  if (host_pointer == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  *host_pointer = nullptr;
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  return aclrtMallocHost(host_pointer, bytes);
}

template<typename T>
inline aclError cudaMallocHost(T** host_pointer, size_t bytes) {
  if (host_pointer == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  *host_pointer = nullptr;
  void* raw_pointer = nullptr;
  const aclError status = cudaMallocHost(&raw_pointer, bytes);
  if (status == ACL_SUCCESS) { *host_pointer = static_cast<T*>(raw_pointer); }
  return status;
}

inline aclError cudaFreeHost(void* host_pointer) {
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  return aclrtFreeHost(host_pointer);
}

namespace detail {

inline bool isSupportedMemcpyKind(aclrtMemcpyKind kind) {
  return kind == ACL_MEMCPY_HOST_TO_HOST ||
         kind == ACL_MEMCPY_HOST_TO_DEVICE ||
         kind == ACL_MEMCPY_DEVICE_TO_HOST ||
         kind == ACL_MEMCPY_DEVICE_TO_DEVICE;
}

inline bool nonnullMemoryRange(const void* pointer, size_t bytes) {
  return bytes == 0 || pointer != nullptr;
}

}  // namespace detail

inline aclError cudaMemcpy(void* destination, const void* source, size_t bytes,
                           aclrtMemcpyKind kind) {
  if (!detail::isSupportedMemcpyKind(kind)) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  if (bytes == 0) { return ACL_SUCCESS; }
  if (!detail::nonnullMemoryRange(destination, bytes) ||
      !detail::nonnullMemoryRange(source, bytes)) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  return aclrtMemcpy(destination, bytes, source, bytes, kind);
}

inline aclError cudaMemcpyAsync(void* destination, const void* source,
                                size_t bytes, aclrtMemcpyKind kind,
                                aclrtStream stream) {
  if (!detail::isSupportedMemcpyKind(kind)) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  if (bytes == 0) { return ACL_SUCCESS; }
  if (!detail::nonnullMemoryRange(destination, bytes) ||
      !detail::nonnullMemoryRange(source, bytes)) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  return aclrtMemcpyAsync(destination, bytes, source, bytes, kind, stream);
}

inline aclError cudaMemset(void* destination, int value, size_t bytes) {
  if (bytes == 0) { return ACL_SUCCESS; }
  if (!detail::nonnullMemoryRange(destination, bytes)) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  return aclrtMemset(destination, bytes, value, bytes);
}

inline aclError cudaMemsetAsync(void* destination, int value, size_t bytes,
                                aclrtStream stream) {
  if (bytes == 0) { return ACL_SUCCESS; }
  if (!detail::nonnullMemoryRange(destination, bytes)) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  return aclrtMemsetAsync(destination, bytes, value, bytes, stream);
}

inline aclError cudaStreamCreate(aclrtStream* stream) {
  if (stream == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  *stream = nullptr;
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  return aclrtCreateStream(stream);
}

inline aclError cudaStreamCreateWithFlags(
    aclrtStream* stream, unsigned int flags) {
  if (stream == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  *stream = nullptr;
  if (flags == cudaStreamNonBlocking) {
    // CUDA's bit 0 suppresses implicit synchronization with the legacy
    // default stream. ACL's public bit 0 instead requests FAST_LAUNCH, so
    // forwarding the numeric value would silently change semantics. The
    // coherent public ACL surface exposes no proven equivalent.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  }
  if (flags != cudaStreamDefault) { return ACL_ERROR_RT_PARAM_INVALID; }
  return cudaStreamCreate(stream);
}

inline aclError cudaStreamDestroy(aclrtStream stream) {
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  return aclrtDestroyStream(stream);
}

inline aclError cudaStreamSynchronize(aclrtStream stream) {
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  return aclrtSynchronizeStream(stream);
}

inline aclError cudaEventCreate(aclrtEvent* event) {
  if (event == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  *event = nullptr;
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  return aclrtCreateEvent(event);
}

inline aclError cudaEventDestroy(aclrtEvent event) {
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  return aclrtDestroyEvent(event);
}

inline aclError cudaEventRecord(
    aclrtEvent event, aclrtStream stream = nullptr) {
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  if (stream == nullptr) {
#if defined(ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3) && \
    defined(ACL_MAJOR_VERSION) && defined(ACL_MINOR_VERSION) && \
    ((ACL_MAJOR_VERSION > 1) || \
     (ACL_MAJOR_VERSION == 1 && ACL_MINOR_VERSION >= 17))
    // This query is present in the coherent 9.1 beta3 / ACL 1.17 surface.
    // Public 8.5 has not been admitted for this overload, so it must not name
    // an API that may be absent from that header family.
    const aclError status = aclrtCtxGetCurrentDefaultStream(&stream);
    if (status != ACL_SUCCESS) { return status; }
#else
    return ACL_ERROR_FEATURE_UNSUPPORTED;
#endif
  }
  return aclrtRecordEvent(event, stream);
}

inline aclError cudaEventSynchronize(aclrtEvent event) {
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  return aclrtSynchronizeEvent(event);
}

inline aclError cudaEventElapsedTime(float* milliseconds, aclrtEvent start,
                                     aclrtEvent end) {
  if (milliseconds == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  return aclrtEventElapsedTime(milliseconds, start, end);
}

inline aclError cudaGetLastError() {
  aclError& pending = detail::pendingLifecycleErrorForThread();
  if (pending != ACL_SUCCESS) {
    const aclError status = pending;
    pending = ACL_SUCCESS;
    return status;
  }
  return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}

inline const char* cudaGetErrorString(aclError error) {
  if (error == ACL_SUCCESS) { return "ACL_SUCCESS"; }
  // ACL reports the current thread's recent diagnostic rather than formatting
  // an arbitrary historic status code.  Preserve the CUDA call signature, but
  // make the weaker diagnostic contract explicit and never return null.
  const char* message = aclGetRecentErrMsg();
  return message == nullptr ? "ACL runtime error" : message;
}

// Internal targets for the two admitted NVIDIA cuda-samples helper macros.
// The frontend rewrites only macro expansions whose definition comes from a
// fingerprinted helper_cuda.h; these functions are not name-only substitutes
// for user code. Passing status by value guarantees that a CUDA/ACL call with
// side effects is evaluated exactly once.
inline void sampleCheckCudaErrors(aclError status, const char* expression,
                                  const char* file, int line) {
  if (status == ACL_SUCCESS) { return; }
  fprintf(stderr,
          "%s(%d): Ascify CUDA sample helper error: code=%d (%s) \"%s\"\n",
          file == nullptr ? "" : file, line, static_cast<int>(status),
          cudaGetErrorString(status), expression == nullptr ? "" : expression);
  exit(EXIT_FAILURE);
}

// cudaGetLastError is the real Ascify last-error surface. It first consumes
// and clears an Ascify lifecycle error, otherwise delegates to ACL's consuming
// thread-level query. Never implement this helper with cudaPeekAtLastError:
// NVIDIA getLastCudaError has consume/reset semantics.
inline void sampleGetLastCudaError(const char* message, const char* file,
                                   int line) {
  const aclError status = cudaGetLastError();
  if (status == ACL_SUCCESS) { return; }
  fprintf(stderr,
          "%s(%d): Ascify CUDA sample helper last error: %s: code=%d (%s)\n",
          file == nullptr ? "" : file, line,
          message == nullptr ? "" : message,
          static_cast<int>(status), cudaGetErrorString(status));
  exit(EXIT_FAILURE);
}

// Internal target for a provenance- and AST-proven NVIDIA cuda-samples
// findCudaDevice call. Ascify intentionally does not reproduce CUDA's
// architecture-specific GFLOPS ranking: the admitted runtime contract is one
// visible logical device, selected as logical device zero.
namespace detail {

[[noreturn]] inline void sampleFindCudaDeviceFailure(const char* reason,
                                                     aclError status) {
  if (status == ACL_SUCCESS) {
    fprintf(stderr, "Ascify CUDA sample device selection failed: %s\n",
            reason == nullptr ? "invalid device selection" : reason);
  } else {
    fprintf(stderr,
            "Ascify CUDA sample device selection failed: %s: code=%d (%s)\n",
            reason == nullptr ? "runtime failure" : reason,
            static_cast<int>(status), cudaGetErrorString(status));
  }
  exit(EXIT_FAILURE);
}

inline bool sampleFindCudaDeviceArgumentIsSelector(const char* argument) {
  if (argument == nullptr) { return false; }
  while (*argument == '-') { ++argument; }
  static constexpr char selector[] = "device";
  for (size_t index = 0; index + 1 < sizeof(selector); ++index) {
    const char actual = argument[index];
    if (actual == '\0') { return false; }
    const char expected = selector[index];
    const char folded = actual >= 'A' && actual <= 'Z'
                            ? static_cast<char>(actual - 'A' + 'a')
                            : actual;
    if (folded != expected) { return false; }
  }
  const char suffix = argument[sizeof(selector) - 1];
  return suffix == '\0' || suffix == '=';
}

inline bool sampleFindCudaDeviceArgumentIsExactZero(const char* argument) {
  if (argument == nullptr) { return false; }
  static constexpr char admitted[] = "--device=0";
  size_t index = 0;
  for (; admitted[index] != '\0'; ++index) {
    if (argument[index] != admitted[index]) { return false; }
  }
  return argument[index] == '\0';
}

}  // namespace detail

inline int sampleFindCudaDevice(int argc, const char** argv) {
  if (argc < 0 || (argc > 0 && argv == nullptr)) {
    detail::sampleFindCudaDeviceFailure("invalid argc/argv", ACL_SUCCESS);
  }

  bool explicit_zero = false;
  for (int index = 1; index < argc; ++index) {
    const char* const argument = argv[index];
    if (argument == nullptr) {
      detail::sampleFindCudaDeviceFailure(
          "null command-line argument", ACL_SUCCESS);
    }
    if (!detail::sampleFindCudaDeviceArgumentIsSelector(argument)) {
      continue;
    }
    const bool is_exact_zero =
        detail::sampleFindCudaDeviceArgumentIsExactZero(argument);
    if (!is_exact_zero || explicit_zero) {
      detail::sampleFindCudaDeviceFailure(
          "only one explicit --device=0 selector is supported", ACL_SUCCESS);
    }
    explicit_zero = true;
  }

  int count = 0;
  const aclError count_status = cudaGetDeviceCount(&count);
  if (count_status != ACL_SUCCESS) {
    detail::sampleFindCudaDeviceFailure(
        "device-count query failed", count_status);
  }
  if (count != 1) {
    detail::sampleFindCudaDeviceFailure(
        "exactly one visible logical device is required", ACL_SUCCESS);
  }

  const aclError bind_status = cudaSetDevice(0);
  if (bind_status != ACL_SUCCESS) {
    detail::sampleFindCudaDeviceFailure(
        "binding logical device 0 failed", bind_status);
  }
  return 0;
}

#ifndef ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS
#define ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS(expression)                  \
  ::ascify::sampleCheckCudaErrors(                                          \
      (expression), #expression, __FILE__, __LINE__)
#endif

#ifndef ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR
#define ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR(message)                   \
  ::ascify::sampleGetLastCudaError((message), __FILE__, __LINE__)
#endif

// Keep the CUDA argument order and result width at the translated call site.
// aclrtGetDeviceInfo instead takes device first and writes an int64_t.
inline aclError cudaDeviceGetAttribute(int* value, aclrtDevAttr attribute, int device) {
  if (value == nullptr || device < 0) { return ACL_ERROR_RT_PARAM_INVALID; }

  const aclError initialized = detail::runtime_manager.ensureInitialized();
  if (initialized != ACL_SUCCESS) {
    return detail::rememberLifecycleError(initialized);
  }

  int64_t raw_value = 0;
  const aclError status =
      aclrtGetDeviceInfo(static_cast<uint32_t>(device), attribute, &raw_value);
  if (status == ACL_SUCCESS) { *value = static_cast<int>(raw_value); }
  return status;
}

// A deliberately partial property surface. CUDA SM versions and CUDA core
// counts have no portable meaning on an NPU; code reading those fields must
// fail target compilation instead of receiving invented NVIDIA capabilities.
struct cudaDeviceProp {
  char name[256];
  int multiProcessorCount;
};

inline aclError cudaGetDeviceProperties(cudaDeviceProp* properties, int device) {
  if (properties == nullptr || device < 0) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  const aclError initialized = detail::runtime_manager.ensureInitialized();
  if (initialized != ACL_SUCCESS) {
    return detail::rememberLifecycleError(initialized);
  }
  // GetSocName describes the bound device. Never silently return the current
  // device's name for a different requested device, or rebind the caller.
  int32_t current = -1;
  aclError status = aclrtGetDevice(&current);
  if (status != ACL_SUCCESS) { return status; }
  if (current != device) { return ACL_ERROR_FEATURE_UNSUPPORTED; }
  int64_t cores = 0;
  status = aclrtGetDeviceInfo(static_cast<uint32_t>(device),
                            ACL_DEV_ATTR_VECTOR_CORE_NUM, &cores);
  if (status != ACL_SUCCESS) { return status; }
  if (cores <= 0 || cores > INT_MAX) { return ACL_ERROR_FEATURE_UNSUPPORTED; }
  const char* const soc_name = aclrtGetSocName();
  if (soc_name == nullptr || soc_name[0] == '\0') {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  }
  cudaDeviceProp result = {};
  size_t index = 0;
  while (index + 1 < sizeof(result.name) && soc_name[index] != '\0') {
    result.name[index] = soc_name[index];
    ++index;
  }
  result.multiProcessorCount = static_cast<int>(cores);
  *properties = result;
  return ACL_SUCCESS;
}

// CUDA's API has no arguments. CANN requires the error scope explicitly.
inline aclError cudaPeekAtLastError() {
  const aclError pending = detail::pendingLifecycleErrorForThread();
  if (pending != ACL_SUCCESS) { return pending; }
  return aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
}

// CANN has no direct CUDA occupancy counterpart. This approximate estimate
// reports active blocks per vector core, bounded by threads and dynamic UB.
// Static UB, register pressure, and a hardware resident-block cap are not
// exposed by the beta3 runtime and therefore are not represented here.
// The kernel parameter is intentionally retained so existing CUDA call sites,
// including specialized kernel templates, remain source compatible.
template<typename Kernel>
inline aclError cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    int* active_blocks, Kernel, int block_size, size_t dynamic_ub_bytes) {
  if (active_blocks == nullptr || block_size <= 0) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }

  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }

  int32_t device = 0;
  aclError status = aclrtGetDevice(&device);
  if (status != ACL_SUCCESS) { return status; }

  int64_t max_threads = 0;
  status = aclrtGetDeviceInfo(static_cast<uint32_t>(device),
                              cudaDevAttrMaxThreadsPerVectorCore,
                              &max_threads);
  if (status != ACL_SUCCESS) { return status; }

  int64_t blocks = max_threads / block_size;
  if (dynamic_ub_bytes != 0) {
    int64_t ub_bytes = 0;
    status = aclrtGetDeviceInfo(static_cast<uint32_t>(device),
                                cudaDevAttrLocalMemoryPerVectorCore,
                                &ub_bytes);
    if (status != ACL_SUCCESS) { return status; }
    const int64_t ub_blocks =
        ub_bytes > 0 ? ub_bytes / static_cast<int64_t>(dynamic_ub_bytes) : 0;
    if (ub_blocks < blocks) { blocks = ub_blocks; }
  }

  if (blocks < 0) { blocks = 0; }
  if (blocks > INT_MAX) { blocks = INT_MAX; }
  *active_blocks = static_cast<int>(blocks);
  return ACL_SUCCESS;
}

// beta3 does not expose CUDA-style per-kernel static/dynamic UB metadata.
// Static shared usage is conservatively reported as zero; setting the dynamic
// limit is a successful no-op because launch-time UB size remains authoritative.
template<typename Kernel>
inline aclError cudaFuncGetAttributes(cudaFuncAttributes* attributes, Kernel) {
  if (attributes == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  attributes->sharedSizeBytes = 0;
  return ACL_SUCCESS;
}

template<typename Kernel>
inline aclError cudaFuncSetAttribute(Kernel, cudaFuncAttribute, int) {
  return ACL_SUCCESS;
}

namespace detail {

__aicore__ ASCIFY_FORCEINLINE void requireFullWarpMask(uint32_t mask) {
  // Both admitted target families expose unmasked shuffle operations. Trap
  // instead of silently widening a CUDA partial-mask collective.
  if (mask != UINT32_MAX) {
#if defined(ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3)
    reject_device_contract();
#else
    __builtin_trap();
#endif
  }
}

__aicore__ ASCIFY_FORCEINLINE void requireSupportedWarpWidth(int width) {
  if (width != 2 && width != 4 && width != 8 && width != 16 && width != 32) {
#if defined(ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3)
    reject_device_contract();
#else
    __builtin_trap();
#endif
  }
}

}  // namespace detail

__aicore__ ASCIFY_FORCEINLINE void syncwarp(uint32_t mask = UINT32_MAX) {
  // Both target families execute a warp in lockstep and expose no warp-only
  // barrier. A full-warp synchronization is a no-op; a partial mask is not.
  detail::requireFullWarpMask(mask);
}

__aicore__ ASCIFY_FORCEINLINE int any_sync(uint32_t mask, int predicate) {
  detail::requireFullWarpMask(mask);
  return warp_reduce_max(int32_t{predicate != 0});
}

__aicore__ ASCIFY_FORCEINLINE int all_sync(uint32_t mask, int predicate) {
  detail::requireFullWarpMask(mask);
  return warp_reduce_min(int32_t{predicate != 0});
}

template<typename Dependency = void>
__aicore__ ASCIFY_FORCEINLINE uint32_t ballot_sync(uint32_t mask, int predicate) {
#if defined(ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3)
  return detail::converged_ballot_sync(mask, predicate);
#else
  static_assert(detail::dependentFalse<Dependency>::value,
                "Ascify ballot_sync requires the verified legacy active-mask API");
  __builtin_unreachable();
#endif
}

template<typename T>
__aicore__ ASCIFY_FORCEINLINE T shfl_sync(uint32_t mask, T value, int source_lane,
                                          int width = 32) {
  detail::requireFullWarpMask(mask);
  detail::requireSupportedWarpWidth(width);
#if defined(ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85)
  return AscendC::Simt::WarpShflSync(value, source_lane, width);
#else
  return asc_shfl(value, source_lane, width);
#endif
}

template<typename T>
__aicore__ ASCIFY_FORCEINLINE T shfl_up_sync(uint32_t mask, T value, unsigned int delta,
                                             int width = 32) {
  detail::requireFullWarpMask(mask);
  detail::requireSupportedWarpWidth(width);
#if defined(ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85)
  return AscendC::Simt::WarpShflUpSync(value, delta, width);
#else
  return asc_shfl_up(value, delta, width);
#endif
}

template<typename T>
__aicore__ ASCIFY_FORCEINLINE T shfl_down_sync(uint32_t mask, T value, unsigned int delta,
                                               int width = 32) {
#if defined(ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3)
  if constexpr (std::is_same<T, int32_t>::value ||
                std::is_same<T, uint32_t>::value || std::is_same<T, float>::value) {
    // Keep existing full-mask calls on their original native path. Restricted
    // partial-mask calls need an already converged cohort; other scalar types
    // retain the earlier full-mask-only contract and native overload set.
    if (mask != UINT32_MAX || width == 1) {
      return detail::converged_shfl_down_sync(mask, value, delta, width);
    }
  }
#endif
  detail::requireFullWarpMask(mask);
  detail::requireSupportedWarpWidth(width);
#if defined(ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85)
  return AscendC::Simt::WarpShflDownSync(value, delta, width);
#else
  return asc_shfl_down(value, delta, width);
#endif
}

template<typename T>
__aicore__ ASCIFY_FORCEINLINE T shfl_xor_sync(uint32_t mask, T value, int lane_mask,
                                              int width = 32) {
  detail::requireFullWarpMask(mask);
  detail::requireSupportedWarpWidth(width);
#if defined(ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85)
  return AscendC::Simt::WarpShflXorSync(value, lane_mask, width);
#else
  return asc_shfl_xor(value, lane_mask, width);
#endif
}

__aicore__ ASCIFY_FORCEINLINE float fdividef(float numerator, float denominator) {
#if defined(ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85)
  // Public 8.5 does not export a fast-divide helper. Preserve a valid device
  // expression instead of naming a legacy-only, non-dependent symbol.
  return numerator / denominator;
#else
  return ::fdividef(numerator, denominator);
#endif
}

__aicore__ ASCIFY_FORCEINLINE void syncthreads() {
#if defined(ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85)
  AscendC::Simt::ThreadBarrier();
#else
  asc_syncthreads();
#endif
}

template<typename Predicate>
__aicore__ ASCIFY_FORCEINLINE int syncthreads_and(Predicate) {
  static_assert(detail::dependentFalse<Predicate>::value,
                "Ascify: no admitted block-wide syncthreads_and API");
  return 0;
}

template<typename Predicate>
__aicore__ ASCIFY_FORCEINLINE int syncthreads_or(Predicate) {
  static_assert(detail::dependentFalse<Predicate>::value,
                "Ascify: no admitted block-wide syncthreads_or API");
  return 0;
}

template<typename Predicate>
__aicore__ ASCIFY_FORCEINLINE int syncthreads_count(Predicate) {
  static_assert(detail::dependentFalse<Predicate>::value,
                "Ascify: no admitted block-wide syncthreads_count API");
  return 0;
}

__aicore__ ASCIFY_FORCEINLINE void threadfence_block() {
#if defined(ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85)
  // Public 8.5 exposes one device fence. Its stronger scope satisfies the
  // ordering required by CUDA's block-scoped fence.
  AscendC::Simt::ThreadFence();
#else
  asc_threadfence_block();
#endif
}

__aicore__ ASCIFY_FORCEINLINE void threadfence() {
#if defined(ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85)
  AscendC::Simt::ThreadFence();
#else
  asc_threadfence();
#endif
}

template<typename Unsupported = void>
__aicore__ ASCIFY_FORCEINLINE void threadfence_system() {
  static_assert(detail::dependentFalse<Unsupported>::value,
                "Ascify: no admitted CUDA system-scope fence mapping");
}

}  // namespace ascify

#include "symbol_compat.hpp"

#endif  // ASCIFY_ASCIFY_CUDA_COMPAT_HPP
