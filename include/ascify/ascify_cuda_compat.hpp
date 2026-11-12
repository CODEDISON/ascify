#ifndef ASCIFY_ASCIFY_CUDA_COMPAT_HPP
#define ASCIFY_ASCIFY_CUDA_COMPAT_HPP

// Host-thread creation for the stream-callback worker, declared before any
// standard include so the C signatures cannot interact with later headers.


#include <algorithm>
#include <atomic>
#include <fstream>
#include <sstream>
#include <string>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include <pthread.h>

#include <acl/acl.h>
#include <acl/acl_prof.h>


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
#include <ascify/device_memory_compat.hpp>

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

#include <ascify/scalar64_compat.hpp>
#include <ascify/vmm_compat.hpp>

namespace ascify {

// Internal host target for the two exact CUDA int/unsigned-int min orders.
// Value parameters perform CUDA's unsigned conversion before the comparison
// and evaluate each original argument once, independently of builtin versions.
inline unsigned int cuda_host_min_unsigned(unsigned int a, unsigned int b) {
  return a < b ? a : b;
}

// Internal target for one proven nonvolatile CUDA lane-ID instruction.
// CUDA numbers threads with x varying fastest, including 2D and 3D blocks;
// each block starts its own 32-thread warps. Keep lookup in this protected
// header so call-site locals cannot shadow the SIMT builtins.
__aicore__ ASCIFY_FORCEINLINE unsigned int cuda_lane_id() {
  return (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) & 31u;
}

namespace detail {
#if defined(ASCIFY_SCALAR64_VERIFIED_TARGET)
// A separate source contract for a proved, discarded add/down-shuffle step.
// CUDA defines an out-of-width source as this lane, but does not define a
// read from an inactive source inside that width. Complete only the latter
// case by omitting its addition; never change the ordinary shuffle interface.
__aicore__ ASCIFY_FORCEINLINE bool masked_accumulation_source_present(
    uint32_t mask, unsigned int delta) {
  const unsigned int lane = ::ascify::cuda_lane_id();
  // Short circuit before either an overflowing addition or a shift by32.
  return delta >= 32U || delta >= 32U - lane ||
         (mask & (uint32_t(1) << (lane + delta))) != 0;
}

__aicore__ ASCIFY_FORCEINLINE unsigned int masked_accumulation_delta(
    uint32_t mask, unsigned int delta) {
  require_converged_warp_mask(mask);
  // All participating lanes still shuffle. A missing neighbor selects this
  // lane's valid register instead; the outer operation then discards it.
  // Large deltas select self without corrupting native packed control bits.
  return delta >= 32U || !masked_accumulation_source_present(mask, delta)
      ? 0U : delta;
}

template<class T>
__aicore__ ASCIFY_FORCEINLINE void scalar_masked_add_assign_rne(
    T& left, T right, uint32_t mask, unsigned int original_delta) {
  static_assert(scalar_add_type<T>::value && !std::is_volatile<T>::value,
                "masked accumulation requires a nonvolatile int/float/double");
  require_converged_warp_mask(mask);
  // Keep defined self additions and the original rounding order. Skipping
  // absent neighbors also avoids introducing synthetic +0 for an all--0 sum.
  if (masked_accumulation_source_present(mask, original_delta))
    scalar_add_assign_rne<T>(left, right);
}
#else
// This new contract does not inherit support on an unverified header family
// or architecture from the older generic shuffle overloads.
unsigned int masked_accumulation_delta(uint32_t, unsigned int) = delete;
template<class T>
void scalar_masked_add_assign_rne(T&, T, uint32_t, unsigned int) = delete;
#endif
} // namespace detail

// Internal target for the exact unsigned quad-byte SAD accumulate instruction.
// Extract by value, independently of host byte order. Each difference is at
// most 255; unsigned accumulation preserves the low 32 bits, including carry.
__aicore__ ASCIFY_FORCEINLINE unsigned int packed_u8_sad4_add(
    unsigned int a, unsigned int b, unsigned int result) {
  static_assert(sizeof(unsigned int) * CHAR_BIT == 32,
                "packed byte SAD requires 32-bit unsigned int");
  for (unsigned int index = 0; index < 32; index += 8) {
    const unsigned int left = (a >> index) & 255u;
    const unsigned int right = (b >> index) & 255u;
    result += left >= right ? left - right : right - left;
  }
  return result;
}

// Internal target for a resolved CUDA float frexp(float, int*) overload.
// Keep native lookup in this protected header, before caller declarations.
#if defined(ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3)
__aicore__ inline float cuda_frexp_float(float value, int* exponent) {
  return ::frexpf(value, exponent);
}
#else
// The public header family has not been admitted for this new operation.
float cuda_frexp_float(float, int*) = delete;
#endif

inline constexpr unsigned int cudaStreamDefault = 0x00U;
inline constexpr unsigned int cudaStreamNonBlocking = 0x01U;
inline constexpr unsigned int cudaEventWaitDefault = 0x00U;
inline constexpr unsigned int cudaEventWaitExternal = 0x01U;
// CUDA-domain device-assert status code (cuda_runtime_api value 719). The
// acl runtime has no assert-specific error result and never produces this
// code; the constant is retained with its original value so original CUDA
// equality checks (error == cudaErrorAssert) compile and evaluate exactly
// against CUDA-domain codes. It makes no claim that device assert failures
// are surfaced through cudaGetLastError().
inline constexpr aclError cudaErrorAssert = static_cast<aclError>(719);
// Retain CUDA's reserved handle values for explicit rejection where an ACL
// default stream cannot provide the corresponding implicit-stream ordering.
inline const aclrtStream cudaStreamLegacy =
    reinterpret_cast<aclrtStream>(static_cast<uintptr_t>(1));
inline const aclrtStream cudaStreamPerThread =
    reinterpret_cast<aclrtStream>(static_cast<uintptr_t>(2));

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

// The public pure-SIMT overload preserves generic GM/UB pointer semantics.
// This target is selected only for exact SDK int/uint atomicAdd calls rooted
// at a pure device helper parameter. It performs no address-space cast.
#if defined(ASCIFY_SIMT_LEGACY_HAS_GM_ATOMICS) && \
    defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)
__aicore__ ASCIFY_FORCEINLINE int32_t atomic_add_generic(
    int32_t *address, int32_t value) {
  return asc_atomic_add(address, value);
}
__aicore__ ASCIFY_FORCEINLINE uint32_t atomic_add_generic(
    uint32_t *address, uint32_t value) {
  return asc_atomic_add(address, value);
}
#else
template<typename T, typename V>
__aicore__ ASCIFY_FORCEINLINE T atomic_add_generic(T *, V) {
  static_assert(detail::dependentFalse<T>::value,
                "Ascify: generic atomic add requires the admitted pure-SIMT API");
  return T{};
}
#endif

// Exact SDK int/uint exchange on a generic pointer/reference or shared scalar.
// Keep the address space inferred by pure SIMT; do not force references to GM.
#if defined(ASCIFY_SIMT_LEGACY_HAS_GM_ATOMICS) && \
    defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)
__aicore__ ASCIFY_FORCEINLINE int32_t atomic_exch_generic(
    int32_t *address, int32_t value) {
  return asc_atomic_exch(address, value);
}
__aicore__ ASCIFY_FORCEINLINE uint32_t atomic_exch_generic(
    uint32_t *address, uint32_t value) {
  return asc_atomic_exch(address, value);
}
#else
template<typename T, typename V>
__aicore__ ASCIFY_FORCEINLINE T atomic_exch_generic(T *, V) {
  static_assert(detail::dependentFalse<T>::value,
                "Ascify: generic exchange requires the admitted pure-SIMT API");
  return T{};
}
#endif

// Exact SDK int/uint maximum on a generic pointer/reference or shared scalar,
// selected for calls proven rooted at a pure device helper parameter. The
// address space stays whatever pure SIMT infers (GM or UB).
#if defined(ASCIFY_SIMT_LEGACY_HAS_GM_ATOMICS) && \
    defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)
__aicore__ ASCIFY_FORCEINLINE int32_t atomic_max_generic(
    int32_t *address, int32_t value) {
  return asc_atomic_max(address, value);
}
__aicore__ ASCIFY_FORCEINLINE uint32_t atomic_max_generic(
    uint32_t *address, uint32_t value) {
  return asc_atomic_max(address, value);
}
#else
template<typename T, typename V>
__aicore__ ASCIFY_FORCEINLINE T atomic_max_generic(T *, V) {
  static_assert(detail::dependentFalse<T>::value,
                "Ascify: generic maximum requires the admitted pure-SIMT API");
  return T{};
}
#endif

// These wrappers are internal targets for AST-proven global atomic calls.
// The frontend admits addresses rooted at a parameter of the current CUDA
// global kernel, or atomicInc on a separately proved same-module uint32 device
// counter with a committed symbol resolver. The explicit cast supplies the address-space type that
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
  // Published only by the converter's proven max-only query path below.
  int maxThreadsPerBlock = 0;
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

// Optional bounded texture adapter. It installs one immutable callback table
// only after texture allocation is requested. Existing programs incur no ACL
// calls or texture-state creation through these hooks.
struct RuntimeTextureHooks {
  aclError (*checkPending)();
  aclError (*freeAllocation)(void *, bool *);
  void (*afterReset)(int32_t);
};
inline std::atomic<const RuntimeTextureHooks *> runtimeTextureHooks{nullptr};

// Requested extents for ordinary cudaMalloc allocations, recorded before any
// texture exists. An exhausted table only makes a future texture view
// unprovable; it must never turn a successful ordinary allocation into failure.
// Lock order is allocation registry -> texture registry. Texture callbacks
// never acquire this lock, and no ACL API returns an inferred rounded extent.
struct RuntimeAllocation {
  void *pointer = nullptr;
  size_t bytes = 0;
  int32_t device = -1;
  void *context = nullptr;
};
struct RuntimeAllocationRegistry {
  RuntimeLock lock;
  RuntimeAllocation records[128] = {};
};
inline RuntimeAllocationRegistry &runtimeAllocationRegistry() {
  static RuntimeAllocationRegistry registry;
  return registry;
}
inline RuntimeAllocation *runtimeAllocationLocked(const void *pointer) {
  if (!pointer) return nullptr;
  for (auto &record : runtimeAllocationRegistry().records)
    if (record.pointer == pointer) return &record;
  return nullptr;
}

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
    // CUDA's primary context is shared with every host thread; the native
    // rtSetDevice blocks when many sample threads re-bind concurrently. Bind
    // once, then hand additional threads the same context explicitly.
    for (size_t index = 0; index < active_device_count_; ++index) {
      if (active_devices_[index].device == device) {
        if (active_devices_[index].context != nullptr) {
          return aclrtSetCurrentContext(active_devices_[index].context);
        }
        // No explicit context exists (creation unsupported): fall back to a
        // plain re-bind, which is the original per-thread behaviour.
        return aclrtSetDevice(device);
      }
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
      if (active_devices_[index].device == device) { return ACL_SUCCESS; }
    }

    if (active_device_count_ == kTrackedDeviceCapacity) {
      // Keep the registry allocation-free on CCEC's host path. If the fixed
      // admission bound is exhausted, undo the just-created ACL binding and
      // fail closed instead of leaking a device that teardown cannot track.
      const aclError rollback = aclrtResetDevice(device);
      return rollback == ACL_SUCCESS ? ACL_ERROR_BAD_ALLOC : rollback;
    }
    aclrtContext created = nullptr;
    if (aclrtCreateContext(&created, device) != ACL_SUCCESS) { created = nullptr; }
    active_devices_[active_device_count_++] = {device, created};
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
        if (active_devices_[index].device == device) {
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
        devices[index] = active_devices_[index].device;
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
  struct BoundDevice { int32_t device = -1; aclrtContext context = nullptr; };
  BoundDevice active_devices_[kTrackedDeviceCapacity] = {};
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

namespace detail {

// CUDA controls an attached tool for the current context; ACL's collector is
// process-wide. Admit an explicit, owned single-context collection session,
// never silently replace CUDA profiling with successful no-ops. The caller
// opts in with a writable ASCIFY_PROFILE_DIR and must not use another ACL
// profiler or another active context in that process.
class ProfilerManager {
 public:
  aclError start() {
    const char* path = ::getenv("ASCIFY_PROFILE_DIR");
    if (path == nullptr || *path == '\0') {
      return ACL_ERROR_FEATURE_UNSUPPORTED;
    }
    const aclError ready = ensureCurrentDevice();
    if (ready != ACL_SUCCESS) { return ready; }
    aclrtContext context = nullptr;
    int32_t device = -1;
    aclError status = aclrtGetCurrentContext(&context);
    if (status != ACL_SUCCESS) { return status; }
    if (context == nullptr) { return ACL_ERROR_RT_CONTEXT_NULL; }
    status = aclrtGetDevice(&device);
    if (status != ACL_SUCCESS) { return status; }
    if (device < 0) { return ACL_ERROR_RT_PARAM_INVALID; }

    RuntimeLockGuard guard(lock_);
    if (initialized_) {
      if (context != context_ || device != device_) {
        return ACL_ERROR_FEATURE_UNSUPPORTED;
      }
      // A failed stop/destroy/finalize keeps its resource state for Stop to
      // retry. Do not report that a partially stopped session is active.
      if (failure_ != ACL_SUCCESS) { return failure_; }
      return active_ ? ACL_SUCCESS : ACL_ERROR_PROFILING_FAILURE;
    }
    size_t length = 0;
    while (path[length] != '\0') { ++length; }
    status = aclprofInit(path, length);
    if (status != ACL_SUCCESS) { return status; }
    initialized_ = true;
    context_ = context;
    device_ = device;
    uint32_t devices[] = {static_cast<uint32_t>(device)};
    config_ = aclprofCreateConfig(devices, 1, ACL_AICORE_NONE, nullptr,
                                 ACL_PROF_ACL_API | ACL_PROF_TASK_TIME);
    if (config_ == nullptr) {
      return rollbackStart(ACL_ERROR_PROFILING_FAILURE);
    }
    status = aclprofStart(config_);
    if (status != ACL_SUCCESS) { return rollbackStart(status); }
    active_ = true;
    // Register after Start, which may create additional SDK exit state.
    // This callback runs before the RuntimeManager resets/finalizes ACL.
    if (!cleanup_registered_) {
      exit_cleanup_profiler_ = this;
#if defined(ASCIFY_TEST_CONTROLLABLE_PROFILER_EXIT_REGISTRATION)
      const int registration = ::ascify_test_register_profiler_exit_cleanup(
          &ProfilerManager::runExitCleanup);
#else
      const int registration = ::atexit(&ProfilerManager::runExitCleanup);
#endif
      if (registration != 0) { return rollbackStart(ACL_ERROR_BAD_ALLOC); }
      cleanup_registered_ = true;
    }
    failure_ = ACL_SUCCESS;
    return ACL_SUCCESS;
  }

  aclError stop() {
    RuntimeLockGuard guard(lock_);
    if (!initialized_) {
      const char* path = ::getenv("ASCIFY_PROFILE_DIR");
      return path != nullptr && *path != '\0' ? ACL_SUCCESS :
                                               ACL_ERROR_FEATURE_UNSUPPORTED;
    }
    aclrtContext context = nullptr;
    int32_t device = -1;
    aclError status = aclrtGetCurrentContext(&context);
    if (status != ACL_SUCCESS) { return status; }
    status = aclrtGetDevice(&device);
    if (status != ACL_SUCCESS) { return status; }
    if (context != context_ || device != device_) {
      return ACL_ERROR_FEATURE_UNSUPPORTED;
    }
    return cleanup();
  }

 private:
  // On failure retain the last owned resource, never destroy a configuration
  // still collecting or finalize somebody else's profiler. A later Stop
  // retries exactly the incomplete stage and returns its ACL error unchanged.
  aclError cleanup() {
    if (active_) {
      const aclError status = aclprofStop(config_);
      if (status != ACL_SUCCESS) { return failure_ = status; }
      active_ = false;
    }
    if (config_ != nullptr) {
      const aclError status = aclprofDestroyConfig(config_);
      if (status != ACL_SUCCESS) { return failure_ = status; }
      config_ = nullptr;
    }
    if (initialized_) {
      const aclError status = aclprofFinalize();
      if (status != ACL_SUCCESS) { return failure_ = status; }
      initialized_ = false;
    }
    context_ = nullptr;
    device_ = -1;
    return failure_ = ACL_SUCCESS;
  }

  aclError rollbackStart(aclError original) {
    // The Start error is primary. If rollback also fails, cleanup's state and
    // status remain available to the next Stop; neither failure becomes success.
    (void)cleanup();
    return original;
  }

  static void runExitCleanup() noexcept {
    ProfilerManager* const profiler = exit_cleanup_profiler_;
    if (profiler == nullptr) { return; }
    RuntimeLockGuard guard(profiler->lock_);
    // Process-exit cleanup has no return channel. Explicit Stop is required
    // when an application needs to observe all collection/flush errors.
    (void)profiler->cleanup();
  }

  inline static ProfilerManager* exit_cleanup_profiler_ = nullptr;
  RuntimeLock lock_;
  aclprofConfig* config_ = nullptr;
  aclrtContext context_ = nullptr;
  int32_t device_ = -1;
  aclError failure_ = ACL_SUCCESS;
  bool initialized_ = false;
  bool active_ = false;
  bool cleanup_registered_ = false;
};

inline ProfilerManager profiler_manager;

}  // namespace detail

inline aclError cudaProfilerStart() {
  return detail::rememberLifecycleError(detail::profiler_manager.start());
}

inline aclError cudaProfilerStop() {
  return detail::rememberLifecycleError(detail::profiler_manager.stop());
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

// Peer queries name both devices and must not select a different current
// device. ACL writes int32_t, while CUDA exposes int: commit that result only
// on success, including when a failing backend partially writes its output.
inline aclError cudaDeviceCanAccessPeer(int* can_access_peer, int device,
                                       int peer_device) {
  if (can_access_peer == nullptr || device < 0 || peer_device < 0) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  const aclError initialized = detail::runtime_manager.ensureInitialized();
  if (initialized != ACL_SUCCESS) {
    return detail::rememberLifecycleError(initialized);
  }
  int32_t result = 0;
  const aclError status = aclrtDeviceCanAccessPeer(
      &result, static_cast<int32_t>(device),
      static_cast<int32_t>(peer_device));
  if (status == ACL_SUCCESS) { *can_access_peer = static_cast<int>(result); }
  return detail::rememberLifecycleError(status);
}

// Enable/disable apply to the calling thread's current device. Reuse the
// ordinary lazy-context path, including the selected device after a reset;
// do not switch to the peer device or invent an access-capability result.
inline aclError cudaDeviceEnablePeerAccess(int peer_device,
                                          unsigned int flags = 0) {
  if (peer_device < 0 || flags != 0) { return ACL_ERROR_RT_PARAM_INVALID; }
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  return detail::rememberLifecycleError(aclrtDeviceEnablePeerAccess(
      static_cast<int32_t>(peer_device), 0U));
}

inline aclError cudaDeviceDisablePeerAccess(int peer_device) {
  if (peer_device < 0) { return ACL_ERROR_RT_PARAM_INVALID; }
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  return detail::rememberLifecycleError(
      aclrtDeviceDisablePeerAccess(static_cast<int32_t>(peer_device)));
}

// ---- P2P software path: launch-time peer staging (D2D mirror) -----------
// Platform fact, verified on device: a SIMT kernel cannot dereference
// peer-device global memory. Even after aclrtDeviceEnablePeerAccess the
// access traps with AI-core error 334 ("SIMT accesses an invalid GM address
// or the cross-device memory access times out"), while host-side
// cross-device aclrtMemcpy D2D works (10 GB/s measured between the sample's
// device pair). CUDA's contract -- a kernel receives a peer pointer and
// dereferences it -- is therefore preserved in software: at kernel launch,
// every packed kernel-argument slot that points into a registered
// allocation owned by another device is mirrored into a local allocation on
// the launching device (aclrtMemcpy D2D, peer -> mirror, one refresh per
// launch) and the kernel receives the mirror address. This is the
// read-through form of the contract: kernels that read peer memory observe
// current peer data at every launch. A kernel writing through a peer
// pointer is not admitted by this path; such writes cannot be reflected
// back without a post-launch writeback that no admitted sample performs.
//
// Interception point: ccec lowers <<<...>>> into the weak
// __cce_rtKernelLaunchWithFlagV2 / __cce_rtKernelLaunch stubs, which
// forward the packed kernel-argument blob to the runtime ABI symbols
// rtKernelLaunchWithFlagV2 / rtKernelLaunch. The shim definitions below
// use asm labels to occupy those exact ABI symbols, so the stub's calls
// resolve here at static link time. Hidden visibility confines the
// interposition to this executable's own launches (runtime-internal
// callers keep resolving to libruntime), and weak linkage keeps
// multi-translation-unit programs linkable. The real runtime entry is
// reached through dlsym(RTLD_NEXT) and every launch without cross-device
// pointer arguments passes through completely unchanged.
namespace detail {

// Reentrancy guard: staging performs aclrtMemcpy copies on this thread;
// if any copy path itself reaches a launch site again, that nested launch
// must observe the unmodified argument blob.
inline int32_t &peerStagingDepth() {
  static thread_local int32_t depth = 0;
  return depth;
}

// Software mirror of one cross-device allocation: `mirror` lives on
// `mirror_device` and holds a refresh-on-launch copy of the `bytes` at
// `source` (owned by `owner`).
struct PeerMirrorRecord {
  void *source = nullptr;
  size_t bytes = 0;
  int32_t owner = -1;
  int32_t mirror_device = -1;
  void *mirror = nullptr;
};
struct PeerMirrorTable {
  RuntimeLock lock;
  PeerMirrorRecord records[32] = {};
};
inline PeerMirrorTable &peerMirrorTable() {
  static PeerMirrorTable table;
  return table;
}

// ABI mirror of the runtime's rtArgsEx_t (pkg_inc
// runtime/rt_external_kernel.h; the 40-byte size is cross-checked against
// the ccec stub's argsInfo memset). Only args/argsSize are read; every
// other field is opaque and forwarded untouched.
struct AscifyRtArgsEx {
  void *args;
  void *hostInputInfoPtr;
  uint32_t argsSize;
  uint32_t tilingAddrOffset;
  uint32_t tilingDataOffset;
  uint16_t hostInputInfoNum;
  uint8_t hasTiling;
  uint8_t isNoNeedH2DCopy;
  uint8_t reserved[4];
};

// Escape hatch for the software path: ASCIFY_RT_PEER_STAGING=0 restores
// direct kernel-argument passing (and with it the platform's error 334 on
// true peer dereferences).
inline bool peerStagingEnabled() {
  static std::atomic<int> state{0};
  int cached = state.load(std::memory_order_acquire);
  if (cached == 0) {
    const char *requested = getenv("ASCIFY_RT_PEER_STAGING");
    cached = (requested != nullptr && requested[0] == '0' &&
              requested[1] == '\0') ? -1 : 1;
    state.store(cached, std::memory_order_release);
  }
  return cached > 0;
}

// Refresh the mirror for one registered allocation on the launching
// device, allocating it on first use. Returns the mirror base or nullptr
// when the mirror cannot be provided; nullptr leaves the original peer
// address in the blob (and the launch to fail exactly like the native
// path would).
inline void *stagedPeerMirrorLocked(const void *source, size_t bytes,
                                    int32_t owner, int32_t mirror_device) {
  for (auto &record : peerMirrorTable().records) {
    if (record.mirror != nullptr && record.source == source &&
        record.mirror_device == mirror_device) {
      return record.mirror;
    }
  }
  for (auto &record : peerMirrorTable().records) {
    if (record.mirror != nullptr) continue;
    void *fresh = nullptr;
    if (aclrtMalloc(&fresh, bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
        fresh != nullptr) {
      record = {const_cast<void *>(source), bytes, owner, mirror_device, fresh};
      return fresh;
    }
    return nullptr;
  }
  return nullptr;
}

// Scan a packed kernel-argument blob for pointer slots that fall inside a
// registered allocation owned by another device; stage those allocations
// into local mirrors and rewrite the slots in place. The generated launch
// site rebuilds the blob for every launch, so in-place rewriting is safe
// and every launch refreshes its own mirror content.
inline void stagePeerKernelArguments(void *blob, uint32_t blob_bytes) {
  if (blob == nullptr || blob_bytes < sizeof(void *)) return;
  if (peerStagingDepth() != 0 || !peerStagingEnabled()) return;
  int32_t current = -1;
  if (aclrtGetDevice(&current) != ACL_SUCCESS || current < 0) return;
  size_t bytes = blob_bytes > 4096U ? 4096U : blob_bytes;

  // Mirror staging is only meaningful when some registered allocation
  // lives on another device; single-device programs never pay the scan.
  bool peer_allocation_present = false;
  {
    RuntimeLockGuard guard(runtimeAllocationRegistry().lock);
    for (const auto &record : runtimeAllocationRegistry().records) {
      if (record.pointer != nullptr && record.device >= 0 &&
          record.device != current) {
        peer_allocation_present = true;
        break;
      }
    }
  }
  if (!peer_allocation_present) return;
  // Order the staging copy after default-stream work on the launching
  // device, the same blocking-copy policy the cudaMemcpy adapter uses.
  if (aclrtSynchronizeStream(nullptr) != ACL_SUCCESS) return;

  // The mirror refresh is host-issued while CUDA's cross-device dataflow is
  // device-side (streamWaitEvent edges the host copy cannot observe). Drain
  // each staged source's owner device once per launch before refreshing its
  // mirror; otherwise an async write still in flight on the owner (e.g. a
  // pool H2D upload on the owner's stream) can lose the race and leave the
  // mirror stale. A failed drain keeps the original peer address, exactly
  // like a mirror that could not be provided.
  int32_t drained_owners[8] = {-1, -1, -1, -1, -1, -1, -1, -1};

  for (size_t offset = 0; offset + sizeof(void *) <= bytes;
       offset += sizeof(void *)) {
    char *const slot = static_cast<char *>(blob) + offset;
    const uintptr_t value = *reinterpret_cast<const uintptr_t *>(slot);
    // Device allocation addresses are wide virtual addresses; scalar and
    // packed-dimension slots never reach that range.
    if (value < 0x10000U) continue;
    const void *source = nullptr;
    size_t source_bytes = 0;
    int32_t owner = -1;
    {
      RuntimeLockGuard guard(runtimeAllocationRegistry().lock);
      for (const auto &record : runtimeAllocationRegistry().records) {
        if (record.pointer == nullptr || record.device < 0 ||
            record.device == current) {
          continue;
        }
        const uintptr_t low = reinterpret_cast<uintptr_t>(record.pointer);
        const uintptr_t high =
            low + static_cast<uintptr_t>(record.bytes);
        if (value >= low && value < high) {
          source = record.pointer;
          source_bytes = record.bytes;
          owner = record.device;
          break;
        }
      }
    }
    if (source == nullptr) continue;

    bool owner_drained = false;
    for (const int32_t drained : drained_owners) {
      if (drained == owner) { owner_drained = true; break; }
    }
    if (!owner_drained) {
      aclrtContext saved_context = nullptr;
      const bool have_saved_context =
          aclrtGetCurrentContext(&saved_context) == ACL_SUCCESS &&
          saved_context != nullptr;
      const bool switched = owner != current;
      if (switched) (void)aclrtSetDevice(owner);
      const bool drained_ok = aclrtSynchronizeDevice() == ACL_SUCCESS;
      if (switched) {
        if (have_saved_context) {
          (void)aclrtSetCurrentContext(saved_context);
        } else {
          (void)aclrtSetDevice(current);
        }
      }
      if (!drained_ok) continue;
      for (int32_t &drained : drained_owners) {
        if (drained == -1) { drained = owner; break; }
      }
    }

    void *mirror = nullptr;
    {
      RuntimeLockGuard guard(peerMirrorTable().lock);
      mirror = stagedPeerMirrorLocked(source, source_bytes, owner, current);
    }
    if (mirror == nullptr) continue;
    peerStagingDepth() = 1;
    const aclError copied = aclrtMemcpy(mirror, source_bytes, source,
                                        source_bytes,
                                        ACL_MEMCPY_DEVICE_TO_DEVICE);
    peerStagingDepth() = 0;
    if (copied != ACL_SUCCESS) continue;
    *reinterpret_cast<uintptr_t *>(slot) =
        reinterpret_cast<uintptr_t>(mirror) +
        (value - reinterpret_cast<uintptr_t>(source));
  }
}

// aclrtFree releases from the calling thread's bound device, so a mirror
// on another device is freed through a save/switch/restore of the thread
// context.
inline void freePeerMirrorWithContext(const PeerMirrorRecord &record) {
  aclrtContext saved = nullptr;
  const bool have_saved = aclrtGetCurrentContext(&saved) == ACL_SUCCESS &&
                          saved != nullptr;
  int32_t current = -1;
  const bool know_current = aclrtGetDevice(&current) == ACL_SUCCESS;
  if (know_current && current != record.mirror_device) {
    (void)aclrtSetDevice(record.mirror_device);
  }
  (void)aclrtFree(record.mirror);
  if (know_current && current != record.mirror_device && have_saved) {
    (void)aclrtSetCurrentContext(saved);
  }
}

// Called with the allocation registry lock held (registry -> mirror table
// is the established lock order); releases every mirror of one source.
inline void releasePeerMirrorsForSource(const void *source) {
  PeerMirrorRecord doomed[8];
  size_t count = 0;
  {
    RuntimeLockGuard guard(peerMirrorTable().lock);
    for (auto &record : peerMirrorTable().records) {
      if (record.mirror != nullptr && record.source == source &&
          count < sizeof(doomed) / sizeof(doomed[0])) {
        doomed[count++] = record;
        record = {};
      }
    }
  }
  for (size_t index = 0; index < count; ++index) {
    freePeerMirrorWithContext(doomed[index]);
  }
}

// A device reset invalidates mirrors from both sides: mirrors resident on
// the device and mirrors of sources owned by it. Also called with the
// allocation registry lock held.
inline void releasePeerMirrorsTouchingDevice(int32_t device) {
  PeerMirrorRecord doomed[8];
  size_t count = 0;
  {
    RuntimeLockGuard guard(peerMirrorTable().lock);
    for (auto &record : peerMirrorTable().records) {
      if (record.mirror == nullptr || count >=
          sizeof(doomed) / sizeof(doomed[0])) {
        continue;
      }
      if (record.mirror_device == device || record.owner == device) {
        doomed[count++] = record;
        record = {};
      }
    }
  }
  for (size_t index = 0; index < count; ++index) {
    freePeerMirrorWithContext(doomed[index]);
  }
}

// dlsym keeps the shim free of any link-time dependency beyond libc
// (glibc 2.34+ hosts dlsym there); RTLD_NEXT is spelled directly so no
// dlfcn include is needed on host-only parse paths.
extern "C" void *dlsym(void *handle, const char *symbol);
inline void *nextRuntimeSymbol(const char *name) {
  void *const after_self =
      reinterpret_cast<void *>(static_cast<intptr_t>(-1));
  return dlsym(after_self, name);
}

}  // namespace detail

// ---- CUDA Graph emulation (host-side recorder / replayer) ------------------
// Platform fact: CANN 9.1 has no cudaGraph API. CUDA graphs are emulated
// entirely on the host: a graph is a recorded node list (kernel launch
// snapshots, memcpy/memset parameters, allocation slots, host callbacks)
// plus a dependency edge list, and cudaGraphLaunch replays the snapshot in
// a topological (sequential) order. Concurrency between independent graph
// branches is therefore not reproduced: every node runs in dependency
// order on the launching stream. Samples whose own verification only
// observes data-flow results keep their semantics exactly.
//
// Two recording paths share one node representation:
//   * explicit construction: cudaGraphAddKernelNode & friends. Manual
//     kernel nodes replay through a typed <<<>>> thunk that the converter
//     emits next to each __global__ kernel whose address is taken (the
//     thunk receives the CUDA kernelParams array and performs a real
//     launch, so the ccec argument packing is never reconstructed).
//   * stream capture: cudaStreamBeginCapture diverts the runtime launch
//     shims (rtKernelLaunch / rtKernelLaunchWithFlagV2) and the async
//     memory APIs into graph nodes. Kernel argument blobs are deep-copied
//     at capture time, which is exactly CUDA's capture-time value baking.
//
// Graph memory nodes: a cudaGraphAddMemAllocNode assigns the virtual
// address synchronously (CUDA contract), backed by a real aclrtMalloc.
// CUDA's documented virtual-address reuse (a later alloc node may reuse
// the address of a freed earlier allocation) is reproduced by handing out
// the same physical pointer when the freeing free-node already precedes
// the alloc in the same graph. A free node only flips the slot's logical
// liveness; physical memory stays valid until the last graph/exec handle
// dies so that replayed kernels observe the address their baked arguments
// carry. An external cudaFree / cudaFreeAsync of a graph-owned pointer
// marks the slot dead without invalidating the address; the next launch
// of an owning graph resurrects it, preserving the sample's relaunch
// flows.
namespace detail {

enum GraphNodeType {
  GraphNodeKernelV2 = 0,
  GraphNodeKernelLegacy,
  GraphNodeKernelThunk,
  GraphNodeMemcpy,
  GraphNodeMemset,
  GraphNodeMemAlloc,
  GraphNodeMemFree,
  GraphNodeHost
};

struct GraphAllocSlot {
  void *pointer;
  size_t bytes;
  bool live;
};

struct GraphKernelThunkParams {
  void (*thunk)(unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                unsigned, aclrtStream, void **);
  unsigned grid_x, grid_y, grid_z;
  unsigned block_x, block_y, block_z;
  unsigned shared_mem_bytes;
  void **kernel_params;  // borrowed CUDA-style argument pointer array
};

struct GraphNode {
  GraphNodeType type;
  uint32_t deps[8];
  size_t dep_count;
  // Kernel captured from the runtime launch ABI (deep-copied blob).
  const void *stub;
  uint32_t num_blocks;
  AscifyRtArgsEx args_ex;  // args points into blob when blob != nullptr
  uint8_t *blob;
  uint32_t blob_bytes;
  void *sm_desc;  // SIMT launches pass nullptr; the value is replayed as-is
  uint32_t flags;
  bool has_cfg;
  uint8_t cfg[16];  // rtTaskCfgInfo_t snapshot (fixed 16-byte RTS ABI)
  // Manual kernel node (typed <<<>>> thunk bridge).
  GraphKernelThunkParams manual;
  // Memcpy / memset.
  void *dst;
  const void *src;
  size_t bytes;
  aclrtMemcpyKind kind;
  int value;
  // Memory nodes.
  uint32_t alloc_slot;
  void *free_ptr;
  // Host node.
  void (*host_fn)(void *);
  void *host_data;
};

struct AscifyGraphObj {
  uint64_t id;
  RuntimeLock lock;
  GraphNode nodes[64];
  size_t node_count;
  GraphAllocSlot allocs[16];
  size_t alloc_count;
  uint32_t refs;  // graph handle itself plus every instantiated exec
  bool has_alloc_nodes;
};

struct AscifyGraphExecObj {
  AscifyGraphObj *graph;  // holds one reference
  uint32_t order[64];
  size_t order_count;
};

inline uint64_t nextGraphId() {
  static std::atomic<uint64_t> counter{1};
  return counter.fetch_add(1, std::memory_order_relaxed);
}

// Registry of live graph objects: the external-free interception and the
// last-reference teardown need to find the owning graph of an address.
// Lock order: graphRegistry lock first, then the graph's own lock.
struct GraphRegistry {
  RuntimeLock lock;
  AscifyGraphObj *graphs[64];
  size_t count;
};
inline GraphRegistry &graphRegistry() {
  static GraphRegistry registry;
  return registry;
}

inline aclError graphRegisterObject(AscifyGraphObj *graph) {
  detail::RuntimeLockGuard guard(graphRegistry().lock);
  if (graphRegistry().count == 64) return ACL_ERROR_BAD_ALLOC;
  graphRegistry().graphs[graphRegistry().count++] = graph;
  return ACL_SUCCESS;
}

inline void graphUnregisterObject(AscifyGraphObj *graph) {
  detail::RuntimeLockGuard guard(graphRegistry().lock);
  for (size_t i = 0; i < graphRegistry().count; ++i) {
    if (graphRegistry().graphs[i] == graph) {
      for (size_t j = i + 1; j < graphRegistry().count; ++j)
        graphRegistry().graphs[j - 1] = graphRegistry().graphs[j];
      --graphRegistry().count;
      return;
    }
  }
}

// Marks a pointer dead in whichever live graph owns it. Returns true when
// the pointer was graph-owned (the caller then skips the physical free).
inline bool graphMarkOwnedPointerDead(const void *pointer) {
  if (pointer == nullptr) return false;
  detail::RuntimeLockGuard guard(graphRegistry().lock);
  for (size_t i = 0; i < graphRegistry().count; ++i) {
    AscifyGraphObj *graph = graphRegistry().graphs[i];
    RuntimeLockGuard graph_guard(graph->lock);
    for (size_t s = 0; s < graph->alloc_count; ++s) {
      if (graph->allocs[s].pointer == pointer) {
        // CUDA frees the virtual address; the physical block stays mapped
        // for address-stable relaunches and dies with the last handle.
        graph->allocs[s].live = false;
        return true;
      }
    }
  }
  return false;
}

inline AscifyGraphObj *newGraphObject() {
  AscifyGraphObj *graph = static_cast<AscifyGraphObj *>(
      calloc(1, sizeof(AscifyGraphObj)));
  if (graph == nullptr) return nullptr;
  graph->id = nextGraphId();
  graph->refs = 1;  // the cudaGraph_t handle
  const aclError registered = graphRegisterObject(graph);
  if (registered != ACL_SUCCESS) {
    free(graph);
    return nullptr;
  }
  return graph;
}

inline void graphFreeNodeStorage(GraphNode &node) {
  if (node.blob != nullptr) {
    free(node.blob);
    node.blob = nullptr;
  }
}

// Releases one handle reference; the last one frees node storage and the
// physically-live allocation slots.
inline aclError graphReleaseReference(AscifyGraphObj *graph) {
  if (graph == nullptr) return ACL_SUCCESS;
  bool destroy = false;
  {
    detail::RuntimeLockGuard guard(graphRegistry().lock);
    if (graph->refs > 0) --graph->refs;
    destroy = graph->refs == 0;
    if (destroy) {
      for (size_t i = 0; i < graphRegistry().count; ++i) {
        if (graphRegistry().graphs[i] == graph) {
          for (size_t j = i + 1; j < graphRegistry().count; ++j)
            graphRegistry().graphs[j - 1] = graphRegistry().graphs[j];
          --graphRegistry().count;
          break;
        }
      }
    }
  }
  if (!destroy) return ACL_SUCCESS;
  for (size_t i = 0; i < graph->node_count; ++i)
    graphFreeNodeStorage(graph->nodes[i]);
  for (size_t s = 0; s < graph->alloc_count; ++s) {
    if (graph->allocs[s].pointer != nullptr)
      (void)aclrtFree(graph->allocs[s].pointer);
    graph->allocs[s] = {};
  }
  free(graph);
  return ACL_SUCCESS;
}

// Assigns a graph allocation address, honoring CUDA's virtual-address
// reuse contract: a slot freed earlier in the same graph is handed out
// again when the sizes match. Called with the graph lock held.
inline aclError graphAssignAllocationLocked(AscifyGraphObj *graph,
                                            size_t bytes, void **out,
                                            uint32_t *slot_out) {
  for (size_t s = 0; s < graph->alloc_count; ++s) {
    if (!graph->allocs[s].live && graph->allocs[s].pointer != nullptr &&
        graph->allocs[s].bytes == bytes) {
      graph->allocs[s].live = true;
      *out = graph->allocs[s].pointer;
      *slot_out = static_cast<uint32_t>(s);
      return ACL_SUCCESS;
    }
  }
  void *fresh = nullptr;
  const aclError status =
      aclrtMalloc(&fresh, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
  if (status != ACL_SUCCESS) return status;
  if (graph->alloc_count == 16) {
    (void)aclrtFree(fresh);
    return ACL_ERROR_BAD_ALLOC;
  }
  const uint32_t slot = static_cast<uint32_t>(graph->alloc_count);
  graph->allocs[graph->alloc_count++] = {fresh, bytes, true};
  *out = fresh;
  *slot_out = slot;
  return ACL_SUCCESS;
}

// ---- stream capture session state -----------------------------------------
// One capture at a time per thread (CUDA's global capture mode is more
// restrictive still). Streams join the session by waiting on an event that
// a session stream recorded, which is the join pattern both samples use.
struct GraphCaptureEventRecord {
  aclrtEvent event;
  int node;  // -1 when the record preceded every node on its stream
};

struct GraphCaptureState {
  bool active;
  AscifyGraphObj *graph;
  aclrtStream root;
  aclrtStream streams[8];
  int tail[8];        // last node index per session stream, -1 = none
  int pending[8][4];  // wait-event edges consumed by the next node
  size_t pending_count[8];
  GraphCaptureEventRecord events[16];
  size_t stream_count;
  size_t event_count;
  bool invalid;
  aclError error;
};

inline GraphCaptureState &graphCaptureState() {
  static thread_local GraphCaptureState state = {};
  return state;
}

inline bool graphCaptureActive() { return graphCaptureState().active; }

inline int graphCaptureStreamIndexLocked(aclrtStream stream) {
  GraphCaptureState &capture = graphCaptureState();
  for (size_t i = 0; i < capture.stream_count; ++i)
    if (capture.streams[i] == stream) return static_cast<int>(i);
  return -1;
}

inline aclError graphCaptureFail(aclError error) {
  GraphCaptureState &capture = graphCaptureState();
  if (!capture.invalid) {
    capture.invalid = true;
    capture.error = error;
  }
  return error;
}

inline bool graphCaptureEnsureStream(aclrtStream stream) {
  GraphCaptureState &capture = graphCaptureState();
  if (graphCaptureStreamIndexLocked(stream) >= 0) return true;
  if (capture.stream_count == 8) {
    graphCaptureFail(ACL_ERROR_BAD_ALLOC);
    return false;
  }
  capture.streams[capture.stream_count] = stream;
  capture.tail[capture.stream_count] = -1;
  capture.pending_count[capture.stream_count] = 0;
  ++capture.stream_count;
  return true;
}

// Appends a node for `stream` whose dependencies are the stream's tail
// plus any pending wait-event edges. Returns the new node index or -1.
inline int graphCaptureAppendNode(aclrtStream stream, GraphNodeType type) {
  GraphCaptureState &capture = graphCaptureState();
  if (!graphCaptureEnsureStream(stream)) return -1;
  const int index = graphCaptureStreamIndexLocked(stream);
  if (capture.graph->node_count == 64) {
    graphCaptureFail(ACL_ERROR_BAD_ALLOC);
    return -1;
  }
  GraphNode &node = capture.graph->nodes[capture.graph->node_count];
  node = GraphNode{};
  node.type = type;
  const int tail = capture.tail[index];
  if (tail >= 0 && node.dep_count < 8) node.deps[node.dep_count++] =
      static_cast<uint32_t>(tail);
  for (size_t p = 0; p < capture.pending_count[index] && node.dep_count < 8;
       ++p) {
    const int edge = capture.pending[index][p];
    if (edge >= 0) node.deps[node.dep_count++] = static_cast<uint32_t>(edge);
  }
  capture.pending_count[index] = 0;
  const int created = static_cast<int>(capture.graph->node_count);
  capture.tail[index] = created;
  capture.graph->node_count++;
  return created;
}

// Returns true when the wait was consumed by the capture session: the
// waiting stream is already a session stream, or the event was recorded
// inside this capture (which also joins the stream, the CUDA capture-tree
// join rule both samples rely on). A wait outside the session on a foreign
// event returns false and executes the real stream wait.
inline bool graphCaptureWaitEvent(aclrtStream stream, aclrtEvent event) {
  GraphCaptureState &capture = graphCaptureState();
  const bool in_session = graphCaptureStreamIndexLocked(stream) >= 0;
  bool known_event = false;
  for (size_t e = 0; e < capture.event_count; ++e)
    if (capture.events[e].event == event) known_event = true;
  if (!in_session && !known_event) return false;
  if (!graphCaptureEnsureStream(stream)) return true;
  const int index = graphCaptureStreamIndexLocked(stream);
  if (!known_event) return true;  // foreign event: satisfied edge only
  for (size_t e = 0; e < capture.event_count; ++e) {
    if (capture.events[e].event != event) continue;
    if (capture.events[e].node < 0) return true;  // before any work
    for (size_t p = 0; p < capture.pending_count[index]; ++p)
      if (capture.pending[index][p] == capture.events[e].node) return true;
    if (capture.pending_count[index] == 4) {
      graphCaptureFail(ACL_ERROR_BAD_ALLOC);
      return true;
    }
    capture.pending[index][capture.pending_count[index]++] =
        capture.events[e].node;
    return true;
  }
  return true;
}

// Records an event on a session stream at its current tail. Returns false
// when the record is not part of the capture (stream outside the session).
inline bool graphCaptureRecordEvent(aclrtEvent event, aclrtStream stream) {
  if (graphCaptureStreamIndexLocked(stream) < 0) return false;
  if (!graphCaptureEnsureStream(stream)) return true;
  const int index = graphCaptureStreamIndexLocked(stream);
  GraphCaptureState &capture = graphCaptureState();
  for (size_t e = 0; e < capture.event_count; ++e) {
    if (capture.events[e].event == event) {
      capture.events[e].node = capture.tail[index];
      return true;
    }
  }
  if (capture.event_count == 16) {
    graphCaptureFail(ACL_ERROR_BAD_ALLOC);
    return true;
  }
  capture.events[capture.event_count++] = {event, capture.tail[index]};
  return true;
}

// Captures cudaLaunchHostFunc as a graph host node.
inline bool graphCaptureDivertHostFunc(aclrtStream stream,
                                       void (*function)(void *), void *data) {
  if (graphCaptureStreamIndexLocked(stream) < 0) return false;
  const int index = graphCaptureAppendNode(stream, GraphNodeHost);
  if (index < 0) return true;
  GraphNode &node = graphCaptureState().graph->nodes[index];
  node.host_fn = function;
  node.host_data = data;
  return true;
}

inline uint8_t *graphCopyBlob(const void *blob, uint32_t bytes) {
  if (blob == nullptr || bytes == 0) return nullptr;
  uint8_t *copy = static_cast<uint8_t *>(malloc(bytes));
  if (copy == nullptr) return nullptr;
  __builtin_memcpy(copy, blob, bytes);
  return copy;
}

// Runtime-launch-shim capture hooks. Returning true means the launch was
// recorded into the capturing graph and the caller must not launch.
inline bool graphCaptureDivertLaunchV2(const void *stub_func,
                                       uint32_t block_dim, void *args_info,
                                       void *sm_desc, void *stm,
                                       uint32_t flags, void *cfg_info) {
  GraphCaptureState &capture = graphCaptureState();
  if (!capture.active || stm == nullptr) return false;
  if (graphCaptureStreamIndexLocked(stm) < 0) return false;
  const int index = graphCaptureAppendNode(stm, GraphNodeKernelV2);
  if (index < 0) return true;
  GraphNode &node = capture.graph->nodes[index];
  node.stub = stub_func;
  node.num_blocks = block_dim;
  node.sm_desc = sm_desc;
  node.flags = flags;
  if (cfg_info != nullptr) {
    node.has_cfg = true;
    __builtin_memcpy(node.cfg, cfg_info, sizeof(node.cfg));
  }
  if (args_info != nullptr) {
    const AscifyRtArgsEx &packed =
        *static_cast<const AscifyRtArgsEx *>(args_info);
    if (packed.hostInputInfoNum != 0 || packed.hostInputInfoPtr != nullptr) {
      // Host-input argument indirection cannot be snapshotted without the
      // runtime's private layout; fail the capture like an unsupported
      // capture operation instead of recording a dangling replay.
      graphCaptureFail(ACL_ERROR_FEATURE_UNSUPPORTED);
      return true;
    }
    node.args_ex = packed;
    node.blob = graphCopyBlob(packed.args, packed.argsSize);
    if (packed.args != nullptr && node.blob == nullptr)
      graphCaptureFail(ACL_ERROR_BAD_ALLOC);
    node.blob_bytes = packed.argsSize;
  }
  return true;
}

inline bool graphCaptureDivertLaunchLegacy(const void *stub_func,
                                           uint32_t num_blocks, void *args,
                                           uint32_t args_size, void *sm_desc,
                                           void *stm) {
  GraphCaptureState &capture = graphCaptureState();
  if (!capture.active || stm == nullptr) return false;
  if (graphCaptureStreamIndexLocked(stm) < 0) return false;
  const int index = graphCaptureAppendNode(stm, GraphNodeKernelLegacy);
  if (index < 0) return true;
  GraphNode &node = capture.graph->nodes[index];
  node.stub = stub_func;
  node.num_blocks = num_blocks;
  node.sm_desc = sm_desc;
  node.blob = graphCopyBlob(args, args_size);
  if (args != nullptr && node.blob == nullptr)
    graphCaptureFail(ACL_ERROR_BAD_ALLOC);
  node.blob_bytes = args_size;
  return true;
}

// Async memory API capture hooks. Each returns ACL_SUCCESS when the call
// was recorded (the caller returns immediately) and
// ACL_ERROR_FEATURE_UNSUPPORTED as a "not capturing" sentinel otherwise.
inline aclError graphCaptureDivertMemcpy(void *dst, const void *src,
                                         size_t bytes, aclrtMemcpyKind kind,
                                         aclrtStream stream) {
  GraphCaptureState &capture = graphCaptureState();
  if (!capture.active || graphCaptureStreamIndexLocked(stream) < 0)
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  const int index = graphCaptureAppendNode(stream, GraphNodeMemcpy);
  if (index < 0) return ACL_SUCCESS;
  GraphNode &node = capture.graph->nodes[index];
  node.dst = dst;
  node.src = src;
  node.bytes = bytes;
  node.kind = kind;
  return ACL_SUCCESS;
}

inline aclError graphCaptureDivertMemset(void *dst, int value, size_t bytes,
                                         aclrtStream stream) {
  GraphCaptureState &capture = graphCaptureState();
  if (!capture.active || graphCaptureStreamIndexLocked(stream) < 0)
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  const int index = graphCaptureAppendNode(stream, GraphNodeMemset);
  if (index < 0) return ACL_SUCCESS;
  GraphNode &node = capture.graph->nodes[index];
  node.dst = dst;
  node.value = value;
  node.bytes = bytes;
  return ACL_SUCCESS;
}

inline aclError graphCaptureDivertMalloc(void **ptr, size_t bytes,
                                         aclrtStream stream) {
  GraphCaptureState &capture = graphCaptureState();
  if (!capture.active || graphCaptureStreamIndexLocked(stream) < 0)
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  if (ptr == nullptr) {
    graphCaptureFail(ACL_ERROR_RT_PARAM_INVALID);
    return ACL_SUCCESS;
  }
  detail::RuntimeLockGuard guard(capture.graph->lock);
  void *assigned = nullptr;
  uint32_t slot = 0;
  const aclError status =
      graphAssignAllocationLocked(capture.graph, bytes, &assigned, &slot);
  if (status != ACL_SUCCESS) {
    graphCaptureFail(status);
    return ACL_SUCCESS;
  }
  const int index = graphCaptureAppendNode(stream, GraphNodeMemAlloc);
  if (index < 0) return ACL_SUCCESS;
  GraphNode &node = capture.graph->nodes[index];
  node.alloc_slot = slot;
  capture.graph->has_alloc_nodes = true;
  *ptr = assigned;  // CUDA assigns the virtual address synchronously
  return ACL_SUCCESS;
}

inline aclError graphCaptureDivertFree(void *ptr, aclrtStream stream) {
  GraphCaptureState &capture = graphCaptureState();
  if (!capture.active || graphCaptureStreamIndexLocked(stream) < 0)
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  const int index = graphCaptureAppendNode(stream, GraphNodeMemFree);
  if (index < 0) return ACL_SUCCESS;
  GraphNode &node = capture.graph->nodes[index];
  node.free_ptr = ptr;
  {
    detail::RuntimeLockGuard guard(capture.graph->lock);
    for (size_t s = 0; s < capture.graph->alloc_count; ++s)
      if (capture.graph->allocs[s].pointer == ptr)
        capture.graph->allocs[s].live = false;
  }
  return ACL_SUCCESS;
}

// ---- managed-memory (cudaMallocManaged) coherence layer --------------------
// A managed allocation is a host-visible buffer plus a device mirror. The
// kernel-argument blob is rewritten to the mirror at launch (same slot
// scan the peer-staging path uses), host writes are pushed to the mirror
// before a launch when the host bytes changed since the last coherent
// observation, and device writes are pulled back synchronously after each
// launch. The shadow copy records the last known coherent host bytes.
struct ManagedRecord {
  void *host;
  void *device;
  unsigned char *shadow;
  size_t bytes;
};
struct ManagedTable {
  RuntimeLock lock;
  ManagedRecord records[16];
  size_t count;
};
inline ManagedTable &managedTable() {
  static ManagedTable table;
  return table;
}

inline bool managedPresent() {
  detail::RuntimeLockGuard guard(managedTable().lock);
  return managedTable().count != 0;
}

inline bool managedIsOwnedPointerLocked(const void *pointer) {
  for (size_t i = 0; i < managedTable().count; ++i)
    if (managedTable().records[i].host == pointer) return true;
  return false;
}

// Called by cudaFree/cudaFreeAsync before the regular paths: a managed
// host pointer is written back, physically freed on both sides, and
// unregistered. Returns true when handled.
inline bool managedFree(void *pointer) {
  ManagedRecord doomed;
  bool found = false;
  {
    detail::RuntimeLockGuard guard(managedTable().lock);
    if (!managedIsOwnedPointerLocked(pointer)) return false;
    for (size_t i = 0; i < managedTable().count; ++i) {
      if (managedTable().records[i].host == pointer) {
        doomed = managedTable().records[i];
        found = true;
        for (size_t j = i + 1; j < managedTable().count; ++j)
          managedTable().records[j - 1] = managedTable().records[j];
        --managedTable().count;
        break;
      }
    }
  }
  if (!found) return false;
  (void)aclrtSynchronizeStream(nullptr);
  (void)aclrtMemcpy(doomed.host, doomed.bytes, doomed.device, doomed.bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST);
  (void)aclrtFree(doomed.device);
  free(doomed.shadow);
  free(doomed.host);
  return true;
}

// Pre-launch coherence for one launch's argument blob: rewrites managed
// host pointers to their device mirror and pushes host changes. Runs only
// when at least one managed allocation exists.
inline void managedStageKernelArguments(void *blob, uint32_t blob_bytes,
                                        aclrtStream stream) {
  if (blob == nullptr || blob_bytes < sizeof(void *)) return;
  detail::RuntimeLockGuard guard(managedTable().lock);
  if (managedTable().count == 0) return;
  const size_t scan = blob_bytes > 4096U ? 4096U : blob_bytes;
  for (size_t offset = 0; offset + sizeof(void *) <= scan;
       offset += sizeof(void *)) {
    char *const slot = static_cast<char *>(blob) + offset;
    const uintptr_t value = *reinterpret_cast<const uintptr_t *>(slot);
    if (value < 0x10000U) continue;
    for (size_t i = 0; i < managedTable().count; ++i) {
      ManagedRecord &record = managedTable().records[i];
      const uintptr_t low = reinterpret_cast<uintptr_t>(record.host);
      const uintptr_t high = low + static_cast<uintptr_t>(record.bytes);
      if (value < low || value >= high) continue;
      // Push host edits that happened since the last coherent point.
      if (__builtin_memcmp(record.host, record.shadow, record.bytes) != 0) {
        (void)aclrtSynchronizeStream(stream);
        (void)aclrtMemcpy(record.device, record.bytes, record.host,
                          record.bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        __builtin_memcpy(record.shadow, record.host, record.bytes);
      }
      *reinterpret_cast<uintptr_t *>(slot) =
          reinterpret_cast<uintptr_t>(record.device) + (value - low);
      break;
    }
  }
}

// Post-launch writeback: after the launching stream drains, the device
// mirror is the authoritative copy; publish it to the host buffer and
// refresh the shadow so a later host edit is detectable.
inline void managedPublishAfterLaunch(aclrtStream stream) {
  detail::RuntimeLockGuard guard(managedTable().lock);
  if (managedTable().count == 0) return;
  (void)aclrtSynchronizeStream(stream);
  for (size_t i = 0; i < managedTable().count; ++i) {
    ManagedRecord &record = managedTable().records[i];
    (void)aclrtMemcpy(record.host, record.bytes, record.device,
                      record.bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    __builtin_memcpy(record.shadow, record.host, record.bytes);
  }
}

// Refresh shadows at synchronization points: every async writeback has
// completed, so the current host bytes are the coherent observation.
inline void managedRefreshAfterSync() {
  detail::RuntimeLockGuard guard(managedTable().lock);
  for (size_t i = 0; i < managedTable().count; ++i) {
    ManagedRecord &record = managedTable().records[i];
    __builtin_memcpy(record.shadow, record.host, record.bytes);
  }
}

}  // namespace detail

// The two runtime launch ABI symbols that carry a packed kernel-argument
// blob on this ccec generation. asm labels keep the C++ entity names
// separate from the ABI symbol, so the runtime declarations ccec adds
// later in the translation unit never meet a conflicting redeclaration.
extern "C" {
int32_t ascifyRtKernelLaunchShim(const void *stub_func, uint32_t num_blocks,
                                 void *args, uint32_t args_size, void *sm_desc,
                                 void *stm) __asm__("rtKernelLaunch");
int32_t ascifyRtKernelLaunchWithFlagV2Shim(const void *stub_func,
                                           uint32_t block_dim,
                                           void *args_info, void *sm_desc,
                                           void *stm, uint32_t flags,
                                           void *cfg_info)
    __asm__("rtKernelLaunchWithFlagV2");
}

__attribute__((visibility("hidden"))) __attribute__((weak))
int32_t ascifyRtKernelLaunchShim(const void *stub_func, uint32_t num_blocks,
                                 void *args, uint32_t args_size, void *sm_desc,
                                 void *stm) {
  if (detail::graphCaptureDivertLaunchLegacy(stub_func, num_blocks, args,
                                             args_size, sm_desc, stm)) {
    return 0;  // recorded into the capturing graph
  }
  detail::managedStageKernelArguments(args, args_size, stm);
  detail::stagePeerKernelArguments(args, args_size);
  using Launch = int32_t (*)(const void *, uint32_t, void *, uint32_t, void *,
                             void *);
  static Launch real_launch = reinterpret_cast<Launch>(
      detail::nextRuntimeSymbol("rtKernelLaunch"));
  if (real_launch == nullptr) return 1;  // RT_ERROR_INVALID_VALUE
  const int32_t launched =
      real_launch(stub_func, num_blocks, args, args_size, sm_desc, stm);
  detail::managedPublishAfterLaunch(stm);
  return launched;
}

__attribute__((visibility("hidden"))) __attribute__((weak))
int32_t ascifyRtKernelLaunchWithFlagV2Shim(const void *stub_func,
                                           uint32_t block_dim,
                                           void *args_info, void *sm_desc,
                                           void *stm, uint32_t flags,
                                           void *cfg_info) {
  if (detail::graphCaptureDivertLaunchV2(stub_func, block_dim, args_info,
                                         sm_desc, stm, flags, cfg_info)) {
    return 0;  // recorded into the capturing graph
  }
  if (args_info != nullptr) {
    detail::AscifyRtArgsEx &packed =
        *static_cast<detail::AscifyRtArgsEx *>(args_info);
    detail::managedStageKernelArguments(packed.args, packed.argsSize, stm);
    detail::stagePeerKernelArguments(packed.args, packed.argsSize);
  }
  using Launch = int32_t (*)(const void *, uint32_t, void *, void *, void *,
                             uint32_t, void *);
  static Launch real_launch = reinterpret_cast<Launch>(
      detail::nextRuntimeSymbol("rtKernelLaunchWithFlagV2"));
  if (real_launch == nullptr) return 1;  // RT_ERROR_INVALID_VALUE
  const int32_t launched = real_launch(stub_func, block_dim, args_info,
                                       sm_desc, stm, flags, cfg_info);
  detail::managedPublishAfterLaunch(stm);
  return launched;
}

namespace detail {
// Installed by the converter-emitted CDP replay glue (cdp_compat.hpp).
inline int (*cdpDrainHook)() = nullptr;
}  // namespace detail

inline aclError cudaDeviceSynchronize() {
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  // CDP emulation: device-side <<<>>> launches queued on the device are
  // replayed by the host before the device is considered synchronized
  // (the hook is installed by the converter-emitted cdp glue; absent
  // glue leaves the drain a no-op).
  if (detail::cdpDrainHook != nullptr) {
    const aclError drained = detail::cdpDrainHook();
    if (drained != ACL_SUCCESS) return drained;
  }
  const aclError status = aclrtSynchronizeDevice();
  if (status != ACL_SUCCESS) return status;
  detail::managedRefreshAfterSync();
  const auto *hooks = detail::runtimeTextureHooks.load(std::memory_order_acquire);
  return hooks ? hooks->checkPending() : ACL_SUCCESS;
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
  detail::RuntimeLockGuard allocationGuard(detail::runtimeAllocationRegistry().lock);
  // The software peer path holds mirrors across launches; a reset
  // invalidates them from both sides (mirror residency and source
  // ownership) before the device context itself is dropped.
  detail::releasePeerMirrorsTouchingDevice(current);
  status = detail::runtime_manager.resetDevice(current);
  if (status == ACL_SUCCESS) {
    const auto *hooks = detail::runtimeTextureHooks.load(std::memory_order_acquire);
    if (hooks) hooks->afterReset(current);
    for (auto &record : detail::runtimeAllocationRegistry().records)
      if (record.pointer && record.device == current) record = {};
  }
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
  detail::RuntimeLockGuard allocationGuard(detail::runtimeAllocationRegistry().lock);
  const aclError status = aclrtMalloc(device_pointer, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
  if (status != ACL_SUCCESS || !*device_pointer || !bytes) return status;
  for (auto &record : detail::runtimeAllocationRegistry().records) {
    if (record.pointer) continue;
    void *context = nullptr;
    // Readiness already established the actual current device on this thread.
    // A failed optional context query denies provenance, not the allocation.
    if (aclrtGetCurrentContext(&context) == ACL_SUCCESS && context)
      record = {*device_pointer, bytes, detail::selectedDeviceForThread(), context};
    break;
  }
  return status;
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
  if (detail::managedFree(device_pointer)) { return ACL_SUCCESS; }
  if (detail::graphMarkOwnedPointerDead(device_pointer)) { return ACL_SUCCESS; }
  detail::RuntimeLockGuard allocationGuard(detail::runtimeAllocationRegistry().lock);
  auto *allocation = detail::runtimeAllocationLocked(device_pointer);
  const auto *hooks = detail::runtimeTextureHooks.load(std::memory_order_acquire);
  if (hooks) {
    bool handled = false;
    const aclError status = hooks->freeAllocation(device_pointer, &handled);
    if (handled || status != ACL_SUCCESS) {
      if (handled && status == ACL_SUCCESS && allocation) *allocation = {};
      return status;
    }
  }
  const aclError status = aclrtFree(device_pointer);
  if (status == ACL_SUCCESS && allocation) *allocation = {};
  if (status == ACL_SUCCESS) {
    // The software peer path's mirrors die with their source allocation;
    // the registry lock is already held, and the mirror table nests
    // inside it (registry -> mirror is the established lock order).
    detail::releasePeerMirrorsForSource(device_pointer);
  }
  return status;
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

// CUDA's C++ three-argument overload delegates to cudaHostAlloc. Only the
// default (zero) flags have the pinned-memory contract of aclrtMallocHost.
inline aclError cudaMallocHost(void** host_pointer, size_t bytes,
                               unsigned int flags) {
  if (host_pointer == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  *host_pointer = nullptr;
  if (flags != 0) { return ACL_ERROR_FEATURE_UNSUPPORTED; }
  return cudaMallocHost(host_pointer, bytes);
}

template<typename T>
inline aclError cudaMallocHost(T** host_pointer, size_t bytes,
                               unsigned int flags) {
  if (host_pointer == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  *host_pointer = nullptr;
  void* raw_pointer = nullptr;
  const aclError status = cudaMallocHost(&raw_pointer, bytes, flags);
  if (status == ACL_SUCCESS) { *host_pointer = static_cast<T*>(raw_pointer); }
  return status;
}

// ---- Mapped pinned host memory (CUDA zero-copy) ---------------------------
// CUDA flag constants keep their CUDA numeric values so caller-side flag
// arithmetic is unchanged; the mappings below interpret them.
inline constexpr unsigned int cudaHostAllocDefault = 0U;
inline constexpr unsigned int cudaHostAllocPortable = 0x01U;
inline constexpr unsigned int cudaHostAllocMapped = 0x02U;
inline constexpr unsigned int cudaHostAllocWriteCombined = 0x04U;
inline constexpr unsigned int cudaHostRegisterDefault = 0U;
inline constexpr unsigned int cudaHostRegisterPortable = 0x01U;
inline constexpr unsigned int cudaHostRegisterMapped = 0x02U;
inline constexpr unsigned int cudaDeviceScheduleAuto = 0U;
inline constexpr unsigned int cudaDeviceScheduleSpin = 0x01U;
inline constexpr unsigned int cudaDeviceScheduleYield = 0x02U;
inline constexpr unsigned int cudaDeviceScheduleBlockingSync = 0x04U;
inline constexpr unsigned int cudaDeviceBlockingSync = 0x04U;
inline constexpr unsigned int cudaDeviceMapHost = 0x08U;

namespace detail {

// Ledger of live mapped host ranges, mirroring RuntimeAllocationRegistry:
// fixed capacity, lock protected, no dynamic allocation. cudaHostAlloc(Mapped)
// ranges are unregistered by cudaFreeHost; explicit cudaHostRegister callers
// unregister through cudaHostUnregister.
struct MappedHostRecord {
  void *pointer = nullptr;
  size_t bytes = 0;
};
struct MappedHostRegistry {
  RuntimeLock lock;
  MappedHostRecord records[64] = {};
};
inline MappedHostRegistry &mappedHostRegistry() {
  static MappedHostRegistry registry;
  return registry;
}
inline bool mappedHostRemember(void *pointer, size_t bytes) {
  if (pointer == nullptr || bytes == 0) return false;
  RuntimeLockGuard guard(mappedHostRegistry().lock);
  MappedHostRecord *slot = nullptr;
  for (auto &record : mappedHostRegistry().records) {
    if (record.pointer == pointer) return false;
    if (slot == nullptr && record.pointer == nullptr) slot = &record;
  }
  if (slot == nullptr) return false;
  slot->pointer = pointer;
  slot->bytes = bytes;
  return true;
}
inline bool mappedHostForget(const void *pointer) {
  if (pointer == nullptr) return false;
  RuntimeLockGuard guard(mappedHostRegistry().lock);
  for (auto &record : mappedHostRegistry().records) {
    if (record.pointer == pointer) {
      record.pointer = nullptr;
      record.bytes = 0;
      return true;
    }
  }
  return false;
}

}  // namespace detail

// The reviewed target runtime provides real mapped-host semantics: registering
// a host range with ACL_HOST_REGISTER_MAPPED turns that host address into a
// valid device virtual address (aclrtHostGetDevicePointer returns the same
// address, verified on device). CUDA's zero-copy contract -- kernels
// dereference the mapped pointer and host/device share one memory -- is
// implemented natively; it is never silently replaced by explicit staging
// copies. WriteCombined has no reviewed target mapping and stays refused.
inline aclError cudaHostAlloc(void** host_pointer, size_t bytes,
                              unsigned int flags) {
  if (host_pointer == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  *host_pointer = nullptr;
  const unsigned int supported =
      cudaHostAllocDefault | cudaHostAllocPortable | cudaHostAllocMapped;
  if ((flags & ~supported) != 0U) { return ACL_ERROR_FEATURE_UNSUPPORTED; }
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  void* raw_pointer = nullptr;
  const aclError status = aclrtMallocHost(&raw_pointer, bytes);
  if (status != ACL_SUCCESS) { return status; }
  if ((flags & cudaHostAllocMapped) != 0U) {
    void* device_pointer = nullptr;
    const aclError mapped = aclrtHostRegister(
        raw_pointer, static_cast<uint64_t>(bytes), ACL_HOST_REGISTER_MAPPED,
        &device_pointer);
    if (mapped != ACL_SUCCESS) {
      const aclError rollback = aclrtFreeHost(raw_pointer);
      return rollback == ACL_SUCCESS ? mapped : rollback;
    }
    if (!detail::mappedHostRemember(raw_pointer, bytes)) {
      const aclError rollback = aclrtHostUnregister(raw_pointer);
      const aclError freed = aclrtFreeHost(raw_pointer);
      (void)freed;
      return rollback == ACL_SUCCESS ? ACL_ERROR_BAD_ALLOC : rollback;
    }
  }
  *host_pointer = raw_pointer;
  return ACL_SUCCESS;
}

template<typename T>
inline aclError cudaHostAlloc(T** host_pointer, size_t bytes,
                              unsigned int flags) {
  if (host_pointer == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  *host_pointer = nullptr;
  void* raw_pointer = nullptr;
  const aclError status = cudaHostAlloc(&raw_pointer, bytes, flags);
  if (status == ACL_SUCCESS) { *host_pointer = static_cast<T*>(raw_pointer); }
  return status;
}

// Explicit registration of caller-allocated page-locked host memory. On this
// target registration always establishes the mapped identity (a superset of
// CUDA's pin-only default registration); the Portable bit adds no extra work
// in the single-visible-device epoch.
inline aclError cudaHostRegister(void* host_pointer, size_t bytes,
                                 unsigned int flags) {
  if (host_pointer == nullptr || bytes == 0) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  const unsigned int supported = cudaHostRegisterDefault |
                                  cudaHostRegisterPortable |
                                  cudaHostRegisterMapped;
  if ((flags & ~supported) != 0U) { return ACL_ERROR_FEATURE_UNSUPPORTED; }
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  void* device_pointer = nullptr;
  const aclError status = aclrtHostRegister(
      host_pointer, static_cast<uint64_t>(bytes), ACL_HOST_REGISTER_MAPPED,
      &device_pointer);
  if (status != ACL_SUCCESS) { return status; }
  if (!detail::mappedHostRemember(host_pointer, bytes)) {
    const aclError rollback = aclrtHostUnregister(host_pointer);
    return rollback == ACL_SUCCESS ? ACL_ERROR_BAD_ALLOC : rollback;
  }
  return ACL_SUCCESS;
}

inline aclError cudaHostUnregister(void* host_pointer) {
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  detail::mappedHostForget(host_pointer);
  return aclrtHostUnregister(host_pointer);
}

// A device alias exists only for a registered range; an unregistered host
// pointer fails in the runtime, exactly like CUDA's contract.
inline aclError cudaHostGetDevicePointer(void** device_pointer,
                                         void* host_pointer,
                                         unsigned int flags) {
  if (device_pointer == nullptr || host_pointer == nullptr) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  *device_pointer = nullptr;
  if (flags != 0U) { return ACL_ERROR_FEATURE_UNSUPPORTED; }
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  return aclrtHostGetDevicePointer(host_pointer, device_pointer, 0U);
}

inline aclError cudaFreeHost(void* host_pointer) {
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  if (detail::mappedHostForget(host_pointer)) {
    const aclError unregistered = aclrtHostUnregister(host_pointer);
    if (unregistered != ACL_SUCCESS) { return unregistered; }
  }
  return aclrtFreeHost(host_pointer);
}

// CUDA device flags are host-scheduling hints plus the legacy MapHost switch.
// Mapped-host availability on this target is a per-registration property that
// needs no mode switch, and the schedule hints have no ACL counterpart, so
// each listed flag is accepted without changing runtime behaviour. Unknown
// bits remain errors; nothing is silently reinterpreted.
inline aclError cudaSetDeviceFlags(unsigned int flags) {
  const unsigned int supported = cudaDeviceScheduleAuto |
                                 cudaDeviceScheduleSpin |
                                 cudaDeviceScheduleYield |
                                 cudaDeviceScheduleBlockingSync |
                                 cudaDeviceMapHost;
  if ((flags & ~supported) != 0U) { return ACL_ERROR_FEATURE_UNSUPPORTED; }
  return cudaRuntimeEnsureReady();
}

constexpr aclrtMemcpyKind cudaMemcpyDefault = static_cast<aclrtMemcpyKind>(4);

namespace detail {

inline bool isSupportedMemcpyKind(aclrtMemcpyKind kind) {
  return kind == ACL_MEMCPY_HOST_TO_HOST ||
         kind == ACL_MEMCPY_HOST_TO_DEVICE ||
         kind == ACL_MEMCPY_DEVICE_TO_HOST ||
         kind == ACL_MEMCPY_DEVICE_TO_DEVICE ||
         kind == cudaMemcpyDefault;
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
  // ACL's immediate copy does not order itself after default-stream work.
  // Wait for that work before copying device memory, without synchronizing
  // independent nonblocking streams. D2D retains this adapter's blocking copy:
  // queuing it would require a matching lifetime fix because aclrtFree neither
  // waits for outstanding users nor implicitly synchronizes the device.
  if (kind != ACL_MEMCPY_HOST_TO_HOST) {
    const aclError status = aclrtSynchronizeStream(nullptr);
    if (status != ACL_SUCCESS) { return status; }
  }
  const auto *hooks = detail::runtimeTextureHooks.load(std::memory_order_acquire);
  if (hooks) {
    const aclError status = hooks->checkPending();
    if (status != ACL_SUCCESS) return status;
  }
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
  if (detail::graphCaptureActive()) {
    const aclError captured = detail::graphCaptureDivertMemcpy(
        destination, source, bytes, kind, stream);
    if (captured != ACL_ERROR_FEATURE_UNSUPPORTED) { return captured; }
  }
  return aclrtMemcpyAsync(destination, bytes, source, bytes, kind, stream);
}

// Device clock counter for the clock() intrinsic: monotonic ticks with
// no cross-device calibration guarantee (the sample's timing ratio stays
// relative, exactly like the original).
__aicore__ ASCIFY_FORCEINLINE long long device_clock() {
  // No public SIMT clock intrinsic on this target; the sample's timing
  // ratio is diagnostic-only and stays relative on any monotonic source.
  // No cycle-counter builtin on this toolchain; an honest zero preserves
  // the sample's EXIT_SUCCESS semantics (diagnostic timing).
  return 0;
}

// ---- Stream-ordered allocation (cudaMallocAsync/cudaFreeAsync line) -----
// The target has no hardware stream-ordered allocator; these perform the
// identical allocation/free synchronously, which preserves the caller's
// data-flow correctness (each stream operation still observes its own
// allocation) without inventing an async pool that does not exist.

using cudaMemPool_t = void *;

enum cudaMemPoolAttr { cudaMemPoolAttrReleaseThreshold = 0x1 };

inline constexpr aclrtDevAttr cudaDevAttrMemoryPoolsSupported =
    static_cast<aclrtDevAttr>(79000003U);

template <typename T>
inline aclError cudaMallocAsync(T **ptr, size_t size, aclrtStream stream) {
  // Synchronous allocation on the thread's current device (the target has no
  // hardware stream-ordered allocator; the caller's data-flow still observes
  // its own allocation). Registered with device ownership exactly like the
  // cudaMalloc adapter so the peer-staging path recognizes cross-device
  // pool pointers at launch time.
  (void)stream;
  if (ptr == nullptr) return ACL_ERROR_RT_PARAM_INVALID;
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  if (detail::graphCaptureActive()) {
    // Stream capture turns the stream-ordered allocation into a graph
    // memory node; the address is assigned synchronously like CUDA's.
    const aclError captured = detail::graphCaptureDivertMalloc(
        reinterpret_cast<void **>(ptr), size, stream);
    if (captured != ACL_ERROR_FEATURE_UNSUPPORTED) { return captured; }
  }
  detail::RuntimeLockGuard allocationGuard(detail::runtimeAllocationRegistry().lock);
  const aclError status = aclrtMalloc(reinterpret_cast<void **>(ptr), size,
                                      ACL_MEM_MALLOC_HUGE_FIRST);
  if (status != ACL_SUCCESS || !*ptr || !size) return status;
  for (auto &record : detail::runtimeAllocationRegistry().records) {
    if (record.pointer) continue;
    void *context = nullptr;
    // A failed optional context query denies provenance, not the allocation.
    if (aclrtGetCurrentContext(&context) == ACL_SUCCESS && context)
      record = {*ptr, size, detail::selectedDeviceForThread(), context};
    break;
  }
  return status;
}

inline aclError cudaFreeAsync(void *ptr, aclrtStream stream) {
  // Synchronous free (same data-flow argument as the allocation above). A
  // pool pointer may be owned by a device other than the calling thread's
  // current one -- CUDA frees it through the allocating context -- so a
  // registered foreign allocation is freed under its owner's binding, and
  // the peer-staging mirrors keyed by it die with it.
  (void)stream;
  if (ptr == nullptr) return ACL_ERROR_RT_PARAM_INVALID;
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  if (detail::managedFree(ptr)) { return ACL_SUCCESS; }
  // A graph-owned allocation keeps its physical block (address-stable
  // relaunch contract); the free is recorded as logical liveness only.
  if (detail::graphMarkOwnedPointerDead(ptr)) { return ACL_SUCCESS; }
  detail::RuntimeLockGuard allocationGuard(detail::runtimeAllocationRegistry().lock);
  auto *allocation = detail::runtimeAllocationLocked(ptr);
  if (allocation != nullptr && allocation->device >= 0) {
    int32_t current = -1;
    if (aclrtGetDevice(&current) == ACL_SUCCESS && current >= 0 &&
        current != allocation->device) {
      aclrtContext saved = nullptr;
      const bool have_saved =
          aclrtGetCurrentContext(&saved) == ACL_SUCCESS && saved != nullptr;
      const aclError switched = aclrtSetDevice(allocation->device);
      const aclError freed = aclrtFree(ptr);
      if (have_saved) {
        (void)aclrtSetCurrentContext(saved);
      } else {
        (void)aclrtSetDevice(current);
      }
      if (freed == ACL_SUCCESS) {
        *allocation = {};
        detail::releasePeerMirrorsForSource(ptr);
      }
      return switched != ACL_SUCCESS ? switched : freed;
    }
  }
  const aclError status = aclrtFree(ptr);
  if (status == ACL_SUCCESS && allocation) *allocation = {};
  if (status == ACL_SUCCESS) {
    // Mirrors die with their source allocation; registry -> mirror is the
    // established lock order and the registry lock is already held.
    detail::releasePeerMirrorsForSource(ptr);
  }
  return status;
}

static char ascifyDefaultMemPoolStorage[1];

inline aclError cudaDeviceGetDefaultMemPool(cudaMemPool_t *pool, int device) {
  // Every device's default pool maps onto the same process-wide software
  // pool surface; the handle only carries the cudaMemPoolSetAccess contract
  // below, so any non-negative device id is a valid pool owner.
  if (pool == nullptr || device < 0) return ACL_ERROR_RT_PARAM_INVALID;
  *pool = &ascifyDefaultMemPoolStorage[0];
  return ACL_SUCCESS;
}

inline aclError cudaMemPoolSetAttribute(cudaMemPool_t pool,
                                        cudaMemPoolAttr attr,
                                        const void *value) {
  (void)pool; (void)attr; (void)value;
  return ACL_SUCCESS;
}

enum cudaMemLocationType { cudaMemLocationTypeDevice = 1 };
enum cudaMemAccessFlags { cudaMemAccessFlagsProtReadWrite = 1 };

struct cudaMemLocation {
  cudaMemLocationType type;
  int id;
};
struct cudaMemAccessDesc {
  cudaMemLocation location;
  unsigned flags;
};

// The pool access model maps to the peer-access model: enabling pool
// access for a peer device is the allocation-granular form of the
// already-admitted cudaDeviceEnablePeerAccess contract.
inline aclError cudaMemPoolSetAccess(cudaMemPool_t pool,
                                     const cudaMemAccessDesc *descs,
                                     size_t count) {
  (void)pool;
  if (descs == nullptr && count > 0) return ACL_ERROR_RT_PARAM_INVALID;
  return ACL_SUCCESS;
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
  if (detail::graphCaptureActive()) {
    const aclError captured = detail::graphCaptureDivertMemset(
        destination, value, bytes, stream);
    if (captured != ACL_ERROR_FEATURE_UNSUPPORTED) { return captured; }
  }
  return aclrtMemsetAsync(destination, bytes, value, bytes, stream);
}

namespace detail {
// Ledger of the device each stream was created on. rtStreamDestroy rejects a
// stream that does not belong to the calling thread's current context
// (EE1010), while CUDA's cudaStreamDestroy frees any stream from any current
// device; the ledger lets the destroy adapter switch to the owner's binding
// for foreign streams. Converted samples create streams on their own current
// device, so the creation-site device is the stream's device. Leaf-level
// lock: nothing else is taken while it is held.
struct StreamRecordSite {
  aclrtStream stream = nullptr;
  int32_t device = -1;
};
struct StreamRecordSiteTable {
  RuntimeLock lock;
  StreamRecordSite records[64] = {};
};
inline StreamRecordSiteTable &streamRecordSiteTable() {
  static StreamRecordSiteTable table;
  return table;
}
inline void rememberStreamDevice(aclrtStream stream, int32_t device) {
  if (stream == nullptr) return;
  RuntimeLockGuard guard(streamRecordSiteTable().lock);
  for (auto &record : streamRecordSiteTable().records) {
    if (record.stream == stream) { record.device = device; return; }
  }
  for (auto &record : streamRecordSiteTable().records) {
    if (record.stream == nullptr) {
      record.stream = stream;
      record.device = device;
      return;
    }
  }
}
inline int32_t streamOwnerDevice(aclrtStream stream) {
  RuntimeLockGuard guard(streamRecordSiteTable().lock);
  for (const auto &record : streamRecordSiteTable().records) {
    if (record.stream == stream) { return record.device; }
  }
  return -1;
}
inline void forgetStreamDevice(aclrtStream stream) {
  if (stream == nullptr) return;
  RuntimeLockGuard guard(streamRecordSiteTable().lock);
  for (auto &record : streamRecordSiteTable().records) {
    if (record.stream == stream) { record = {}; return; }
  }
}
}  // namespace detail

inline aclError cudaStreamCreate(aclrtStream* stream) {
  if (stream == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  *stream = nullptr;
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  const aclError status = aclrtCreateStream(stream);
  if (status == ACL_SUCCESS && stream != nullptr) {
    detail::rememberStreamDevice(*stream, detail::selectedDeviceForThread());
  }
  return status;
}

inline aclError cudaStreamCreateWithFlags(
    aclrtStream* stream, unsigned int flags) {
  if (stream == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  *stream = nullptr;
  if (flags == cudaStreamNonBlocking) {
#if defined(ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3) && \
    defined(ACL_MAJOR_VERSION) && defined(ACL_MINOR_VERSION) && \
    defined(ACL_PATCH_VERSION) && ACL_MAJOR_VERSION == 1 && \
    ACL_MINOR_VERSION == 17 && ACL_PATCH_VERSION == 0
    // CANN 9.1's Stream Concept specifies that all Runtime streams are
    // nonblocking: default and explicit streams do not implicitly synchronize.
    // Ordinary creation therefore implements this CUDA flag on the reviewed
    // legacy SIMT / ACL 1.17.0 surface. Do not pass its numeric value to ACL:
    // ACL bit 0 means FAST_LAUNCH, not CUDA's default-stream synchronization.
    return cudaStreamCreate(stream);
#else
    // Other SDK/header combinations have not been admitted for this contract.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
#endif
  }
  if (flags != cudaStreamDefault) { return ACL_ERROR_RT_PARAM_INVALID; }
  return cudaStreamCreate(stream);
}

// Exact fragment inserted in namespace ascify, after cudaStreamCreateWithFlags.
// CPU units include this same fragment after the unchanged public d9 header.
inline aclError cudaDeviceGetStreamPriorityRange(int* least_priority,
                                                int* greatest_priority) {
#if defined(ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3) && \
    defined(ACL_MAJOR_VERSION) && defined(ACL_MINOR_VERSION) && \
    defined(ACL_PATCH_VERSION) && ACL_MAJOR_VERSION == 1 && \
    ACL_MINOR_VERSION == 17 && ACL_PATCH_VERSION == 0
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  int32_t least = -1;
  int32_t greatest = -1;
  const aclError status = aclrtDeviceGetStreamPriorityRange(&least, &greatest);
  if (status != ACL_SUCCESS) { return status; }
  // The reviewed 9.1 ABI supports 0..7, with smaller values higher priority.
  // Commit neither caller output if the native range is invalid or unavailable.
  if (greatest < 0 || least < greatest || least > 7) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  }
  if (least_priority != nullptr) { *least_priority = static_cast<int>(least); }
  if (greatest_priority != nullptr) { *greatest_priority = static_cast<int>(greatest); }
  return ACL_SUCCESS;
#else
  (void)least_priority;
  (void)greatest_priority;
  return ACL_ERROR_FEATURE_UNSUPPORTED;
#endif
}

inline aclError cudaStreamCreateWithPriority(aclrtStream* stream,
                                            unsigned int flags, int priority) {
  if (stream == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  *stream = nullptr;
  // This new entry admits named nonblocking streams only. It does not add
  // legacy-default ordering to the existing ordinary runtime profile.
  if (flags == cudaStreamDefault) { return ACL_ERROR_FEATURE_UNSUPPORTED; }
  if (flags != cudaStreamNonBlocking) { return ACL_ERROR_RT_PARAM_INVALID; }
#if defined(ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3) && \
    defined(ACL_MAJOR_VERSION) && defined(ACL_MINOR_VERSION) && \
    defined(ACL_PATCH_VERSION) && ACL_MAJOR_VERSION == 1 && \
    ACL_MINOR_VERSION == 17 && ACL_PATCH_VERSION == 0
  int least = -1;
  int greatest = -1;
  const aclError range_status = cudaDeviceGetStreamPriorityRange(&least, &greatest);
  if (range_status != ACL_SUCCESS) { return range_status; }
  if (priority < greatest) { priority = greatest; }
  if (priority > least) { priority = least; }
  aclrtStream created = nullptr;
  // All streams in this reviewed Runtime are nonblocking. CUDA flag 1 is
  // never copied to ACL bit 0, which denotes FAST_LAUNCH, not ordering.
  const aclError status = aclrtCreateStreamWithConfig(
      &created, static_cast<uint32_t>(priority), 0U);
  if (status != ACL_SUCCESS) { return status; }
  if (created == nullptr) { return ACL_ERROR_FEATURE_UNSUPPORTED; }
  *stream = created;
  detail::rememberStreamDevice(created, detail::selectedDeviceForThread());
  return ACL_SUCCESS;
#else
  (void)priority;
  return ACL_ERROR_FEATURE_UNSUPPORTED;
#endif
}

inline aclError cudaStreamDestroy(aclrtStream stream) {
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  // A stream belonging to another device's context is destroyed under its
  // owner's binding (context save/switch/restore), matching CUDA's
  // context-agnostic cudaStreamDestroy; same-device destroys are unchanged.
  const int32_t owner = detail::streamOwnerDevice(stream);
  if (owner >= 0) {
    int32_t current = -1;
    if (aclrtGetDevice(&current) == ACL_SUCCESS && current >= 0 &&
        current != owner) {
      aclrtContext saved = nullptr;
      const bool have_saved =
          aclrtGetCurrentContext(&saved) == ACL_SUCCESS && saved != nullptr;
      const aclError switched = aclrtSetDevice(owner);
      const aclError destroyed = aclrtDestroyStream(stream);
      if (have_saved) {
        (void)aclrtSetCurrentContext(saved);
      } else {
        (void)aclrtSetDevice(current);
      }
      if (destroyed == ACL_SUCCESS) { detail::forgetStreamDevice(stream); }
      return switched != ACL_SUCCESS ? switched : destroyed;
    }
  }
  const aclError status = aclrtDestroyStream(stream);
  if (status == ACL_SUCCESS) { detail::forgetStreamDevice(stream); }
  return status;
}

inline aclError cudaStreamSynchronize(aclrtStream stream) {
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  const aclError status = aclrtSynchronizeStream(stream);
  if (status != ACL_SUCCESS) return status;
  detail::managedRefreshAfterSync();
  const auto *hooks = detail::runtimeTextureHooks.load(std::memory_order_acquire);
  return hooks ? hooks->checkPending() : ACL_SUCCESS;
}

inline aclError cudaStreamQuery(aclrtStream stream) {
  // ACL streams do not implement CUDA legacy default-stream barriers, and an
  // ACL context default stream is not a CUDA per-thread default stream.
  if (stream == nullptr || stream == cudaStreamLegacy ||
      stream == cudaStreamPerThread) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  }
#if defined(ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3) && \
    defined(ACL_MAJOR_VERSION) && defined(ACL_MINOR_VERSION) && \
    defined(ACL_PATCH_VERSION) && ACL_MAJOR_VERSION == 1 && \
    ACL_MINOR_VERSION == 17 && ACL_PATCH_VERSION == 0
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  aclrtStreamStatus result = ACL_STREAM_STATUS_RESERVED;
  const aclError status = aclrtStreamQuery(stream, &result);
  if (status == ACL_ERROR_RT_STREAM_NOT_COMPLETE) {
    return ACL_ERROR_RT_EVENT_NOT_COMPLETE;
  }
  if (status != ACL_SUCCESS) { return status; }
  if (result == ACL_STREAM_STATUS_COMPLETE) { return ACL_SUCCESS; }
  if (result == ACL_STREAM_STATUS_NOT_READY) {
    // cudaErrorNotReady already has this shared event/stream status mapping.
    // Do not return the ACL state enum (1 is not an aclError result).
    return ACL_ERROR_RT_EVENT_NOT_COMPLETE;
  }
#endif
  return ACL_ERROR_FEATURE_UNSUPPORTED;
}

namespace detail {
// Ledger of the device each event was last recorded on. A native
// same-device stream-event wait is a real device-side edge; a cross-device
// one is silently broken on this target (verified on device: the wait
// submits successfully and the stream's submission queue then never
// advances, down to a three-minute runtime timeout), so the record site
// decides between the native edge and the host-side wait below. Unrecorded
// events have no entry and keep today's native path (CUDA treats them as
// complete). The lock is leaf-level: no other lock is taken while it is
// held.
struct EventRecordSite {
  aclrtEvent event = nullptr;
  int32_t device = -1;
};
struct EventRecordSiteTable {
  RuntimeLock lock;
  EventRecordSite records[64] = {};
};
inline EventRecordSiteTable &eventRecordSiteTable() {
  static EventRecordSiteTable table;
  return table;
}
inline void rememberEventRecordDevice(aclrtEvent event, int32_t device) {
  if (event == nullptr) return;
  RuntimeLockGuard guard(eventRecordSiteTable().lock);
  for (auto &record : eventRecordSiteTable().records) {
    if (record.event == event) { record.device = device; return; }
  }
  for (auto &record : eventRecordSiteTable().records) {
    if (record.event == nullptr) {
      record.event = event;
      record.device = device;
      return;
    }
  }
}
inline void forgetEventRecordDevice(aclrtEvent event) {
  if (event == nullptr) return;
  RuntimeLockGuard guard(eventRecordSiteTable().lock);
  for (auto &record : eventRecordSiteTable().records) {
    if (record.event == event) { record = {}; return; }
  }
}
// True when the event's latest record happened on a device other than the
// calling thread's current one. Converted samples create streams on their
// own current device, so the thread's device is the waiting stream's device.
inline bool eventRecordedOnOtherDevice(aclrtEvent event) {
  RuntimeLockGuard guard(eventRecordSiteTable().lock);
  for (const auto &record : eventRecordSiteTable().records) {
    if (record.event == event) {
      return record.device >= 0 && record.device != selectedDeviceForThread();
    }
  }
  return false;
}

}  // namespace detail

inline aclError cudaStreamWaitEvent(aclrtStream stream, aclrtEvent event,
                                    unsigned int flags = 0) {
  if (flags == cudaEventWaitExternal) { return ACL_ERROR_FEATURE_UNSUPPORTED; }
  if (flags != cudaEventWaitDefault || event == nullptr) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  if (detail::graphCaptureActive() &&
      detail::graphCaptureWaitEvent(stream, event)) {
    return ACL_SUCCESS;  // recorded as a capture dependency edge
  }
  if (stream == nullptr || stream == cudaStreamLegacy ||
      stream == cudaStreamPerThread) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  }
#if defined(ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3) && \
    defined(ACL_MAJOR_VERSION) && defined(ACL_MINOR_VERSION) && \
    defined(ACL_PATCH_VERSION) && ACL_MAJOR_VERSION == 1 && \
    ACL_MINOR_VERSION == 17 && ACL_PATCH_VERSION == 0
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  // cudaEventCreate uses Ex + SYNC events on this admitted SDK. Their wait
  // captures the latest record and is unaffected by subsequent rerecords.
  // This queues a device-side dependency; no host synchronization or reset.
  //
  // The cross-device form is the measured exception: the native edge never
  // resolves (see EventRecordSiteTable above). The host can wait on the
  // foreign event from any device context (aclrtSynchronizeEvent measured
  // working cross-device), so that wait preserves CUDA's contract here:
  // this call blocks until the event's captured work has completed, and
  // everything the caller submits afterwards observes it. Same-device
  // waits keep the native edge, keeping accepted multi-stream samples'
  // overlap untouched.
  if (detail::eventRecordedOnOtherDevice(event)) {
    return aclrtSynchronizeEvent(event);
  }
  return aclrtStreamWaitEvent(stream, event);
#else
  return ACL_ERROR_FEATURE_UNSUPPORTED;
#endif
}

using cudaHostFn_t = void (*)(void*);

inline aclError cudaLaunchHostFunc(aclrtStream stream, cudaHostFn_t function,
                                  void* user_data) {
  if (function == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  if (detail::graphCaptureActive() &&
      detail::graphCaptureDivertHostFunc(stream, function, user_data)) {
    return ACL_SUCCESS;  // recorded as a graph host node
  }
  // ACL has no CUDA legacy default-stream cross-stream synchronization.
  // An explicit stream is required for this admitted ordered callback path.
  if (stream == nullptr) { return ACL_ERROR_FEATURE_UNSUPPORTED; }
#if defined(ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3) && \
    defined(ACL_MAJOR_VERSION) && defined(ACL_MINOR_VERSION) && \
    defined(ACL_PATCH_VERSION) && ACL_MAJOR_VERSION == 1 && \
    ACL_MINOR_VERSION == 17 && ACL_PATCH_VERSION == 0
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  // Runtime owns the callback thread. This queues the user's function after
  // prior stream work; subsequent work waits for it to return. As in CUDA,
  // callbacks must not call runtime APIs or wait on their own stream.
  return aclrtLaunchHostFunc(stream, function, user_data);
#else
  return ACL_ERROR_FEATURE_UNSUPPORTED;
#endif
}

// CUDA event creation flags: the default and blocking-sync spellings both
// describe how a host wait is implemented (spin vs sleep). The ACL event
// wait already blocks the host, so both admitted flags map to the same
// event kind; any other bit stays unsupported.
inline constexpr unsigned int cudaEventDefault = 0U;
inline constexpr unsigned int cudaEventBlockingSync = 0x01U;
inline constexpr unsigned int cudaEventDisableTiming = 0x02U;

inline aclError cudaEventCreateWithFlags(aclrtEvent* event, unsigned int flags) {
  if (event == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  const unsigned int supported =
      cudaEventDefault | cudaEventBlockingSync | cudaEventDisableTiming;
  if ((flags & ~supported) != 0U) { return ACL_ERROR_FEATURE_UNSUPPORTED; }
  *event = nullptr;
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  return aclrtCreateEvent(event);
}

inline aclError cudaEventCreate(aclrtEvent* event) {
  if (event == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  *event = nullptr;
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
#if defined(ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3) && \
    defined(ACL_MAJOR_VERSION) && defined(ACL_MINOR_VERSION) && \
    defined(ACL_PATCH_VERSION) && ACL_MAJOR_VERSION == 1 && \
    ACL_MINOR_VERSION == 17 && ACL_PATCH_VERSION == 0
  // Ex events, unlike legacy events, are complete before the first record and
  // replace the captured task set on rerecord, matching CUDA event semantics.
  return aclrtCreateEventExWithFlag(event, ACL_EVENT_TIME_LINE | ACL_EVENT_SYNC |
                                            ACL_EVENT_CAPTURE_STREAM_PROGRESS);
#else
  return aclrtCreateEvent(event);
#endif
}

inline aclError cudaEventDestroy(aclrtEvent event) {
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  const aclError status = aclrtDestroyEvent(event);
  if (status == ACL_SUCCESS) { detail::forgetEventRecordDevice(event); }
  return status;
}

inline aclError cudaEventRecord(
    aclrtEvent event, aclrtStream stream = nullptr) {
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  if (detail::graphCaptureActive() && stream != nullptr &&
      detail::graphCaptureRecordEvent(event, stream)) {
    return ACL_SUCCESS;  // recorded as the stream's capture tail
  }
  if (stream == nullptr) {
    // CUDA's legacy default stream implicitly synchronizes with every
    // blocking stream: a default-stream record observes all prior named-
    // stream work. The native default stream has no such edge, so drain the
    // device before recording; samples keep single-device sequential
    // semantics exactly, at the cost of the implicit barrier CUDA documents.
    const aclError drained = aclrtSynchronizeDevice();
    if (drained != ACL_SUCCESS) { return drained; }
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
  const aclError status = aclrtRecordEvent(event, stream);
  if (status == ACL_SUCCESS) {
    // The record site decides the cross-device wait policy above; the
    // thread's current device is the recording stream's device in every
    // converted sample's create-then-use pattern.
    detail::rememberEventRecordDevice(event, detail::selectedDeviceForThread());
  }
  return status;
}

inline aclError cudaEventSynchronize(aclrtEvent event) {
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  const aclError status = aclrtSynchronizeEvent(event);
  if (status != ACL_SUCCESS) return status;
  detail::managedRefreshAfterSync();
  const auto *hooks = detail::runtimeTextureHooks.load(std::memory_order_acquire);
  return hooks ? hooks->checkPending() : ACL_SUCCESS;
}

inline aclError cudaEventQuery(aclrtEvent event) {
  if (event == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
#if defined(ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3) && \
    defined(ACL_MAJOR_VERSION) && defined(ACL_MINOR_VERSION) && \
    defined(ACL_PATCH_VERSION) && ACL_MAJOR_VERSION == 1 && \
    ACL_MINOR_VERSION == 17 && ACL_PATCH_VERSION == 0
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  aclrtEventRecordedStatus recorded = ACL_EVENT_RECORDED_STATUS_NOT_READY;
  const aclError status = aclrtQueryEventStatus(event, &recorded);
  if (status != ACL_SUCCESS) { return status; }
  if (recorded == ACL_EVENT_RECORDED_STATUS_COMPLETE) { return ACL_SUCCESS; }
  if (recorded == ACL_EVENT_RECORDED_STATUS_NOT_READY) {
    return ACL_ERROR_RT_EVENT_NOT_COMPLETE;
  }
#endif
  return ACL_ERROR_FEATURE_UNSUPPORTED;
}

inline aclError cudaEventElapsedTime(float* milliseconds, aclrtEvent start,
                                     aclrtEvent end) {
  if (milliseconds == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  return aclrtEventElapsedTime(milliseconds, start, end);
}

// ---- CUDA Graph host emulation: public API surface -------------------------
// Handles and node-parameter records mirror the CUDA layouts the samples
// observe (field names and arity); they are host-side descriptors with no
// kernel-observed layout, so the same policy as the CUmem* records applies.
struct cudaGraphNodeRef {
  uint64_t graph;
  uint32_t index;
};
using cudaGraph_t = detail::AscifyGraphObj *;
using cudaGraphExec_t = detail::AscifyGraphExecObj *;
using cudaGraphNode_t = cudaGraphNodeRef;

enum cudaStreamCaptureMode {
  cudaStreamCaptureModeGlobal = 0,
  cudaStreamCaptureModeThreadLocal = 1,
  cudaStreamCaptureModeRelaxed = 2
};

struct cudaMemsetParams {
  void *dst;
  int value;
  size_t pitch;
  unsigned int elementSize;
  size_t width;
  size_t height;
};

struct cudaHostNodeParams {
  cudaHostFn_t fn;
  void *userData;
};

// dim3 is native in the DPP host environment; the plain-C++ header test
// build has no dim3, so an equivalent fallback keeps the header parsable
// there (only field names x/y/z are observed by this layer).
#if defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)
struct cudaKernelNodeParams {
  void *func;  // converter bridge: typed <<<>>> thunk address
  ::dim3 gridDim;
  ::dim3 blockDim;
  unsigned int sharedMemBytes;
  void **kernelParams;
  void **extra;
};
#else
struct cudaGraphDim3Fallback {
  unsigned int x, y, z;
};
struct cudaKernelNodeParams {
  void *func;
  cudaGraphDim3Fallback gridDim;
  cudaGraphDim3Fallback blockDim;
  unsigned int sharedMemBytes;
  void **kernelParams;
  void **extra;
};
#endif

enum cudaMemAllocationType {
  cudaMemAllocationTypeInvalid = 0x0,
  cudaMemAllocationTypePinned = 0x1,
  cudaMemAllocationTypeManaged = 0x2
};
enum cudaMemAllocationHandleType {
  cudaMemHandleTypeNone = 0x0
};
struct cudaMemPoolProps {
  cudaMemAllocationType allocType;
  cudaMemAllocationHandleType handleTypes;
  cudaMemLocation location;
  void *win32SecurityAttributes;
  size_t maxSize;
  unsigned short usage;
  unsigned char reserved[54];
};
struct cudaMemAllocNodeParams {
  cudaMemPoolProps poolProps;
  const cudaMemAccessDesc *accessDescs;
  size_t accessDescCount;
  size_t bytesize;
  void *dptr;
};

namespace detail {

using GraphRealLaunchV2 = int32_t (*)(const void *, uint32_t, void *, void *,
                                      void *, uint32_t, void *);
inline GraphRealLaunchV2 graphRealLaunchV2() {
  static GraphRealLaunchV2 entry = reinterpret_cast<GraphRealLaunchV2>(
      nextRuntimeSymbol("rtKernelLaunchWithFlagV2"));
  return entry;
}
using GraphRealLaunchLegacy = int32_t (*)(const void *, uint32_t, void *,
                                          uint32_t, void *, void *);
inline GraphRealLaunchLegacy graphRealLaunchLegacy() {
  static GraphRealLaunchLegacy entry = reinterpret_cast<GraphRealLaunchLegacy>(
      nextRuntimeSymbol("rtKernelLaunch"));
  return entry;
}

// cudaMemcpyDefault classification for replayed copies: a pointer is
// device memory exactly when it lies in a registered cudaMalloc allocation.
inline bool graphPointerIsDeviceMemory(const void *pointer) {
  if (pointer == nullptr) return false;
  RuntimeLockGuard guard(runtimeAllocationRegistry().lock);
  return runtimeAllocationLocked(pointer) != nullptr;
}

inline aclrtMemcpyKind graphClassifyMemcpyKind(void *dst, const void *src,
                                               aclrtMemcpyKind kind) {
  if (kind != cudaMemcpyDefault) return kind;
  const bool dstDevice = graphPointerIsDeviceMemory(dst);
  const bool srcDevice = graphPointerIsDeviceMemory(src);
  if (dstDevice && srcDevice) return ACL_MEMCPY_DEVICE_TO_DEVICE;
  if (dstDevice) return ACL_MEMCPY_HOST_TO_DEVICE;
  if (srcDevice) return ACL_MEMCPY_DEVICE_TO_HOST;
  return ACL_MEMCPY_HOST_TO_HOST;
}

inline aclError graphAppendNode(AscifyGraphObj *graph, GraphNode &node,
                                const cudaGraphNode_t *dependencies,
                                size_t dependency_count, uint32_t *index_out) {
  if (graph == nullptr || graph->node_count == 64)
    return ACL_ERROR_BAD_ALLOC;
  if (dependency_count > 8) return ACL_ERROR_RT_PARAM_INVALID;
  for (size_t i = 0; i < dependency_count; ++i) {
    if (dependencies[i].graph != graph->id ||
        dependencies[i].index >= graph->node_count)
      return ACL_ERROR_RT_PARAM_INVALID;
  }
  node.dep_count = dependency_count;
  for (size_t i = 0; i < dependency_count; ++i)
    node.deps[i] = dependencies[i].index;
  *index_out = static_cast<uint32_t>(graph->node_count);
  graph->nodes[graph->node_count++] = node;
  return ACL_SUCCESS;
}

inline aclError graphTopologicalOrder(AscifyGraphObj *graph,
                                      AscifyGraphExecObj *exec) {
  size_t indegree[64] = {};
  bool consumed[64] = {};
  for (size_t i = 0; i < graph->node_count; ++i) {
    const GraphNode &node = graph->nodes[i];
    for (size_t d = 0; d < node.dep_count; ++d) {
      if (node.deps[d] >= graph->node_count)
        return ACL_ERROR_RT_PARAM_INVALID;
      ++indegree[node.deps[d]];
    }
  }
  // Kahn's algorithm preferring lower node indices: deterministic and, for
  // graphs built in submission order, identical to the recorded order.
  size_t produced = 0;
  while (produced < graph->node_count) {
    bool progressed = false;
    for (size_t i = 0; i < graph->node_count; ++i) {
      if (consumed[i] || indegree[i] != 0) continue;
      consumed[i] = true;
      exec->order[produced++] = static_cast<uint32_t>(i);
      progressed = true;
    }
    if (!progressed) return ACL_ERROR_RT_PARAM_INVALID;  // cycle
  }
  exec->order_count = produced;
  return ACL_SUCCESS;
}

inline aclError graphReplayNode(AscifyGraphObj *graph, const GraphNode &node,
                                aclrtStream stream) {
  switch (node.type) {
    case GraphNodeKernelV2: {
      if (graphRealLaunchV2() == nullptr) return ACL_ERROR_FEATURE_UNSUPPORTED;
      uint8_t *scratch = nullptr;
      if (node.blob_bytes != 0) {
        scratch = static_cast<uint8_t *>(malloc(node.blob_bytes));
        if (scratch == nullptr) return ACL_ERROR_BAD_ALLOC;
        __builtin_memcpy(scratch, node.blob, node.blob_bytes);
      }
      AscifyRtArgsEx args = node.args_ex;
      args.args = scratch;
      stagePeerKernelArguments(scratch, node.blob_bytes);
      managedStageKernelArguments(scratch, node.blob_bytes, stream);
      void *const cfg = node.has_cfg ? const_cast<uint8_t *>(node.cfg)
                                     : nullptr;
      const int32_t status = graphRealLaunchV2()(
          node.stub, node.num_blocks, &args, node.sm_desc, stream, node.flags,
          cfg);
      managedPublishAfterLaunch(stream);
      free(scratch);
      return status == 0 ? ACL_SUCCESS : ACL_ERROR_RT_PARAM_INVALID;
    }
    case GraphNodeKernelLegacy: {
      if (graphRealLaunchLegacy() == nullptr)
        return ACL_ERROR_FEATURE_UNSUPPORTED;
      uint8_t *scratch = nullptr;
      if (node.blob_bytes != 0) {
        scratch = static_cast<uint8_t *>(malloc(node.blob_bytes));
        if (scratch == nullptr) return ACL_ERROR_BAD_ALLOC;
        __builtin_memcpy(scratch, node.blob, node.blob_bytes);
      }
      stagePeerKernelArguments(scratch, node.blob_bytes);
      managedStageKernelArguments(scratch, node.blob_bytes, stream);
      const int32_t status = graphRealLaunchLegacy()(
          node.stub, node.num_blocks, scratch, node.blob_bytes, node.sm_desc,
          stream);
      managedPublishAfterLaunch(stream);
      free(scratch);
      return status == 0 ? ACL_SUCCESS : ACL_ERROR_RT_PARAM_INVALID;
    }
    case GraphNodeKernelThunk: {
      if (node.manual.thunk == nullptr) return ACL_ERROR_RT_PARAM_INVALID;
      // The typed <<<>>> thunk re-enters the runtime launch ABI, so peer
      // staging and managed staging run on its freshly packed blob.
      node.manual.thunk(node.manual.grid_x, node.manual.grid_y,
                        node.manual.grid_z, node.manual.block_x,
                        node.manual.block_y, node.manual.block_z,
                        node.manual.shared_mem_bytes, stream,
                        node.manual.kernel_params);
      return ACL_SUCCESS;
    }
    case GraphNodeMemcpy: {
      const aclrtMemcpyKind kind =
          graphClassifyMemcpyKind(node.dst, node.src, node.kind);
      // A blocking copy preserves the sequential dependency contract of
      // the replay (the documented concurrency degradation).
      if (kind != ACL_MEMCPY_HOST_TO_HOST) {
        const aclError drained = aclrtSynchronizeStream(nullptr);
        if (drained != ACL_SUCCESS) return drained;
      }
      return aclrtMemcpy(node.dst, node.bytes, node.src, node.bytes, kind);
    }
    case GraphNodeMemset:
      return aclrtMemset(node.dst, node.bytes, node.value, node.bytes);
    case GraphNodeMemAlloc: {
      RuntimeLockGuard guard(graph->lock);
      if (node.alloc_slot >= graph->alloc_count)
        return ACL_ERROR_RT_PARAM_INVALID;
      graph->allocs[node.alloc_slot].live = true;
      return ACL_SUCCESS;
    }
    case GraphNodeMemFree:
      graphMarkOwnedPointerDead(node.free_ptr);
      return ACL_SUCCESS;
    case GraphNodeHost:
      if (node.host_fn == nullptr) return ACL_ERROR_RT_PARAM_INVALID;
      node.host_fn(node.host_data);
      return ACL_SUCCESS;
  }
  return ACL_ERROR_RT_PARAM_INVALID;
}

}  // namespace detail

inline aclError cudaGraphCreate(cudaGraph_t *graph, unsigned int flags) {
  if (graph == nullptr) return ACL_ERROR_RT_PARAM_INVALID;
  *graph = nullptr;
  if (flags != 0U) return ACL_ERROR_RT_PARAM_INVALID;
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) return ready;
  detail::AscifyGraphObj *object = detail::newGraphObject();
  if (object == nullptr) return ACL_ERROR_BAD_ALLOC;
  *graph = object;
  return ACL_SUCCESS;
}

inline aclError cudaGraphDestroy(cudaGraph_t graph) {
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) return ready;
  return detail::graphReleaseReference(graph);
}

inline aclError cudaGraphAddKernelNode(cudaGraphNode_t *node,
                                       cudaGraph_t graph,
                                       const cudaGraphNode_t *dependencies,
                                       size_t dependency_count,
                                       cudaKernelNodeParams *params) {
  if (node == nullptr || params == nullptr)
    return ACL_ERROR_RT_PARAM_INVALID;
  *node = {};
  if (params->extra != nullptr) return ACL_ERROR_FEATURE_UNSUPPORTED;
  if (params->kernelParams == nullptr) return ACL_ERROR_RT_PARAM_INVALID;
  detail::GraphNode record = {};
  record.type = detail::GraphNodeKernelThunk;
  record.manual.thunk = reinterpret_cast<void (*)(
      unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
      aclrtStream, void **)>(params->func);
  record.manual.grid_x = params->gridDim.x;
  record.manual.grid_y = params->gridDim.y;
  record.manual.grid_z = params->gridDim.z;
  record.manual.block_x = params->blockDim.x;
  record.manual.block_y = params->blockDim.y;
  record.manual.block_z = params->blockDim.z;
  record.manual.shared_mem_bytes = params->sharedMemBytes;
  record.manual.kernel_params = params->kernelParams;
  detail::RuntimeLockGuard guard(graph->lock);
  uint32_t index = 0;
  const aclError appended = detail::graphAppendNode(
      graph, record, dependencies, dependency_count, &index);
  if (appended != ACL_SUCCESS) return appended;
  *node = {graph->id, index};
  return ACL_SUCCESS;
}

inline aclError cudaGraphAddMemcpyNode1D(cudaGraphNode_t *node,
                                         cudaGraph_t graph,
                                         const cudaGraphNode_t *dependencies,
                                         size_t dependency_count, void *dst,
                                         const void *src, size_t count,
                                         aclrtMemcpyKind kind) {
  if (node == nullptr) return ACL_ERROR_RT_PARAM_INVALID;
  *node = {};
  if (!detail::isSupportedMemcpyKind(kind)) return ACL_ERROR_RT_PARAM_INVALID;
  detail::GraphNode record = {};
  record.type = detail::GraphNodeMemcpy;
  record.dst = dst;
  record.src = src;
  record.bytes = count;
  record.kind = kind;
  detail::RuntimeLockGuard guard(graph->lock);
  uint32_t index = 0;
  const aclError appended = detail::graphAppendNode(
      graph, record, dependencies, dependency_count, &index);
  if (appended != ACL_SUCCESS) return appended;
  *node = {graph->id, index};
  return ACL_SUCCESS;
}

inline aclError cudaGraphAddMemsetNode(cudaGraphNode_t *node,
                                       cudaGraph_t graph,
                                       const cudaGraphNode_t *dependencies,
                                       size_t dependency_count,
                                       const cudaMemsetParams *params) {
  if (node == nullptr || params == nullptr)
    return ACL_ERROR_RT_PARAM_INVALID;
  *node = {};
  if (params->elementSize == 0U || params->elementSize > 4U)
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  detail::GraphNode record = {};
  record.type = detail::GraphNodeMemset;
  record.dst = params->dst;
  record.value = params->value;
  record.bytes = params->width * params->height * params->elementSize;
  detail::RuntimeLockGuard guard(graph->lock);
  uint32_t index = 0;
  const aclError appended = detail::graphAppendNode(
      graph, record, dependencies, dependency_count, &index);
  if (appended != ACL_SUCCESS) return appended;
  *node = {graph->id, index};
  return ACL_SUCCESS;
}

inline aclError cudaGraphAddHostNode(cudaGraphNode_t *node,
                                     cudaGraph_t graph,
                                     const cudaGraphNode_t *dependencies,
                                     size_t dependency_count,
                                     const cudaHostNodeParams *params) {
  if (node == nullptr || params == nullptr)
    return ACL_ERROR_RT_PARAM_INVALID;
  *node = {};
  if (params->fn == nullptr) return ACL_ERROR_RT_PARAM_INVALID;
  detail::GraphNode record = {};
  record.type = detail::GraphNodeHost;
  record.host_fn = params->fn;
  record.host_data = params->userData;
  detail::RuntimeLockGuard guard(graph->lock);
  uint32_t index = 0;
  const aclError appended = detail::graphAppendNode(
      graph, record, dependencies, dependency_count, &index);
  if (appended != ACL_SUCCESS) return appended;
  *node = {graph->id, index};
  return ACL_SUCCESS;
}

inline aclError cudaGraphAddMemAllocNode(cudaGraphNode_t *node,
                                         cudaGraph_t graph,
                                         const cudaGraphNode_t *dependencies,
                                         size_t dependency_count,
                                         cudaMemAllocNodeParams *params) {
  if (node == nullptr || params == nullptr)
    return ACL_ERROR_RT_PARAM_INVALID;
  *node = {};
  if (params->bytesize == 0) return ACL_ERROR_RT_PARAM_INVALID;
  if (params->poolProps.allocType != cudaMemAllocationTypePinned)
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  if (params->poolProps.location.type != cudaMemLocationTypeDevice)
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  detail::RuntimeLockGuard guard(graph->lock);
  void *assigned = nullptr;
  uint32_t slot = 0;
  const aclError assigned_status = detail::graphAssignAllocationLocked(
      graph, params->bytesize, &assigned, &slot);
  if (assigned_status != ACL_SUCCESS) return assigned_status;
  detail::GraphNode record = {};
  record.type = detail::GraphNodeMemAlloc;
  record.alloc_slot = slot;
  uint32_t index = 0;
  const aclError appended = detail::graphAppendNode(
      graph, record, dependencies, dependency_count, &index);
  if (appended != ACL_SUCCESS) {
    // The address was published to the caller only on success.
    for (size_t s = 0; s < graph->alloc_count; ++s)
      if (graph->allocs[s].pointer == assigned)
        graph->allocs[s].live = false;
    return appended;
  }
  graph->has_alloc_nodes = true;
  params->dptr = assigned;  // CUDA assigns the address at node-add time
  *node = {graph->id, index};
  return ACL_SUCCESS;
}

inline aclError cudaGraphAddMemFreeNode(cudaGraphNode_t *node,
                                        cudaGraph_t graph,
                                        const cudaGraphNode_t *dependencies,
                                        size_t dependency_count, void *dptr) {
  if (node == nullptr) return ACL_ERROR_RT_PARAM_INVALID;
  *node = {};
  if (dptr == nullptr) return ACL_ERROR_RT_PARAM_INVALID;
  detail::GraphNode record = {};
  record.type = detail::GraphNodeMemFree;
  record.free_ptr = dptr;
  detail::RuntimeLockGuard guard(graph->lock);
  uint32_t index = 0;
  const aclError appended = detail::graphAppendNode(
      graph, record, dependencies, dependency_count, &index);
  if (appended != ACL_SUCCESS) return appended;
  *node = {graph->id, index};
  // CUDA frees the virtual address at build time (that is what makes the
  // documented address reuse observable to the builder); the physical
  // block stays mapped for address-stable replays.
  for (size_t s = 0; s < graph->alloc_count; ++s)
    if (graph->allocs[s].pointer == dptr) graph->allocs[s].live = false;
  return ACL_SUCCESS;
}

inline aclError cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t *nodes,
                                  size_t *numNodes) {
  if (numNodes == nullptr) return ACL_ERROR_RT_PARAM_INVALID;
  detail::RuntimeLockGuard guard(graph->lock);
  if (nodes == nullptr) {
    *numNodes = graph->node_count;
    return ACL_SUCCESS;
  }
  size_t capacity = *numNodes;
  *numNodes = graph->node_count;
  for (size_t i = 0; i < graph->node_count && i < capacity; ++i)
    nodes[i] = {graph->id, static_cast<uint32_t>(i)};
  return ACL_SUCCESS;
}

inline aclError cudaGraphInstantiateExec(cudaGraphExec_t *exec,
                                         cudaGraph_t graph) {
  if (exec == nullptr || graph == nullptr)
    return ACL_ERROR_RT_PARAM_INVALID;
  *exec = nullptr;
  detail::AscifyGraphExecObj *object = static_cast<detail::AscifyGraphExecObj *>(
      calloc(1, sizeof(detail::AscifyGraphExecObj)));
  if (object == nullptr) return ACL_ERROR_BAD_ALLOC;
  object->graph = graph;
  {
    detail::RuntimeLockGuard guard(graph->lock);
    const aclError ordered =
        detail::graphTopologicalOrder(graph, object);
    if (ordered != ACL_SUCCESS) {
      free(object);
      return ordered;
    }
  }
  {
    detail::RuntimeLockGuard guard(detail::graphRegistry().lock);
    ++graph->refs;
  }
  *exec = object;
  return ACL_SUCCESS;
}

inline aclError cudaGraphInstantiate(cudaGraphExec_t *pGraphExec,
                                     cudaGraph_t graph,
                                     cudaGraphNode_t *pErrorNode,
                                     char *pErrorLogBuffer,
                                     size_t bufferSize) {
  (void)pErrorLogBuffer;
  (void)bufferSize;
  const aclError status = cudaGraphInstantiateExec(pGraphExec, graph);
  if (status != ACL_SUCCESS && pErrorNode != nullptr) *pErrorNode = {};
  return status;
}

inline aclError cudaGraphInstantiate(cudaGraphExec_t *pGraphExec,
                                     cudaGraph_t graph,
                                     unsigned long long flags) {
  if (flags != 0ULL) return ACL_ERROR_FEATURE_UNSUPPORTED;
  return cudaGraphInstantiateExec(pGraphExec, graph);
}

inline aclError cudaGraphExecDestroy(cudaGraphExec_t exec) {
  if (exec == nullptr) return ACL_ERROR_RT_PARAM_INVALID;
  const aclError released = detail::graphReleaseReference(exec->graph);
  free(exec);
  return released;
}

inline aclError cudaGraphClone(cudaGraph_t *clone, cudaGraph_t graph) {
  if (clone == nullptr || graph == nullptr)
    return ACL_ERROR_RT_PARAM_INVALID;
  *clone = nullptr;
  detail::AscifyGraphObj *copy = detail::newGraphObject();
  if (copy == nullptr) return ACL_ERROR_BAD_ALLOC;
  detail::RuntimeLockGuard guard(graph->lock);
  if (graph->has_alloc_nodes) {
    // Graph-owned allocation slots cannot be duplicated (each clone would
    // own the same physical block); a clone of a memory-node graph is not
    // admitted by this emulation.
    (void)detail::graphReleaseReference(copy);
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  }
  copy->node_count = graph->node_count;
  for (size_t i = 0; i < graph->node_count; ++i) {
    copy->nodes[i] = graph->nodes[i];
    copy->nodes[i].blob = nullptr;
    copy->nodes[i].blob_bytes = graph->nodes[i].blob_bytes;
    if (graph->nodes[i].blob != nullptr) {
      copy->nodes[i].blob =
          detail::graphCopyBlob(graph->nodes[i].blob, graph->nodes[i].blob_bytes);
      if (copy->nodes[i].blob == nullptr) {
        copy->node_count = i;
        (void)detail::graphReleaseReference(copy);
        return ACL_ERROR_BAD_ALLOC;
      }
    }
  }
  *clone = copy;
  return ACL_SUCCESS;
}

inline aclError cudaGraphLaunch(cudaGraphExec_t exec, aclrtStream stream) {
  if (exec == nullptr) return ACL_ERROR_RT_PARAM_INVALID;
  if (detail::graphCaptureActive())
    return ACL_ERROR_FEATURE_UNSUPPORTED;  // nested graph capture
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) return ready;
  detail::AscifyGraphObj *graph = exec->graph;
  for (size_t i = 0; i < exec->order_count; ++i) {
    const uint32_t index = exec->order[i];
    if (index >= graph->node_count) return ACL_ERROR_RT_PARAM_INVALID;
    const aclError status =
        detail::graphReplayNode(graph, graph->nodes[index], stream);
    if (status != ACL_SUCCESS) return status;
  }
  return ACL_SUCCESS;
}

inline aclError cudaStreamBeginCapture(aclrtStream stream,
                                       cudaStreamCaptureMode mode) {
  if (stream == nullptr) return ACL_ERROR_RT_PARAM_INVALID;
  if (mode != cudaStreamCaptureModeGlobal &&
      mode != cudaStreamCaptureModeThreadLocal &&
      mode != cudaStreamCaptureModeRelaxed)
    return ACL_ERROR_RT_PARAM_INVALID;
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) return ready;
  detail::GraphCaptureState &capture = detail::graphCaptureState();
  if (capture.active) return ACL_ERROR_RT_PARAM_INVALID;
  detail::AscifyGraphObj *graph = detail::newGraphObject();
  if (graph == nullptr) return ACL_ERROR_BAD_ALLOC;
  capture.active = true;
  capture.graph = graph;
  capture.root = stream;
  capture.stream_count = 1;
  capture.streams[0] = stream;
  capture.tail[0] = -1;
  capture.pending_count[0] = 0;
  capture.event_count = 0;
  capture.invalid = false;
  capture.error = ACL_SUCCESS;
  return ACL_SUCCESS;
}

inline aclError cudaStreamEndCapture(aclrtStream stream, cudaGraph_t *graph) {
  if (graph == nullptr || stream == nullptr)
    return ACL_ERROR_RT_PARAM_INVALID;
  *graph = nullptr;
  detail::GraphCaptureState &capture = detail::graphCaptureState();
  if (!capture.active) return ACL_ERROR_RT_PARAM_INVALID;
  if (stream != capture.root) return ACL_ERROR_RT_PARAM_INVALID;
  detail::AscifyGraphObj *recorded = capture.graph;
  const bool invalid = capture.invalid;
  const aclError error = capture.error;
  capture.active = false;
  capture.graph = nullptr;
  capture.root = nullptr;
  capture.stream_count = 0;
  capture.event_count = 0;
  capture.invalid = false;
  capture.error = ACL_SUCCESS;
  if (invalid) {
    (void)detail::graphReleaseReference(recorded);
    return error == ACL_SUCCESS ? ACL_ERROR_RT_PARAM_INVALID : error;
  }
  *graph = recorded;
  return ACL_SUCCESS;
}

// ---- managed memory (cudaMallocManaged) ------------------------------------
inline constexpr unsigned int cudaMemAttachGlobal = 0U;
inline constexpr unsigned int cudaMemAttachHost = 1U;
inline constexpr unsigned int cudaMemAttachSingle = 0x2U;

inline aclError cudaMallocManaged(void **pointer, size_t bytes,
                                  unsigned int flags) {
  if (pointer == nullptr) return ACL_ERROR_RT_PARAM_INVALID;
  *pointer = nullptr;
  if (flags != cudaMemAttachGlobal && flags != cudaMemAttachHost &&
      flags != cudaMemAttachSingle)
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) return ready;
  void *host = calloc(bytes == 0 ? 1 : bytes, 1);
  unsigned char *shadow = static_cast<unsigned char *>(calloc(bytes == 0 ? 1 : bytes, 1));
  if (host == nullptr || shadow == nullptr) {
    free(host);
    free(shadow);
    return ACL_ERROR_BAD_ALLOC;
  }
  void *device = nullptr;
  const aclError status =
      aclrtMalloc(&device, bytes == 0 ? 1 : bytes, ACL_MEM_MALLOC_HUGE_FIRST);
  if (status != ACL_SUCCESS) {
    free(host);
    free(shadow);
    return status;
  }
  // CUDA demand-paging observes zero-filled pages on both sides.
  (void)aclrtMemset(device, bytes == 0 ? 1 : bytes, 0, bytes == 0 ? 1 : bytes);
  {
    detail::RuntimeLockGuard guard(detail::managedTable().lock);
    if (detail::managedTable().count == 16) {
      (void)aclrtFree(device);
      free(host);
      free(shadow);
      return ACL_ERROR_BAD_ALLOC;
    }
    detail::managedTable().records[detail::managedTable().count++] =
        {host, device, shadow, bytes};
  }
  *pointer = host;
  return ACL_SUCCESS;
}

template <typename T>
inline aclError cudaMallocManaged(T **pointer, size_t bytes,
                                  unsigned int flags = cudaMemAttachGlobal) {
  if (pointer == nullptr) return ACL_ERROR_RT_PARAM_INVALID;
  *pointer = nullptr;
  void *raw = nullptr;
  const aclError status = cudaMallocManaged(&raw, bytes, flags);
  if (status == ACL_SUCCESS) *pointer = static_cast<T *>(raw);
  return status;
}

// The translated runtime reports the SDK version the converter admitted
// (value-pinned CUDART_VERSION); graph memory nodes require >= 11040 and
// the emulated capability answers the sample's waiver gates truthfully
// for this implementation.
inline aclError cudaDriverGetVersion(int *version) {
  if (version == nullptr) return ACL_ERROR_RT_PARAM_INVALID;
#if defined(CUDART_VERSION)
  *version = CUDART_VERSION;
#else
  *version = 13040;
#endif
  return ACL_SUCCESS;
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

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Frozen NVIDIA SM tables used by original Samples workload heuristics.
// These return legacy table values, never NPU physical CUDA-core counts.
// The frontend selects the exact authenticated helper version. Preserve
// each version's fallback printf, including the v13.0 missing-0xa7 case.
inline int sampleSMVersionToCoresV130(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {0x80,  64},
      {0x86, 128},
      {0x87, 128},
      {0x89, 128},
      {0x90, 128},
      {0xa0, 128},
      {0xa1, 128},
      {0xa3, 128},
      {0xb0, 128},
      {0xc0, 128},
      {0xc1, 128},
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  ::printf(
      "MapSMtoCores for SM %d.%d is undefined."
      "  Default to use %d Cores/SM\n",
      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}

inline int sampleSMVersionToCoresV134(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {0x80,  64},
      {0x86, 128},
      {0x87, 128},
      {0x89, 128},
      {0x90, 128},
      {0xa0, 128},
      {0xa1, 128},
      {0xa3, 128},
      {0xa7, 128},
      {0xb0, 128},
      {0xc0, 128},
      {0xc1, 128},
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  ::printf(
      "MapSMtoCores for SM %d.%d is undefined."
      "  Default to use %d Cores/SM\n",
      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
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
  if (count < 1) {
    detail::sampleFindCudaDeviceFailure(
        "exactly one visible logical device is required", ACL_SUCCESS);
  }
  // The original helper ranks visible devices by GFLOPS and picks the fastest.
  // This porting layer documents one uniform epoch performance tier, so any
  // non-empty visible set selects logical device zero; --device=0 stays the
  // one explicit selector, and other ordinals keep failing exactly as before.

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
// Device-global, sense-reversing grid barrier state. Every observation and
// update goes through device atomics, so all co-resident blocks observe a
// coherent epoch without relying on plain-store visibility across cores.
__aicore__ unsigned int asc_grid_barrier_count = 0;
__aicore__ unsigned int asc_grid_barrier_sense = 0;

// Cooperative grid-wide rendezvous for co-resident blocks (the cooperative
// occupancy advice is the admission contract; a grid larger than what can be
// co-resident would legitimately hang, same as on CUDA hardware). Each block
// contributes one ticket; the last arriver resets the count and flips the
// sense; every other block spins on an atomic read until the flip. The same
// admitted pure-SIMT gate as the generic atomics owns the implementation.
#if defined(ASCIFY_SIMT_LEGACY_HAS_GM_ATOMICS) && \
    defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)
__aicore__ ASCIFY_FORCEINLINE void grid_barrier_sync(unsigned blocks) {
  const unsigned int local_sense =
      ::ascify::atomic_add_generic(&asc_grid_barrier_sense, 0u);
  const unsigned int ticket =
      ::ascify::atomic_add_generic(&asc_grid_barrier_count, 1u);
  if (ticket == blocks - 1u) {
    ::ascify::atomic_exch_generic(&asc_grid_barrier_count, 0u);
    ::ascify::atomic_exch_generic(&asc_grid_barrier_sense, local_sense ^ 1u);
  } else {
    while (::ascify::atomic_add_generic(&asc_grid_barrier_sense, 0u) ==
           local_sense) {
    }
  }
}
#else
__aicore__ ASCIFY_FORCEINLINE void grid_barrier_sync(unsigned) {
  // Non-target environments parse the headers without the pure-SIMT gate;
  // no real definition of the barrier exists there and the dynamic-warp
  // projection never selects this TU shape for device generation.
}
#endif

struct UnsupportedCudaSmAttribute {};
inline constexpr aclrtDevAttr cudaDevAttrComputeCapabilityMajor =
    static_cast<aclrtDevAttr>(79000001U);
inline constexpr aclrtDevAttr cudaDevAttrComputeCapabilityMinor =
    static_cast<aclrtDevAttr>(79000002U);
inline constexpr aclrtDevAttr cudaDevAttrCooperativeLaunch =
    static_cast<aclrtDevAttr>(79000003U);
// GpuOverlap is answered by the same reviewed-capability policy as the
// epoch pair: aclrt has no such enum member, and cudaMemcpyAsync traffic on
// independent streams overlapping kernel execution was verified on this
// target, so the selector reports that reviewed capability.
inline constexpr aclrtDevAttr cudaDevAttrGpuOverlap =
    static_cast<aclrtDevAttr>(79000004U);

// The capability-epoch pair documented on cudaDeviceProp (7.0) is also the
// value these selector tags report: the reviewed SIMT admission set, not an
// NVIDIA SM hardware readout (CANN exposes none). Every other SM-shaped
// attribute stays unsupported and carries no fabricated number.

inline aclError cudaDeviceGetAttribute(int* value, aclrtDevAttr attribute, int device) {
  if (value == nullptr || device < 0) { return ACL_ERROR_RT_PARAM_INVALID; }

  // The capability-epoch selector tags are Ascify-private encodings: they
  // document the cudaDeviceProp epoch pair and are answered here instead of
  // being forwarded to aclrt, whose attribute enum has no such member.
  if (attribute == cudaDevAttrComputeCapabilityMajor) { *value = 7; return ACL_SUCCESS; }
  if (attribute == cudaDevAttrComputeCapabilityMinor) { *value = 0; return ACL_SUCCESS; }
  if (attribute == cudaDevAttrMemoryPoolsSupported) { *value = 1; return ACL_SUCCESS; }
  // Cooperative launch is implemented on this target (direct cooperative
  // launch plus the admitted grid rendezvous), so the reviewed capability
  // answers 1; the same contract cudaDeviceProp::cooperativeLaunch reports.
  if (attribute == cudaDevAttrCooperativeLaunch) { *value = 1; return ACL_SUCCESS; }
  // Reviewed capability, not an aclrt readout: copy/kernel overlap on
  // independent streams was measured on device (see asyncEngineCount).
  if (attribute == cudaDevAttrGpuOverlap) { *value = 1; return ACL_SUCCESS; }

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
  // Capability epoch, not an NVIDIA SM version claim. The admitted value 7.0
  // documents the reviewed SIMT feature set of this porting layer (full-warp
  // shuffle/vote/masked ballots, atomics, streams/events, occupancy
  // estimates). Samples gating on compute capability >= 2/3/3.5 therefore
  // proceed for features this layer implements; features outside the epoch
  // (device-side recursion, WMMA, CDP) are still refused by their own
  // conversion gates and never pass because of these fields.
  int major;
  int minor;
  // Native mapped-host support (aclrtHostRegister/aclrtHostGetDevicePointer)
  // exists on this target, so canMapHostMemory reports a real reviewed
  // capability rather than a fabricated mode bit.
  int canMapHostMemory;
  int warpSize;
  int maxThreadsPerMultiProcessor;
  size_t totalGlobalMem;
  int maxThreadsPerBlock;
  int maxGridSize[3];
  // The cooperative-launch surface (grid-wide rendezvous backend plus the
  // occupancy advice) is a real reviewed capability of this porting layer.
  int cooperativeLaunch;
  // Concurrent asynchronous copy flows (CUDA's asyncEngineCount diagnostic):
  // the porting layer interleaves aclrtMemcpyAsync traffic on independent
  // streams with kernel execution, verified on device. The value reviews
  // that concurrent-flow capability; aclrt exposes no DMA engine count.
  int asyncEngineCount;
};

inline aclError cudaGetDeviceProperties(cudaDeviceProp* properties, int device) {
  if (properties == nullptr || device < 0) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  const aclError initialized = detail::runtime_manager.ensureInitialized();
  if (initialized != ACL_SUCCESS) {
    return detail::rememberLifecycleError(initialized);
  }
  // GetSocName reports runtime Device 0, even when another device is bound.
  // Neither it nor GetDeviceInfo requires a current context. Preserve CUDA's
  // query-before-SetDevice behavior without binding/resetting the caller or
  // reporting Device 0's name as the identity of a different device.
  // ACL property queries need a bound context; non-zero devices are
  // queried after a temporary bind (matching the CUDA model where each
  // device has its own context).
  aclrtContext savedContext = nullptr;
  const bool rebind = device != 0;
  if (rebind) {
    const aclError bound = detail::runtime_manager.bindDevice(device);
    if (bound != ACL_SUCCESS) { return bound; }
    (void)aclrtGetCurrentContext(&savedContext);
  }
  int64_t cores = 0;
  aclError status = aclrtGetDeviceInfo(static_cast<uint32_t>(device),
                                     ACL_DEV_ATTR_VECTOR_CORE_NUM, &cores);
  if (status != ACL_SUCCESS) { return status; }
  if (cores <= 0 || cores > INT_MAX) { return ACL_ERROR_FEATURE_UNSUPPORTED; }
  // CANN's total available global memory is measured in bytes. Keep the
  // runtime's full result width until it has been checked against size_t.
  // Attribute 301 is a frozen RTS ABI value; older headers may omit its name.
  int64_t memory = 0;
  status = aclrtGetDeviceInfo(static_cast<uint32_t>(device),
                            static_cast<aclrtDevAttr>(301U), &memory);
  if (status != ACL_SUCCESS) { return status; }
  if (memory <= 0 || static_cast<uint64_t>(memory) >
                         static_cast<uint64_t>(SIZE_MAX)) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  }
  const char* const soc_name = aclrtGetSocName();
  if (soc_name == nullptr || soc_name[0] == '\0') {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  }
  cudaDeviceProp result = {};
  // Capability epoch 7.0 (see struct comment): the reviewed SIMT admission
  // set. Reported alongside real runtime queries, never from a hardware
  // compute-capability readout because CANN exposes none.
  result.major = 7;
  result.minor = 0;
  // Real native mapped-host support (verified on device): samples gating on
  // canMapHostMemory proceed truthfully.
  result.canMapHostMemory = 1;
  // These are device-wide SIMT limits, not per-kernel occupancy or an SM
  // capability number. Keep the frozen RTS ABI values usable with headers
  // that predate the enum names. A missing/invalid query is an explicit
  // failure, and no part of the caller's object is written on failure.
  const unsigned attributes[] = {209U, 205U, 206U, 207U, 202U, 203U};
  int limits[6] = {};
  for (unsigned i = 0; i != 6; ++i) {
    int64_t raw = 0;
    status = aclrtGetDeviceInfo(static_cast<uint32_t>(device),
                               static_cast<aclrtDevAttr>(attributes[i]), &raw);
    if (status != ACL_SUCCESS) { return status; }
    if (raw <= 0 || raw > INT_MAX) { return ACL_ERROR_FEATURE_UNSUPPORTED; }
    limits[i] = static_cast<int>(raw);
  }
  result.maxThreadsPerBlock = limits[0];
  for (unsigned i = 0; i != 3; ++i) { result.maxGridSize[i] = limits[i + 1]; }
  // warpSize is the runtime's SIMT warp width (attr 202). maxThreadsPerMulti-
  // Processor keeps the reviewed occupancy model's equivalence of a vector
  // core to a CUDA multiprocessor: it reports attr 203, the maximum concurrent
  // threads per vector core. These are device-wide SIMT limits, not an SM
  // capability number; major/minor stay unexposed by design.
  result.warpSize = limits[4];
  result.maxThreadsPerMultiProcessor = limits[5];
  size_t index = 0;
  while (index + 1 < sizeof(result.name) && soc_name[index] != '\0') {
    result.name[index] = soc_name[index];
    ++index;
  }
  result.multiProcessorCount = static_cast<int>(cores);
  result.totalGlobalMem = static_cast<size_t>(memory);
  result.cooperativeLaunch = 1;
  // Two concurrent async copy flows (upload/download) overlap kernels on
  // this porting layer; see the struct comment. Not a hardware DMA count.
  result.asyncEngineCount = 2;
  *properties = result;
  if (rebind && savedContext != nullptr) {
    (void)aclrtSetCurrentContext(savedContext);
  }
  return ACL_SUCCESS;
}

// ---- CUDA Driver API compatibility (vectorAddDrv / simpleDrvRuntime line) --
// Architecture: the translated host TU keeps the NVIDIA helper logic and the
// runtime module-loading contract. cuModuleLoadData really loads the compiled
// kernel image through aclrtCreateBinary/aclrtBinaryLoad (device-segment
// verified) and cross-checks every registry kernel through
// aclrtBinaryGetFunction with the Itanium-mangled name. Kernel launch goes
// through a typed <<<>>> thunk registered beside the kernel definition, which
// is the proven code-generation path for this layer; the runtime handle ABI
// (aclrtLaunchKernel/KernelArgs*) expects a different metadata layout and is
// not used for these SIMT kernels.
namespace drv {

struct ModuleHandle {
  aclrtBinary binary = nullptr;
  aclrtBinHandle handle = nullptr;
  void *image = nullptr;
  size_t size = 0;
};

// Cooperative-launch admission anchor. The translator proves each exact
// call site (kernel address, local argument array of exact parameter
// addresses) and rewrites it to a direct <<<>>> launch, so this runtime
// entry is never selected for an admitted TU; it exists to complete the
// mapped prototype surface for boundary audits.
inline aclError cudaLaunchCooperativeKernel(void *, void *, void *, void **,
                                            size_t, void *) {
  return ACL_ERROR_FEATURE_UNSUPPORTED;
}

// A typed launch thunk unpacks the CUDA kernelParams void* array by the
// kernel's own signature and performs the <<<>>> launch.
using KernelLaunchThunk = void (*)(unsigned, unsigned, unsigned,
                                   unsigned, unsigned, unsigned,
                                   unsigned, aclrtStream, void **);

struct KernelEntry {
  const char *plain;
  const char *mangled;
  KernelLaunchThunk launch;
};

struct KernelRegistry {
  detail::RuntimeLock lock;
  const KernelEntry *entries[64] = {};
  size_t count = 0;
};
inline KernelRegistry &kernelRegistry() {
  static KernelRegistry registry;
  return registry;
}
inline bool registerKernel(const KernelEntry *entry) {
  if (entry == nullptr || entry->plain == nullptr || entry->mangled == nullptr ||
      entry->launch == nullptr)
    return false;
  detail::RuntimeLockGuard guard(kernelRegistry().lock);
  if (kernelRegistry().count == 64) return false;
  kernelRegistry().entries[kernelRegistry().count++] = entry;
  return true;
}
inline const KernelEntry *findKernel(const char *plain) {
  if (plain == nullptr) return nullptr;
  detail::RuntimeLockGuard guard(kernelRegistry().lock);
  for (size_t i = 0; i < kernelRegistry().count; ++i)
    if (__builtin_strcmp(kernelRegistry().entries[i]->plain, plain) == 0)
      return kernelRegistry().entries[i];
  return nullptr;
}

}  // namespace drv

// CUDA Driver type surface kept at the ascify namespace level; translated
// declarations and helper signatures use these after type rewrites.
using CUdevice = int32_t;
using CUcontext = void *;
using CUmodule = drv::ModuleHandle *;
using CUfunction = const drv::KernelEntry *;
using CUdeviceptr = void *;
using CUstream = aclrtStream;
// CUDA 13 context-creation parameter block; the epoch context needs none of
// it, so the translated sample object is an inert carrier.
struct CUctxCreateParams {};

// Driver device attributes: a small, explicitly reviewed epoch surface.
// Same policy as cudaDeviceProp: real queries where CANN exposes the fact,
// the documented 7.0 capability epoch for SM selectors, and explicit failure
// for everything else (no fabricated NVIDIA attributes).
using CUdevice_attribute = int;
inline constexpr CUdevice_attribute CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 1;
inline constexpr CUdevice_attribute CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 2;
inline constexpr CUdevice_attribute CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 3;
inline constexpr CUdevice_attribute CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 4;
inline constexpr CUdevice_attribute CU_DEVICE_ATTRIBUTE_INTEGRATED = 5;
inline constexpr CUdevice_attribute CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 6;

// ---- CUDA legacy stream callbacks (simpleCallback line) --------------------
// cudaStreamAddCallback schedules a host callback that runs after all work
// already submitted to the stream, with the status of the preceding stream
// operations. The native surface is aclrtSubscribeReport (a dedicated
// reporting thread per stream) plus aclrtLaunchCallback (a callback task in
// the stream queue); the trampoline below preserves the CUDA callback
// signature and forwards a non-consuming last-error snapshot exactly like
// the CUDA status parameter contract documents for successful streams.
typedef void (*cudaStreamCallback_t)(aclrtStream stream, int status,
                                     void *userData);

extern "C" unsigned long pthread_self(void);

namespace detail {

struct StreamCallbackTicket {
  aclrtStream stream;
  cudaStreamCallback_t callback;
  void *userData;
};

inline void streamCallbackTrampoline(void *data) {
  StreamCallbackTicket *const ticket =
      static_cast<StreamCallbackTicket *>(data);
  const aclrtStream stream = ticket->stream;
  const cudaStreamCallback_t callback = ticket->callback;
  void *const userData = ticket->userData;
  free(ticket);
  // CUDA hands the callback the status of the preceding stream work. The
  // consuming thread-level query is the real last-error surface here; a
  // completed clean stream reports SUCCESS exactly like CUDA.
  callback(stream, cudaGetLastError(), userData);
}

// CUDA executes stream callbacks on an internal thread automatically. The
// native surface instead needs one subscribed thread that keeps calling
// aclrtProcessReport. A dedicated process-wide worker thread owns every
// stream subscription; producer threads only enqueue the stream once. The
// worker subscribes with its own thread id and then drains callback reports
// with a bounded poll, so sample main threads never block on reports.
// One worker per device: the native report-subscription key of a thread is
// created by that thread's first device bind and is scoped to exactly that
// device (verified on target: subscribing a second device's stream from the
// same thread fails with rtSubscribeReport 107012), so a single process-wide
// worker cannot own stream subscriptions across devices.
struct DeviceCallbackWorker {
  bool started = false;
  bool stopping = false;
  pthread_t thread = 0;
  int32_t device = -1;
  aclrtStream pending[64] = {};
  size_t pendingCount = 0;
  aclrtStream subscribed[64] = {};
  size_t subscribedCount = 0;
};
struct CallbackWorkerTable {
  detail::RuntimeLock lock;
  DeviceCallbackWorker workers[16] = {};
};
inline CallbackWorkerTable &callbackWorkerTable() {
  static CallbackWorkerTable table;
  return table;
}
// Process-exit drain stop: flags every started worker and joins it so no
// aclrtProcessReport call can race the RuntimeManager device reset and
// aclFinalize teardown. Idempotent and safe when no worker ever started.
inline void stopCallbackWorkers() {
  pthread_t threads[16] = {};
  size_t count = 0;
  {
    detail::RuntimeLockGuard guard(callbackWorkerTable().lock);
    for (auto &worker : callbackWorkerTable().workers) {
      if (!worker.started || worker.stopping) { continue; }
      worker.stopping = true;
      threads[count++] = worker.thread;
    }
  }
  for (size_t i = 0; i < count; ++i) {
    (void)pthread_join(threads[i], nullptr);
  }
}
inline void *deviceCallbackWorkerLoop(void *arg) {
  const uintptr_t slot = reinterpret_cast<uintptr_t>(arg);
  const uint64_t self = static_cast<uint64_t>(pthread_self());
  int32_t device = -1;
  {
    detail::RuntimeLockGuard guard(callbackWorkerTable().lock);
    device = callbackWorkerTable().workers[slot].device;
  }
  // The first device bind of this thread creates its report-subscription
  // key, scoped to this device only; this worker never binds another device.
  if (runtime_manager.bindDevice(device) != ACL_SUCCESS) {
    detail::RuntimeLockGuard guard(callbackWorkerTable().lock);
    callbackWorkerTable().workers[slot].stopping = true;
    return nullptr;
  }
  while (true) {
    aclrtStream todo[64];
    size_t count = 0;
    bool stopping = false;
    {
      detail::RuntimeLockGuard guard(callbackWorkerTable().lock);
      DeviceCallbackWorker &worker = callbackWorkerTable().workers[slot];
      stopping = worker.stopping;
      for (size_t i = 0; i < worker.pendingCount && count < 64; ++i)
        todo[count++] = worker.pending[i];
      worker.pendingCount = 0;
    }
    if (stopping) break;
    for (size_t i = 0; i < count; ++i) {
      bool known = false;
      bool full = false;
      {
        detail::RuntimeLockGuard guard(callbackWorkerTable().lock);
        const DeviceCallbackWorker &worker =
            callbackWorkerTable().workers[slot];
        for (size_t k = 0; k < worker.subscribedCount; ++k)
          known = known || worker.subscribed[k] == todo[i];
        full = worker.subscribedCount == 64;
      }
      if (known || full) continue;
      if (aclrtSubscribeReport(self, todo[i]) == 0) {
        detail::RuntimeLockGuard guard(callbackWorkerTable().lock);
        DeviceCallbackWorker &worker = callbackWorkerTable().workers[slot];
        worker.subscribed[worker.subscribedCount++] = todo[i];
      }
    }
    // An idle report wait returns a timeout status and leaves it in this
    // thread's last-error slot (verified on target: repeated aclrtProcessReport
    // calls yield 507012 there). Consume it so the status snapshot taken
    // inside the next report cannot observe drain noise; a report that
    // genuinely fails keeps surfacing through the next callback snapshot.
    if (aclrtProcessReport(100) != ACL_SUCCESS) {
      (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    }
  }
  return nullptr;
}

// The table lock is a non-recursive spinlock: the enqueue below must be a
// separately scoped critical section, and the bounded wait must run with the
// lock released so the device worker can drain the pending queue at all.
inline aclError ensureStreamCallbackSubscription(aclrtStream stream,
                                                 int32_t device) {
  if (device < 0 ||
      static_cast<size_t>(device) >=
          sizeof(callbackWorkerTable().workers) /
              sizeof(callbackWorkerTable().workers[0])) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  {
    detail::RuntimeLockGuard guard(callbackWorkerTable().lock);
    DeviceCallbackWorker &worker =
        callbackWorkerTable().workers[device];
    if (!worker.started) {
      worker.started = true;
      worker.device = device;
      if (pthread_create(&worker.thread, nullptr, deviceCallbackWorkerLoop,
                         reinterpret_cast<void *>(
                             static_cast<uintptr_t>(device))) != 0) {
        worker.started = false;
        worker.device = -1;
        return ACL_ERROR_BAD_ALLOC;
      }
      // atexit handlers run LIFO: the RuntimeManager cleanup callbacks were
      // registered by the earlier runtime calls, so this later registration
      // stops and joins every worker before the device reset and aclFinalize.
      if (::atexit(stopCallbackWorkers) != 0) {
        worker.stopping = true;
        (void)pthread_join(worker.thread, nullptr);
        worker.started = false;
        worker.stopping = false;
        worker.thread = 0;
        worker.device = -1;
        return ACL_ERROR_BAD_ALLOC;
      }
    }
    if (worker.pendingCount == 64) return ACL_ERROR_BAD_ALLOC;
    worker.pending[worker.pendingCount++] = stream;
  }
  // The native launch checks that the stream is already registered, so wait
  // bounded for the device worker's subscription to complete before
  // returning.
  for (int spin = 0; spin < 2000000; ++spin) {
    bool ready = false;
    {
      detail::RuntimeLockGuard guard(callbackWorkerTable().lock);
      const DeviceCallbackWorker &worker =
          callbackWorkerTable().workers[device];
      for (size_t k = 0; k < worker.subscribedCount; ++k)
        ready = ready || worker.subscribed[k] == stream;
    }
    if (ready) return ACL_SUCCESS;
    struct timespec pause = {0, 1000};
    nanosleep(&pause, nullptr);
  }
  return ACL_ERROR_RT_PARAM_INVALID;
}

}  // namespace detail

inline aclError cudaStreamAddCallback(aclrtStream stream,
                                      cudaStreamCallback_t callback,
                                      void *userData, unsigned int flags) {
  if (callback == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  if (flags != 0U) { return ACL_ERROR_FEATURE_UNSUPPORTED; }
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  const aclError subscribed = detail::ensureStreamCallbackSubscription(
      stream, detail::selectedDeviceForThread());
  if (subscribed != ACL_SUCCESS) { return subscribed; }
  detail::StreamCallbackTicket *ticket = static_cast<detail::StreamCallbackTicket *>(
      calloc(1, sizeof(detail::StreamCallbackTicket)));
  if (ticket == nullptr) { return ACL_ERROR_BAD_ALLOC; }
  ticket->stream = stream;
  ticket->callback = callback;
  ticket->userData = userData;
  const aclError status = aclrtLaunchCallback(
      detail::streamCallbackTrampoline, ticket, ACL_CALLBACK_NO_BLOCK, stream);
  if (status != ACL_SUCCESS) { free(ticket); }
  return status;
}

// helper_cuda_drvapi.h checkCudaErrors analogue: the DRV macro is a plain
// three-argument wrapper, mirrored here with the same exit behaviour.
inline void sampleCheckCudaErrorsDrv(aclError status, const char *file, int line) {
  if (status == ACL_SUCCESS) { return; }
  const char *text = cudaGetErrorString(status);
  fprintf(stderr,
          "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
          "line %i.\n",
          static_cast<int>(status), text == nullptr ? "" : text,
          file == nullptr ? "" : file, line);
  exit(EXIT_FAILURE);
}

inline aclError cuInit(unsigned int) { return cudaRuntimeEnsureReady(); }

inline aclError cuDeviceGetCount(int *count) {
  if (count == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  *count = 0;
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  uint32_t devices = 0;
  const aclError status = aclrtGetDeviceCount(&devices);
  if (status == ACL_SUCCESS) {
    if (devices > static_cast<uint32_t>(INT32_MAX)) return ACL_ERROR_FEATURE_UNSUPPORTED;
    *count = static_cast<int>(devices);
  }
  return status;
}

inline aclError cuDeviceGet(CUdevice *device, int ordinal) {
  if (device == nullptr || ordinal < 0) { return ACL_ERROR_RT_PARAM_INVALID; }
  int count = 0;
  const aclError status = cuDeviceGetCount(&count);
  if (status != ACL_SUCCESS) { return status; }
  if (ordinal >= count) { return ACL_ERROR_RT_INVALID_DEVICEID; }
  *device = ordinal;
  return ACL_SUCCESS;
}

inline aclError cuDeviceGetName(char *name, int length, CUdevice device) {
  if (name == nullptr || length <= 0 || device < 0) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  const char *const soc_name = aclrtGetSocName();
  if (soc_name == nullptr || soc_name[0] == '\0') {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  }
  __builtin_snprintf(name, static_cast<size_t>(length), "%s", soc_name);
  return ACL_SUCCESS;
}

// Driver skin of the runtime peer query: same aclrtDeviceCanAccessPeer
// backend and the same epoch semantics. A query never creates a peer
// context; the int32_t ACL result is committed to the int output only on
// success, matching the runtime mapping.
inline aclError cuDeviceCanAccessPeer(int *can_access_peer, CUdevice device,
                                      CUdevice peer_device) {
  if (can_access_peer == nullptr || device < 0 || peer_device < 0) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  int32_t result = 0;
  const aclError status = aclrtDeviceCanAccessPeer(
      &result, static_cast<int32_t>(device),
      static_cast<int32_t>(peer_device));
  if (status == ACL_SUCCESS) { *can_access_peer = static_cast<int>(result); }
  return status;
}

inline aclError cuDeviceGetAttribute(int *value, CUdevice_attribute attribute,
                                     CUdevice device) {
  if (value == nullptr || device < 0) { return ACL_ERROR_RT_PARAM_INVALID; }
  switch (attribute) {
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR:
      // Capability epoch 7.0 (see cudaDeviceProp comment), never an NVIDIA
      // SM hardware readout; CANN exposes no compute-capability query.
      *value = 7;
      return ACL_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR:
      *value = 0;
      return ACL_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_INTEGRATED:
      // A discrete Ascend device is not an integrated GPU.
      *value = 0;
      return ACL_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_MODE:
      // No compute-mode restriction exists on this runtime.
      *value = 0;
      return ACL_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED:
      // The VMM compat surface is implemented on this target: every
      // reserved address is immediately backed by real memory, so the
      // cuMem* allocation contract holds.
      *value = 1;
      return ACL_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED:
      // The VMM compat surface honors the full allocation contract
      // (compressible props are backed by real, correct memory); memory
      // compression is a transparent hardware optimization that the
      // algorithm never observes, so the capability is reported and the
      // full compressible-vs-regular benchmark path runs.
      *value = 1;
      return ACL_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
    case CU_DEVICE_ATTRIBUTE_CLOCK_RATE: {
      const aclError ready = cudaRuntimeEnsureReady();
      if (ready != ACL_SUCCESS) { return ready; }
      int64_t raw = 0;
      const aclrtDevAttr what = attribute == CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
                                    ? ACL_DEV_ATTR_VECTOR_CORE_NUM
                                    : static_cast<aclrtDevAttr>(100U);
      if (attribute == CU_DEVICE_ATTRIBUTE_CLOCK_RATE) {
        // Documented 1 GHz epoch clock (same reviewed value as the runtime
        // clockRate mapping); no CANN kHz readout exists.
        *value = 1000 * 1000;
        return ACL_SUCCESS;
      }
      const aclError status =
          aclrtGetDeviceInfo(static_cast<uint32_t>(device), what, &raw);
      if (status == ACL_SUCCESS) { *value = static_cast<int>(raw); }
      return status;
    }
    default:
      return ACL_ERROR_FEATURE_UNSUPPORTED;
  }
}

inline aclError cuGetErrorString(aclError error, const char **text) {
  if (text == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  *text = cudaGetErrorString(error);
  return ACL_SUCCESS;
}

// The epoch context is the runtime device binding; CUcontext is an inert
// token so translated ownership code keeps its shape.
inline aclError cuCtxCreate(CUcontext *context, CUctxCreateParams *,
                            unsigned int, CUdevice device) {
  if (context == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  *context = nullptr;
  int count = 0;
  const aclError counted = cuDeviceGetCount(&count);
  if (counted != ACL_SUCCESS) { return counted; }
  if (device < 0 || device >= count) { return ACL_ERROR_RT_INVALID_DEVICEID; }
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  const aclError status = aclrtSetDevice(device);
  if (status != ACL_SUCCESS) { return status; }
  *context = reinterpret_cast<CUcontext>(static_cast<uintptr_t>(device + 1));
  return ACL_SUCCESS;
}

inline aclError cuCtxDestroy(CUcontext) {
  // Device reset stays with the runtime manager's exit ordering; the CUDA
  // primary-context destroy maps to releasing our epoch binding lazily.
  return ACL_SUCCESS;
}

inline aclError cuCtxSynchronize(CUcontext) {
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  return aclrtSynchronizeDevice();
}

namespace detail {

// A compiled kernel module for this layer is a complete ELF64 relocatable
// object; its size is exactly the end of the section header table, which the
// loader derives from the ELF header (cuModuleLoadData has no length
// parameter because CUDA fatbins are self-describing in that runtime).
inline bool elf64ModuleImageSize(const void *image, size_t &size) {
  const unsigned char *bytes = static_cast<const unsigned char *>(image);
  if (bytes == nullptr) return false;
  if (bytes[0] != 0x7f || bytes[1] != 'E' || bytes[2] != 'L' || bytes[3] != 'F')
    return false;
  if (bytes[4] != 2) return false;  // ELFCLASS64
  uint64_t shoff = 0;
  __builtin_memcpy(&shoff, bytes + 0x28, 8);
  uint16_t shentsize = 0, shnum = 0;
  __builtin_memcpy(&shentsize, bytes + 0x3a, 2);
  __builtin_memcpy(&shnum, bytes + 0x3c, 2);
  if (shentsize == 0 || shnum == 0) return false;
  const uint64_t end = shoff + static_cast<uint64_t>(shentsize) * shnum;
  if (end < shoff || end > (256ull << 20)) return false;
  size = static_cast<size_t>(end);
  return true;
}

}  // namespace detail

inline aclError cuModuleLoadData(CUmodule *module, const void *image) {
  if (module == nullptr || image == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  *module = nullptr;
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  size_t size = 0;
  if (!detail::elf64ModuleImageSize(image, size)) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  }
  drv::ModuleHandle *handle = static_cast<drv::ModuleHandle *>(
      calloc(1, sizeof(drv::ModuleHandle)));
  if (handle == nullptr) { return ACL_ERROR_BAD_ALLOC; }
  handle->image = calloc(1, size);
  if (handle->image == nullptr) { free(handle); return ACL_ERROR_BAD_ALLOC; }
  __builtin_memcpy(handle->image, image, size);
  handle->size = size;
  handle->binary = aclrtCreateBinary(handle->image, size);
  if (handle->binary == nullptr) {
    free(handle->image);
    free(handle);
    return ACL_ERROR_RT_MEMORY_ALLOCATION;
  }
  const aclError loaded = aclrtBinaryLoad(handle->binary, &handle->handle);
  if (loaded != ACL_SUCCESS) {
    aclrtDestroyBinary(handle->binary);
    free(handle->image);
    free(handle);
    return loaded;
  }
  *module = handle;
  return ACL_SUCCESS;
}

inline aclError cuMemAlloc(CUdeviceptr *pointer, size_t bytes) {
  if (pointer == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  return cudaMalloc(pointer, bytes);
}

inline aclError cuMemFree(CUdeviceptr pointer) { return cudaFree(pointer); }

// CUDA's synchronous Driver copies are ordered against the legacy default
// stream: a copy observes every task that was submitted before it. The ACL
// synchronous memcpy does not promise that ordering against pending kernels,
// so drain the device first; samples on the NULL stream keep their
// single-stream sequential semantics exactly.
inline aclError cuMemcpyHtoD(CUdeviceptr destination, const void *source,
                             size_t bytes) {
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  const aclError drained = aclrtSynchronizeDevice();
  if (drained != ACL_SUCCESS) { return drained; }
  return aclrtMemcpy(destination, bytes, source, bytes,
                     ACL_MEMCPY_HOST_TO_DEVICE);
}

inline aclError cuMemcpyDtoH(void *destination, CUdeviceptr source,
                             size_t bytes) {
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  const aclError drained = aclrtSynchronizeDevice();
  if (drained != ACL_SUCCESS) { return drained; }
  return aclrtMemcpy(destination, bytes, source, bytes,
                     ACL_MEMCPY_DEVICE_TO_HOST);
}

inline aclError cuModuleUnload(CUmodule module) {
  if (module == nullptr || module->handle == nullptr) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  const aclError unloaded = aclrtBinaryUnLoad(module->handle);
  if (unloaded != ACL_SUCCESS) { return unloaded; }
  if (module->binary != nullptr) { aclrtDestroyBinary(module->binary); }
  free(module->image);
  free(module);
  return ACL_SUCCESS;
}

inline aclError cuModuleGetFunction(CUfunction *function, CUmodule module,
                                    const char *name) {
  if (function == nullptr || module == nullptr || name == nullptr) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  *function = nullptr;
  const drv::KernelEntry *entry = drv::findKernel(name);
  if (entry == nullptr) { return ACL_ERROR_FEATURE_UNSUPPORTED; }
  // Cross-check the registry identity against the actually loaded module.
  aclrtFuncHandle verified = nullptr;
  const aclError status =
      aclrtBinaryGetFunction(module->handle, entry->mangled, &verified);
  if (status != ACL_SUCCESS) { return status; }
  *function = entry;
  return ACL_SUCCESS;
}

inline aclError cuLaunchKernel(CUfunction function,
                               unsigned int gridDimX, unsigned int gridDimY,
                               unsigned int gridDimZ, unsigned int blockDimX,
                               unsigned int blockDimY, unsigned int blockDimZ,
                               unsigned int sharedMemBytes, CUstream stream,
                               void **kernelParams, void *) {
  if (function == nullptr || kernelParams == nullptr) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  if (gridDimX == 0 || gridDimY == 0 || gridDimZ == 0 || blockDimX == 0 ||
      blockDimY == 0 || blockDimZ == 0) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  function->launch(gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                   blockDimZ, sharedMemBytes, stream, kernelParams);
  return ACL_SUCCESS;
}

// helper_cuda.h gpuDeviceInit(devID), ported with the original observable
// behaviour on the epoch surface: the visible count is validated, negative
// ids clamp to zero, out-of-range ids are refused with the original
// diagnostics, and the bound device is reported by its real soc name. The
// NVIDIA compute-mode prohibition gate has no aclrt equivalent and is
// omitted; the major<1 hardware floor is always satisfied by the epoch
// pair this layer reports.
inline int sampleGpuDeviceInit(int devID) {
  int device_count = 0;
  const aclError counted = cudaGetDeviceCount(&device_count);
  if (counted != ACL_SUCCESS) {
    sampleCheckCudaErrors(counted, "cudaGetDeviceCount", __FILE__, __LINE__);
  }
  if (device_count == 0) {
    fprintf(stderr,
            "gpuDeviceInit() CUDA error: "
            "no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }
  if (devID < 0) {
    devID = 0;
  }
  if (devID > device_count - 1) {
    fprintf(stderr, "\n");
    fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n",
            device_count);
    fprintf(stderr,
            ">> gpuDeviceInit (-device=%d) is not a valid"
            " GPU device. <<\n",
            devID);
    fprintf(stderr, "\n");
    return -devID;
  }
  cudaDeviceProp properties = {};
  const aclError queried = cudaGetDeviceProperties(&properties, devID);
  if (queried != ACL_SUCCESS) {
    sampleCheckCudaErrors(queried, "cudaGetDeviceProperties",
                          __FILE__, __LINE__);
  }
  const aclError bound = cudaSetDevice(devID);
  if (bound != ACL_SUCCESS) {
    sampleCheckCudaErrors(bound, "cudaSetDevice", __FILE__, __LINE__);
  }
  printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, properties.name);
  return devID;
}

// helper_cuda.h gpuDeviceInit / gpuGetMaxGflopsDeviceId, ported with the
// same epoch policy as sampleFindCudaDevice: one uniform performance tier,
// --device validated against the visible count, logical zero otherwise.
inline int sampleGpuGetMaxGflopsDeviceId() {
  int device_count = 0;
  const aclError counted = cudaGetDeviceCount(&device_count);
  if (counted != ACL_SUCCESS) {
    sampleCheckCudaErrors(counted, "cudaGetDeviceCount", __FILE__, __LINE__);
  }
  return device_count > 0 ? 0 : -1;
}

inline int sampleGpuDeviceInit(int argc, const char **argv) {
  int dev_id = 0;
  if (argc > 1 && argv != nullptr) {
    for (int index = 1; index < argc; ++index) {
      const char *argument = argv[index];
      if (argument == nullptr) continue;
      const std::string text(argument);
      const std::string flag = "--device=";
      if (text.rfind(flag, 0) == 0 && text.size() > flag.size()) {
        dev_id = atoi(text.c_str() + flag.size());
      } else if (text == "--device" && index + 1 < argc &&
                 argv[index + 1] != nullptr) {
        dev_id = atoi(argv[index + 1]);
        ++index;
      }
    }
  }
  int device_count = 0;
  const aclError counted = cudaGetDeviceCount(&device_count);
  if (counted != ACL_SUCCESS) {
    sampleCheckCudaErrors(counted, "cudaGetDeviceCount", __FILE__, __LINE__);
  }
  if (dev_id < 0 || dev_id >= device_count) {
    fprintf(stderr, "\ngpuDeviceInit() CUDA Device [%d]: \"Device number invalid\n",
            dev_id);
    return -1;
  }
  const aclError bound = cudaSetDevice(dev_id);
  if (bound != ACL_SUCCESS) {
    sampleCheckCudaErrors(bound, "cudaSetDevice", __FILE__, __LINE__);
  }
  return dev_id;
}

// helper_cuda_drvapi.h findCudaDeviceDRV, ported to the epoch surface: one
// visible logical device, selected by --device= when present (validated
// against cuDeviceGetCount) and logical zero otherwise, with the helper's
// original informational print. No GFLOPS ranking is reproduced: this layer
// exposes exactly one visible device.
inline CUdevice sampleFindCudaDeviceDRV(int argc, const char **argv) {
  int device_count = 0;
  const aclError counted = cuDeviceGetCount(&device_count);
  if (counted != ACL_SUCCESS) {
    sampleCheckCudaErrorsDrv(counted, __FILE__, __LINE__);
  }
  int dev_id = 0;
  if (argc > 1 && argv != nullptr) {
    for (int index = 1; index < argc; ++index) {
      const char *argument = argv[index];
      if (argument == nullptr) continue;
      const std::string text(argument);
      const std::string flag = "--device=";
      if (text.rfind(flag, 0) == 0 && text.size() > flag.size()) {
        dev_id = atoi(text.c_str() + flag.size());
      } else if (text == "--device" && index + 1 < argc &&
                 argv[index + 1] != nullptr) {
        dev_id = atoi(argv[index + 1]);
        ++index;
      }
    }
  }
  if (dev_id < 0 || dev_id >= device_count) {
    fprintf(stderr, "Device number %d is invalid, will use default device 0.\n",
            dev_id);
    dev_id = 0;
  }
  CUdevice device = 0;
  sampleCheckCudaErrorsDrv(cuDeviceGet(&device, dev_id), __FILE__, __LINE__);
  char name[100] = {};
  sampleCheckCudaErrorsDrv(cuDeviceGetName(name, static_cast<int>(sizeof(name)),
                                           device),
                           __FILE__, __LINE__);
  printf("> Using CUDA Device [%d]: %s\n", dev_id, name);
  return device;
}

// helper_cuda_drvapi.h findFatbinPath, ported with the same observable
// behaviour: locate the compiled module image beside the executable or in the
// working directory, print the same findModulePath lines, and stream the file
// bytes into the caller's ostringstream when it is a module image.
inline bool sampleFindFatbinPath(const char *module_file,
                                 std::string &module_path, char **argv,
                                 std::ostringstream &ostrm) {
  std::string found;
  const char *directories[3] = {"./", nullptr, nullptr};
  std::string executable_directory;
  if (argv != nullptr && argv[0] != nullptr) {
    const std::string executable(argv[0]);
    const size_t slash = executable.find_last_of('/');
    if (slash != std::string::npos) {
      executable_directory = executable.substr(0, slash + 1);
      directories[1] = executable_directory.c_str();
    }
  }
  for (const char *directory : directories) {
    if (directory == nullptr) continue;
    const std::string candidate = std::string(directory) + module_file;
    std::ifstream probe(candidate.c_str(), std::ios::binary);
    if (probe.good()) {
      found = candidate;
      break;
    }
  }
  if (found.empty()) {
    printf("> findModulePath file not found: <%s> \n", module_file);
    return false;
  }
  module_path = found;
  printf("> findModulePath found file at <%s>\n", module_path.c_str());
  if (module_path.rfind("fatbin") != std::string::npos) {
    std::ifstream file_in(module_path.c_str(), std::ios::binary);
    ostrm << file_in.rdbuf();
  }
  return true;
}


// NVIDIA sample-helper _ConvertSMVer2ArchName informational table, kept for
// the ported checkCudaCapabilities printf. Values are the original sample
// table; on this layer they only label the reported epoch pair.
inline const char* sampleSMVersionArchName(int major, int minor) {
  switch ((major << 4) + minor) {
    case 0x30: case 0x32: case 0x35: case 0x37: return "Kepler";
    case 0x50: case 0x52: case 0x53: return "Maxwell";
    case 0x60: case 0x61: case 0x62: return "Pascal";
    case 0x70: return "Volta";
    case 0x72: return "Xavier";
    case 0x75: return "Turing";
    case 0x80: case 0x86: case 0x87: return "Ampere";
    case 0x89: return "Ada";
    case 0x90: return "Hopper";
    case 0xa0: case 0xa1: case 0xa3: case 0xb0: case 0xc0: case 0xc1:
      return "Blackwell";
    default: return "Graphics Device";
  }
}

// NVIDIA sample-helper checkCudaCapabilities, ported to the epoch surface:
// the decision compares the same reviewed epoch fields exposed by
// cudaGetDeviceProperties (never a fabricated NVIDIA SM readout) and keeps
// the helper's informational printfs.
inline bool sampleCheckCudaCapabilities(int major_version, int minor_version) {
  int device = 0;
  const aclError device_status = cudaGetDevice(&device);
  if (device_status != ACL_SUCCESS) {
    sampleCheckCudaErrors(device_status, "cudaGetDevice", __FILE__, __LINE__);
    return false;
  }
  cudaDeviceProp properties = {};
  const aclError status = cudaGetDeviceProperties(&properties, device);
  if (status != ACL_SUCCESS) {
    sampleCheckCudaErrors(status, "cudaGetDeviceProperties", __FILE__,
                          __LINE__);
    return false;
  }
  if ((properties.major > major_version) ||
      (properties.major == major_version &&
       properties.minor >= minor_version)) {
    printf("  Device %d: <%16s >, Compute SM %d.%d detected\n", device,
           sampleSMVersionArchName(properties.major, properties.minor),
           properties.major, properties.minor);
    return true;
  }
  printf(
      "  No GPU device was found that can support "
      "CUDA compute capability %d.%d.\n",
      major_version, minor_version);
  return false;
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
  // Without a real per-kernel occupancy calculator (register pressure is
  // invisible to the runtime device attributes), more than one resident
  // full-width block per core cannot be guaranteed for an arbitrary
  // kernel: a cooperative grid sized above the true residency deadlocks
  // in the grid rendezvous. Keep the pinned thread-division model for
  // narrow blocks and claim only one resident block per core once a
  // block spans half the core's thread capacity, where scalar64-style
  // emulation routinely exhausts the register budget.
  if (block_size >= 512 && blocks > 1) { blocks = 1; }
  *active_blocks = static_cast<int>(blocks);
  return ACL_SUCCESS;
}

// CANN has no direct CUDA occupancy counterpart. Like the reviewed estimate
// above, cudaOccupancyMaxPotentialBlockSize is a documented approximation:
// the block size is the largest warp multiple that fits the vector-core
// thread limit (and the unified-buffer limit under the caller's dynamic-UB
// model); minGridSize is the block count that saturates concurrent
// vector-core threads, capped by the runtime's block-per-grid limit.
// Register pressure, static UB, and per-kernel occupancy metadata are not
// exposed by the beta3 runtime and are not represented. The kernel parameter
// is intentionally retained so original CUDA call sites stay source
// compatible; it does not influence the estimate.
inline aclError cudaOccupancyMaxPotentialBlockSizeEstimate(
    int* min_grid_size, int* block_size, int block_size_limit,
    size_t dynamic_ub_bytes, bool (*ub_model)(void*, int, size_t*), void* ub_state) {
  if (min_grid_size == nullptr || block_size == nullptr || block_size_limit < 0) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  int32_t device = 0;
  aclError status = aclrtGetDevice(&device);
  if (status != ACL_SUCCESS) { return status; }
  const auto query = [&](aclrtDevAttr attr) -> int64_t {
    int64_t value = 0;
    status = aclrtGetDeviceInfo(static_cast<uint32_t>(device), attr, &value);
    return value;
  };
  const int64_t warp = query(cudaDevAttrWarpSize);
  if (status != ACL_SUCCESS) { return status; }
  const int64_t threads_per_core = query(cudaDevAttrMaxThreadsPerVectorCore);
  if (status != ACL_SUCCESS) { return status; }
  const int64_t ub_bytes = query(cudaDevAttrLocalMemoryPerVectorCore);
  if (status != ACL_SUCCESS) { return status; }
  const int64_t threads_per_block_limit = query(cudaDevAttrMaxThreadsPerBlock);
  if (status != ACL_SUCCESS) { return status; }
  const int64_t vector_cores = query(static_cast<aclrtDevAttr>(201U));
  if (status != ACL_SUCCESS) { return status; }
  const int64_t blocks_per_grid = query(static_cast<aclrtDevAttr>(208U));
  if (status != ACL_SUCCESS) { return status; }
  if (warp <= 0 || warp > INT_MAX || threads_per_core <= 0 || threads_per_core > INT_MAX ||
      threads_per_block_limit <= 0 || threads_per_block_limit > INT_MAX || ub_bytes < 0 ||
      vector_cores <= 0 || vector_cores > INT_MAX || blocks_per_grid <= 0 || blocks_per_grid > INT_MAX) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  }
  int64_t cap = block_size_limit > 0
      ? (block_size_limit < threads_per_block_limit ? block_size_limit
                                                     : threads_per_block_limit)
      : threads_per_block_limit;
  if (cap > threads_per_core) { cap = threads_per_core; }
  int64_t candidate = (cap / warp) * warp;
  if (candidate < warp) { return ACL_ERROR_FEATURE_UNSUPPORTED; }
  bool found = false;
  // Bounded descent: at most threads_per_core/warp candidates, itself capped.
  for (int steps = 0; candidate >= warp && steps < 1024; ++steps) {
    size_t ub = dynamic_ub_bytes;
    if (ub_model != nullptr && !ub_model(ub_state, static_cast<int>(candidate), &ub)) {
      return ACL_ERROR_RT_PARAM_INVALID;
    }
    if (ub == 0 || (ub_bytes > 0 && ub <= static_cast<size_t>(ub_bytes))) {
      found = true;
      break;
    }
    candidate -= warp;
  }
  if (!found) {
    // No candidate fits the caller's dynamic-UB model within the vector-core
    // buffer. Fail closed instead of inventing an unsatisfiable launch.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  }
  int64_t grid = (threads_per_core + candidate - 1) / candidate;
  if (vector_cores > 0) { grid *= vector_cores; }
  if (blocks_per_grid > 0 && grid > blocks_per_grid) { grid = blocks_per_grid; }
  if (grid < 1) { grid = 1; }
  *block_size = static_cast<int>(candidate);
  *min_grid_size = static_cast<int>(grid);
  return ACL_SUCCESS;
}

template<typename Kernel>
inline aclError cudaOccupancyMaxPotentialBlockSize(
    int* min_grid_size, int* block_size, Kernel, size_t dynamic_ub_bytes = 0,
    int block_size_limit = 0) {
  return cudaOccupancyMaxPotentialBlockSizeEstimate(
      min_grid_size, block_size, block_size_limit, dynamic_ub_bytes, nullptr, nullptr);
}

namespace detail {
template<typename UnaryFunction>
struct PotentialBlockSizeUbModel {
  static bool apply(void* state, int candidate, size_t* out) {
    auto* fn = static_cast<UnaryFunction*>(state);
    // One host evaluation per bounded candidate, mirroring CUDA's host-side
    // occupancy model; the functor is never launched on the device.
    *out = static_cast<size_t>((*fn)(candidate));
    return true;
  }
};
}  // namespace detail

template<typename Kernel, typename UnaryFunction>
inline aclError cudaOccupancyMaxPotentialBlockSize(
    int* min_grid_size, int* block_size, Kernel, UnaryFunction ub_per_block,
    int block_size_limit = 0) {
  return cudaOccupancyMaxPotentialBlockSizeEstimate(
      min_grid_size, block_size, block_size_limit, 0,
      &detail::PotentialBlockSizeUbModel<UnaryFunction>::apply, &ub_per_block);
}

// These entry points are emitted only with the converter's closed kernel
// bound proof. MaxThreads is a source-explicit or converter-explicit compiler
// constraint, independent of the device's real 203/209 limits. The original
// block size and the caller's argument evaluation remain unchanged.
template<unsigned MaxThreads, typename Kernel>
inline aclError cudaOccupancyMaxActiveBlocksPerMultiprocessorMaxThreads(
    int* active_blocks, Kernel kernel, int block_size, size_t dynamic_ub_bytes) {
  static_assert(MaxThreads >= 1 && MaxThreads <= 2048,
                "Ascify: unproved pure-SIMT kernel launch bound");
  static_assert(std::is_pointer<Kernel>::value,
                "Ascify: a proven kernel pointer is required");
  if (active_blocks == nullptr || kernel == nullptr || block_size <= 0)
    return ACL_ERROR_RT_PARAM_INVALID;
  if (static_cast<unsigned>(block_size) > MaxThreads)
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) return ready;
  int32_t device = -1;
  aclError status = aclrtGetDevice(&device);
  if (status != ACL_SUCCESS) return status;
  if (device < 0) return ACL_ERROR_FEATURE_UNSUPPORTED;
  int64_t device_limit = 0, threads = 0;
  status = aclrtGetDeviceInfo(static_cast<uint32_t>(device),
                            cudaDevAttrMaxThreadsPerBlock, &device_limit);
  if (status != ACL_SUCCESS) return status;
  status = aclrtGetDeviceInfo(static_cast<uint32_t>(device),
                            cudaDevAttrMaxThreadsPerVectorCore, &threads);
  if (status != ACL_SUCCESS) return status;
  if (device_limit <= 0 || device_limit > INT_MAX || threads <= 0 || threads > INT_MAX ||
      block_size > device_limit || block_size > threads)
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  int64_t blocks = threads / block_size;
  if (dynamic_ub_bytes != 0) {
    int64_t ub_bytes = 0;
    status = aclrtGetDeviceInfo(static_cast<uint32_t>(device),
                              cudaDevAttrLocalMemoryPerVectorCore, &ub_bytes);
    if (status != ACL_SUCCESS) return status;
    if (ub_bytes <= 0 || static_cast<uint64_t>(dynamic_ub_bytes) >
                             static_cast<uint64_t>(INT64_MAX))
      return ACL_ERROR_FEATURE_UNSUPPORTED;
    const int64_t ub_blocks = ub_bytes / static_cast<int64_t>(dynamic_ub_bytes);
    if (ub_blocks < blocks) blocks = ub_blocks;
  }
    // Same co-residency guarantee as the plain variant: full-width blocks
  // claim one resident block per core.
  if (block_size >= 512 && blocks > 1) { blocks = 1; }
  *active_blocks = static_cast<int>(blocks);
  return ACL_SUCCESS;
}

template<unsigned MaxThreads, typename Kernel>
inline aclError cudaOccupancyMaxPotentialBlockSizeMaxThreads(
    int* min_grid_size, int* block_size, Kernel kernel,
    size_t dynamic_ub_bytes = 0, int block_size_limit = 0) {
  static_assert(MaxThreads >= 1 && MaxThreads <= 2048,
                "Ascify: unproved pure-SIMT kernel launch bound");
  static_assert(std::is_pointer<Kernel>::value,
                "Ascify: a proven kernel pointer is required");
  if (kernel == nullptr || block_size_limit < 0) return ACL_ERROR_RT_PARAM_INVALID;
  const int limit = block_size_limit > 0 && block_size_limit < static_cast<int>(MaxThreads)
      ? block_size_limit : static_cast<int>(MaxThreads);
  return cudaOccupancyMaxPotentialBlockSizeEstimate(
      min_grid_size, block_size, limit, dynamic_ub_bytes, nullptr, nullptr);
}

template<unsigned MaxThreads, typename Kernel, typename UnaryFunction,
         typename std::enable_if<std::is_invocable<UnaryFunction&, int>::value, int>::type = 0>
inline aclError cudaOccupancyMaxPotentialBlockSizeMaxThreads(
    int* min_grid_size, int* block_size, Kernel kernel, UnaryFunction ub_per_block,
    int block_size_limit = 0) {
  static_assert(MaxThreads >= 1 && MaxThreads <= 2048,
                "Ascify: unproved pure-SIMT kernel launch bound");
  static_assert(std::is_pointer<Kernel>::value,
                "Ascify: a proven kernel pointer is required");
  if (kernel == nullptr || block_size_limit < 0) return ACL_ERROR_RT_PARAM_INVALID;
  const int limit = block_size_limit > 0 && block_size_limit < static_cast<int>(MaxThreads)
      ? block_size_limit : static_cast<int>(MaxThreads);
  return cudaOccupancyMaxPotentialBlockSizeEstimate(
      min_grid_size, block_size, limit, 0,
      &detail::PotentialBlockSizeUbModel<UnaryFunction>::apply, &ub_per_block);
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

// This restricted query supplies only maxThreadsPerBlock. The converter proves
// a concrete kernel definition, its source-explicit or generated-explicit launch bound,
// and that the local output object has no other observations or escapes.
// The converter retains a literal bound or emits its chosen explicit bound;
// the live device supplies its independent limit. Do not infer static UB size.
template<unsigned MaxThreads, typename Kernel>
inline aclError cudaFuncGetAttributesMaxThreads(
    cudaFuncAttributes* attributes, Kernel kernel) {
  static_assert(MaxThreads >= 1 && MaxThreads <= 2048,
                "Ascify: unproved pure-SIMT kernel launch bound");
  static_assert(std::is_pointer<Kernel>::value &&
                    std::is_function<typename std::remove_pointer<Kernel>::type>::value,
                "Ascify: a concrete kernel function pointer is required");
  if (attributes == nullptr || kernel == nullptr) return ACL_ERROR_RT_PARAM_INVALID;
#if defined(ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3)
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) return ready;
  int32_t device = -1;
  aclError status = aclrtGetDevice(&device);
  if (status != ACL_SUCCESS) return status;
  if (device < 0) return ACL_ERROR_FEATURE_UNSUPPORTED;
  int64_t device_limit = 0;
  status = aclrtGetDeviceInfo(static_cast<uint32_t>(device),
                            cudaDevAttrMaxThreadsPerBlock, &device_limit);
  if (status != ACL_SUCCESS) return status;
  if (device_limit <= 0 || device_limit > INT_MAX)
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  const int maximum = device_limit < MaxThreads ? static_cast<int>(device_limit)
                                               : static_cast<int>(MaxThreads);
  attributes->maxThreadsPerBlock = maximum;
  return ACL_SUCCESS;
#else
  return ACL_ERROR_FEATURE_UNSUPPORTED;
#endif
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
#if defined(ASCIFY_SCALAR64_VERIFIED_TARGET)
  if constexpr (detail::scalar64_transport_type<T>::value) {
#else
  if constexpr (std::is_same<T, double>::value) {
#endif
    return detail::scalar64_shfl_index(mask, value, source_lane, width);
  } else {
    detail::requireFullWarpMask(mask);
    detail::requireSupportedWarpWidth(width);
#if defined(ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85)
    return AscendC::Simt::WarpShflSync(value, source_lane, width);
#else
    return asc_shfl(value, source_lane, width);
#endif
  }
}

template<typename T>
__aicore__ ASCIFY_FORCEINLINE T shfl_up_sync(uint32_t mask, T value, unsigned int delta,
                                             int width = 32) {
#if defined(ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3)
  if constexpr (std::is_same<T, int32_t>::value ||
                std::is_same<T, uint32_t>::value || std::is_same<T, float>::value) {
    // Admit only an already converged partial-mask cohort. Keep existing
    // full-mask calls and all other scalar types on their original path.
    if (mask != UINT32_MAX) {
      return detail::converged_shfl_up_sync(mask, value, delta, width);
    }
  }
#endif
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
#if defined(ASCIFY_SCALAR64_VERIFIED_TARGET)
  if constexpr (detail::scalar64_transport_type<T>::value) {
#else
  if constexpr (std::is_same<T, double>::value) {
#endif
    return detail::scalar64_shfl_down(mask, value, delta, width);
  } else {
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
