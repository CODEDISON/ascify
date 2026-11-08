#ifndef ASCIFY_ASCIFY_CUDA_COMPAT_HPP
#define ASCIFY_ASCIFY_CUDA_COMPAT_HPP

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

inline aclError cudaDeviceSynchronize() {
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  const aclError status = aclrtSynchronizeDevice();
  if (status != ACL_SUCCESS) return status;
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
  return ACL_SUCCESS;
#else
  (void)priority;
  return ACL_ERROR_FEATURE_UNSUPPORTED;
#endif
}

inline aclError cudaStreamDestroy(aclrtStream stream) {
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  return aclrtDestroyStream(stream);
}

inline aclError cudaStreamSynchronize(aclrtStream stream) {
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  const aclError status = aclrtSynchronizeStream(stream);
  if (status != ACL_SUCCESS) return status;
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

inline aclError cudaStreamWaitEvent(aclrtStream stream, aclrtEvent event,
                                    unsigned int flags = 0) {
  if (flags == cudaEventWaitExternal) { return ACL_ERROR_FEATURE_UNSUPPORTED; }
  if (flags != cudaEventWaitDefault || event == nullptr) {
    return ACL_ERROR_RT_PARAM_INVALID;
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
  return aclrtStreamWaitEvent(stream, event);
#else
  return ACL_ERROR_FEATURE_UNSUPPORTED;
#endif
}

using cudaHostFn_t = void (*)(void*);

inline aclError cudaLaunchHostFunc(aclrtStream stream, cudaHostFn_t function,
                                  void* user_data) {
  if (function == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
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
  const aclError status = aclrtSynchronizeEvent(event);
  if (status != ACL_SUCCESS) return status;
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
struct UnsupportedCudaSmAttribute {};
inline constexpr UnsupportedCudaSmAttribute cudaDevAttrComputeCapabilityMajor{};
inline constexpr UnsupportedCudaSmAttribute cudaDevAttrComputeCapabilityMinor{};

inline aclError cudaDeviceGetAttribute(int* value, UnsupportedCudaSmAttribute,
                                      int device) {
  if (value == nullptr || device < 0) { return ACL_ERROR_RT_PARAM_INVALID; }
  // CUDA SM versions have no NPU equivalent. Preserve the output and return
  // an explicit failure; these selector tags carry no fabricated SM number.
  return ACL_ERROR_FEATURE_UNSUPPORTED;
}

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
  if (device != 0) { return ACL_ERROR_FEATURE_UNSUPPORTED; }
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
  *properties = result;
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
