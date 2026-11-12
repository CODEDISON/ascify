// Ascify CUDA Dynamic Parallelism emulation: a device-side <<<>>> launch
// becomes a work-queue entry (atomically claimed slot in a device-global
// ring); the host replays every entry at the next drained device
// synchronization through the typed replay glue the converter emits beside
// the kernel declarations. The recursive data flow (partition on device,
// sub-range sorts launched by device code) is preserved exactly; the
// replay is sequential, which these samples's disjoint-range recursions
// tolerate.
#ifndef ASCIFY_CDP_COMPAT_HPP
#define ASCIFY_CDP_COMPAT_HPP

#include <type_traits>
#include <vector>

#include "ascify_cuda_compat.hpp"

namespace ascify {
namespace cdp {

inline constexpr unsigned kMaxTasks = 65536;
inline constexpr unsigned kMaxArgs = 12;

struct TaskHeader {
  unsigned id;
  unsigned gx, gy, gz;
  unsigned bx, by, bz;
  unsigned smem;
  unsigned long long stream;
  unsigned nargs;
  unsigned long long args[kMaxArgs];
};

// Device-side queue state. The host reaches it through the address kernel
// below (same contract as the symbol_compat resolvers).
__aicore__ TaskHeader g_tasks[kMaxTasks];
__aicore__ unsigned g_head;

// Host hooks installed by the converter-emitted glue.
using ReplayFn = void (*)(const TaskHeader &);
inline ReplayFn &replayFn() {
  static ReplayFn value = nullptr;
  return value;
}
ASCIFY_GLOBAL void ascify_cdp_resolve(unsigned long long *head,
                                      unsigned long long *tasks);

inline int drainOnce() {
  static unsigned long long headAddress = 0, tasksAddress = 0;
  if (headAddress == 0) {
    const aclError ready = cudaRuntimeEnsureReady();
    if (ready != ACL_SUCCESS) return ready;
    void *probe = nullptr;
    aclError status = aclrtMalloc(&probe, 16, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (status != ACL_SUCCESS) return status;
    ascify_cdp_resolve<<<1, 1>>>(
        static_cast<unsigned long long *>(probe),
        static_cast<unsigned long long *>(probe) + 1);
    status = aclrtSynchronizeStream(nullptr);
    unsigned long long addresses[2] = {0, 0};
    if (status == ACL_SUCCESS)
      status = aclrtMemcpy(addresses, 16, probe, 16, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtFree(probe);
    if (status != ACL_SUCCESS || addresses[0] == 0) return ACL_ERROR_FAILURE;
    headAddress = addresses[0];
    tasksAddress = addresses[1];
  }
  unsigned processed = 0;
  std::vector<TaskHeader> batch;
  for (;;) {
    unsigned head = 0;
    aclError status = aclrtMemcpy(&head, sizeof(head),
        reinterpret_cast<void *>(static_cast<uintptr_t>(headAddress)),
        sizeof(head), ACL_MEMCPY_DEVICE_TO_HOST);
    if (status != ACL_SUCCESS) return status;
    if (head <= processed) break;
    if (head > kMaxTasks) return ACL_ERROR_RT_PARAM_INVALID;
    batch.resize(head - processed);
    status = aclrtMemcpy(batch.data(), batch.size() * sizeof(TaskHeader),
        reinterpret_cast<void *>(static_cast<uintptr_t>(tasksAddress) +
            static_cast<size_t>(processed) * sizeof(TaskHeader)),
        batch.size() * sizeof(TaskHeader), ACL_MEMCPY_DEVICE_TO_HOST);
    if (status != ACL_SUCCESS) return status;
    processed = head;
    const ReplayFn replay = replayFn();
    for (const TaskHeader &task : batch)
      if (replay != nullptr) replay(task);
    // Replayed kernels may enqueue more work; settle the device before
    // reading the head again.
    status = aclrtSynchronizeStream(nullptr);
    if (status != ACL_SUCCESS) return status;
  }
  if (processed != 0) {
    const unsigned zero = 0;
    const aclError status = aclrtMemcpy(
        reinterpret_cast<void *>(static_cast<uintptr_t>(headAddress)),
        sizeof(zero), &zero, sizeof(zero), ACL_MEMCPY_HOST_TO_DEVICE);
    if (status != ACL_SUCCESS) return status;
  }
  return ACL_SUCCESS;
}

}  // namespace cdp
}  // namespace ascify

// Device-side enqueue: claim a slot, store the launch description and the
// argument words. Argument packing is by-value for scalars/pointers, the
// same protocol the host replay glue unpacks.
namespace ascify {
namespace cdp {
namespace detail_cdp {

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline unsigned long long argBits(T value) {
  if constexpr (std::is_same<T, float>::value) {
    return __builtin_bit_cast(unsigned, value);
  } else if constexpr (std::is_same<T, double>::value) {
    return __builtin_bit_cast(unsigned long long, value);
  } else if constexpr (std::is_pointer<T>::value) {
    return static_cast<unsigned long long>(
        reinterpret_cast<uintptr_t>(value));
  } else {
    return static_cast<unsigned long long>(value);
  }
}

}  // namespace detail_cdp

template <unsigned ID, typename... Args>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline void enqueue(
    unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
    unsigned bz, unsigned smem, const void *stream, Args... args) {
  const unsigned slot = ::ascify::atomic_add_generic(&g_head, 1u);
  if (slot >= kMaxTasks) {
    ::ascify::detail::reject_device_contract();
    return;
  }
  TaskHeader &task = g_tasks[slot];
  task.id = ID;
  task.gx = gx; task.gy = gy; task.gz = gz;
  task.bx = bx; task.by = by; task.bz = bz;
  task.smem = smem;
  task.stream = reinterpret_cast<unsigned long long>(stream);
  task.nargs = sizeof...(Args);
  const unsigned long long words[kMaxArgs] = {detail_cdp::argBits(args)...};
  for (unsigned index = 0; index < task.nargs; ++index)
    task.args[index] = index < kMaxArgs ? words[index] : 0u;
}

// Device-context stream stubs: CDP recursion uses fresh non-blocking
// streams per level; the replay is stream-agnostic, so create/destroy
// only maintain benign tokens.
__SIMT_DEVICE_FUNCTIONS_DECL__ inline aclError deviceStreamCreate(
    void **stream) {
  if (stream == nullptr) return ACL_ERROR_RT_PARAM_INVALID;
  *stream = reinterpret_cast<void *>(static_cast<uintptr_t>(0xC0FFEE11ULL));
  return ACL_SUCCESS;
}
__SIMT_DEVICE_FUNCTIONS_DECL__ inline aclError deviceStreamDestroy(void *) {
  return ACL_SUCCESS;
}

}  // namespace cdp
}  // namespace ascify

// Address kernel: writes the queue's device addresses for the host drain.
ASCIFY_GLOBAL void ascify_cdp_resolve(unsigned long long *head,
                                      unsigned long long *tasks) {
  *head = static_cast<unsigned long long>(
      reinterpret_cast<uintptr_t>(&::ascify::cdp::g_head));
  *tasks = static_cast<unsigned long long>(
      reinterpret_cast<uintptr_t>(&::ascify::cdp::g_tasks[0]));
}

#endif  // ASCIFY_CDP_COMPAT_HPP
