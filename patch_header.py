#!/usr/bin/env python3
"""Apply the CUDA Graph host-emulation runtime to ascify_cuda_compat.hpp.

Text-anchored edits with exact-match assertions. Run from
source_vector_next root. Idempotency is not attempted: aborts if an
anchor is missing or duplicated unexpectedly.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
HEADER = ROOT / "include/ascify/ascify_cuda_compat.hpp"
DETAIL = pathlib.Path("/tmp/graph_detail_block.hpp").read_text()
PUBLIC = pathlib.Path("/tmp/graph_public_block.hpp").read_text()

text = HEADER.read_text()

def patch(old, new, count=1):
    global text
    found = text.count(old)
    assert found == count, f"anchor found {found}x (want {count}):\n---\n{old[:300]}\n---"
    text = text.replace(old, new)
    print(f"  ok: {old.splitlines()[0][:72]}")

# ---- P1: detail block before the launch ABI shim declarations ----------
SHIM_ANCHOR = '''// The two runtime launch ABI symbols that carry a packed kernel-argument'''
assert text.count(SHIM_ANCHOR) == 1
patch(SHIM_ANCHOR, DETAIL.rstrip() + "\n\n" + SHIM_ANCHOR)

# ---- P2: legacy shim body ----------------------------------------------
patch('''                                 void *stm) {
  detail::stagePeerKernelArguments(args, args_size);''',
      '''                                 void *stm) {
  if (detail::graphCaptureDivertLaunchLegacy(stub_func, num_blocks, args,
                                             args_size, sm_desc, stm)) {
    return 0;  // recorded into the capturing graph
  }
  detail::managedStageKernelArguments(args, args_size, stm);
  detail::stagePeerKernelArguments(args, args_size);''')

patch('''  if (real_launch == nullptr) return 1;  // RT_ERROR_INVALID_VALUE
  return real_launch(stub_func, num_blocks, args, args_size, sm_desc, stm);
}''',
      '''  if (real_launch == nullptr) return 1;  // RT_ERROR_INVALID_VALUE
  const int32_t launched =
      real_launch(stub_func, num_blocks, args, args_size, sm_desc, stm);
  detail::managedPublishAfterLaunch(stm);
  return launched;
}''')

# ---- P3: V2 shim body ----------------------------------------------------
patch('''  if (args_info != nullptr) {
    detail::AscifyRtArgsEx &packed =
        *static_cast<detail::AscifyRtArgsEx *>(args_info);
    detail::stagePeerKernelArguments(packed.args, packed.argsSize);
  }''',
      '''  if (detail::graphCaptureDivertLaunchV2(stub_func, block_dim, args_info,
                                         sm_desc, stm, flags, cfg_info)) {
    return 0;  // recorded into the capturing graph
  }
  if (args_info != nullptr) {
    detail::AscifyRtArgsEx &packed =
        *static_cast<detail::AscifyRtArgsEx *>(args_info);
    detail::managedStageKernelArguments(packed.args, packed.argsSize, stm);
    detail::stagePeerKernelArguments(packed.args, packed.argsSize);
  }''')

patch('''  return real_launch(stub_func, block_dim, args_info, sm_desc, stm, flags,
                     cfg_info);
}''',
      '''  const int32_t launched = real_launch(stub_func, block_dim, args_info,
                                       sm_desc, stm, flags, cfg_info);
  detail::managedPublishAfterLaunch(stm);
  return launched;
}''')

# ---- P4: cudaMemcpyAsync capture hook ------------------------------------
patch('''  return aclrtMemcpyAsync(destination, bytes, source, bytes, kind, stream);''',
      '''  if (detail::graphCaptureActive()) {
    const aclError captured = detail::graphCaptureDivertMemcpy(
        destination, source, bytes, kind, stream);
    if (captured != ACL_ERROR_FEATURE_UNSUPPORTED) { return captured; }
  }
  return aclrtMemcpyAsync(destination, bytes, source, bytes, kind, stream);''')

# ---- P5: cudaMemsetAsync capture hook ------------------------------------
patch('''  return aclrtMemsetAsync(destination, bytes, value, bytes, stream);''',
      '''  if (detail::graphCaptureActive()) {
    const aclError captured = detail::graphCaptureDivertMemset(
        destination, value, bytes, stream);
    if (captured != ACL_ERROR_FEATURE_UNSUPPORTED) { return captured; }
  }
  return aclrtMemsetAsync(destination, bytes, value, bytes, stream);''')

# ---- P6: cudaMallocAsync capture hook ------------------------------------
patch('''  if (ptr == nullptr) return ACL_ERROR_RT_PARAM_INVALID;
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  detail::RuntimeLockGuard allocationGuard(detail::runtimeAllocationRegistry().lock);
  const aclError status = aclrtMalloc(reinterpret_cast<void **>(ptr), size,
                                      ACL_MEM_MALLOC_HUGE_FIRST);''',
      '''  if (ptr == nullptr) return ACL_ERROR_RT_PARAM_INVALID;
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
                                      ACL_MEM_MALLOC_HUGE_FIRST);''')

# ---- P7: cudaFreeAsync ownership hooks -----------------------------------
patch('''  if (ptr == nullptr) return ACL_ERROR_RT_PARAM_INVALID;
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  detail::RuntimeLockGuard allocationGuard(detail::runtimeAllocationRegistry().lock);
  auto *allocation = detail::runtimeAllocationLocked(ptr);''',
      '''  if (ptr == nullptr) return ACL_ERROR_RT_PARAM_INVALID;
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  if (detail::managedFree(ptr)) { return ACL_SUCCESS; }
  // A graph-owned allocation keeps its physical block (address-stable
  // relaunch contract); the free is recorded as logical liveness only.
  if (detail::graphMarkOwnedPointerDead(ptr)) { return ACL_SUCCESS; }
  detail::RuntimeLockGuard allocationGuard(detail::runtimeAllocationRegistry().lock);
  auto *allocation = detail::runtimeAllocationLocked(ptr);''')

# ---- P8: cudaFree ownership hooks ----------------------------------------
patch('''  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  detail::RuntimeLockGuard allocationGuard(detail::runtimeAllocationRegistry().lock);
  auto *allocation = detail::runtimeAllocationLocked(device_pointer);''',
      '''  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  if (detail::managedFree(device_pointer)) { return ACL_SUCCESS; }
  if (detail::graphMarkOwnedPointerDead(device_pointer)) { return ACL_SUCCESS; }
  detail::RuntimeLockGuard allocationGuard(detail::runtimeAllocationRegistry().lock);
  auto *allocation = detail::runtimeAllocationLocked(device_pointer);''')

# ---- P9: cudaStreamWaitEvent capture hook --------------------------------
patch('''  if (flags != cudaEventWaitDefault || event == nullptr) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  if (stream == nullptr || stream == cudaStreamLegacy ||''',
      '''  if (flags != cudaEventWaitDefault || event == nullptr) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  if (detail::graphCaptureActive() &&
      detail::graphCaptureWaitEvent(stream, event)) {
    return ACL_SUCCESS;  // recorded as a capture dependency edge
  }
  if (stream == nullptr || stream == cudaStreamLegacy ||''')

# ---- P10: cudaLaunchHostFunc capture hook --------------------------------
patch('''  if (function == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  // ACL has no CUDA legacy default-stream cross-stream synchronization.''',
      '''  if (function == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  if (detail::graphCaptureActive() &&
      detail::graphCaptureDivertHostFunc(stream, function, user_data)) {
    return ACL_SUCCESS;  // recorded as a graph host node
  }
  // ACL has no CUDA legacy default-stream cross-stream synchronization.''')

# ---- P11: cudaEventRecord capture hook -----------------------------------
patch('''inline aclError cudaEventRecord(
    aclrtEvent event, aclrtStream stream = nullptr) {
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  if (stream == nullptr) {''',
      '''inline aclError cudaEventRecord(
    aclrtEvent event, aclrtStream stream = nullptr) {
  const aclError ready = cudaRuntimeEnsureReady();
  if (ready != ACL_SUCCESS) { return ready; }
  if (detail::graphCaptureActive() && stream != nullptr &&
      detail::graphCaptureRecordEvent(event, stream)) {
    return ACL_SUCCESS;  // recorded as the stream's capture tail
  }
  if (stream == nullptr) {''')

# ---- P12: managed refresh at synchronization points ----------------------
patch('''  const aclError status = aclrtSynchronizeStream(stream);
  if (status != ACL_SUCCESS) return status;
  const auto *hooks = detail::runtimeTextureHooks.load(std::memory_order_acquire);
  return hooks ? hooks->checkPending() : ACL_SUCCESS;
}

inline aclError cudaStreamQuery(aclrtStream stream) {''',
      '''  const aclError status = aclrtSynchronizeStream(stream);
  if (status != ACL_SUCCESS) return status;
  detail::managedRefreshAfterSync();
  const auto *hooks = detail::runtimeTextureHooks.load(std::memory_order_acquire);
  return hooks ? hooks->checkPending() : ACL_SUCCESS;
}

inline aclError cudaStreamQuery(aclrtStream stream) {''')

patch('''  const aclError status = aclrtSynchronizeDevice();
  if (status != ACL_SUCCESS) return status;
  const auto *hooks = detail::runtimeTextureHooks.load(std::memory_order_acquire);
  return hooks ? hooks->checkPending() : ACL_SUCCESS;
}''',
      '''  const aclError status = aclrtSynchronizeDevice();
  if (status != ACL_SUCCESS) return status;
  detail::managedRefreshAfterSync();
  const auto *hooks = detail::runtimeTextureHooks.load(std::memory_order_acquire);
  return hooks ? hooks->checkPending() : ACL_SUCCESS;
}''')

patch('''  const aclError status = aclrtSynchronizeEvent(event);
  if (status != ACL_SUCCESS) return status;
  const auto *hooks = detail::runtimeTextureHooks.load(std::memory_order_acquire);
  return hooks ? hooks->checkPending() : ACL_SUCCESS;
}''',
      '''  const aclError status = aclrtSynchronizeEvent(event);
  if (status != ACL_SUCCESS) return status;
  detail::managedRefreshAfterSync();
  const auto *hooks = detail::runtimeTextureHooks.load(std::memory_order_acquire);
  return hooks ? hooks->checkPending() : ACL_SUCCESS;
}''')

# ---- P13: public graph API block -----------------------------------------
patch('''inline aclError cudaGetLastError() {''',
      PUBLIC.rstrip() + "\n\ninline aclError cudaGetLastError() {", count=1)

HEADER.write_text(text)
print("HEADER PATCHED")
print("new size:", len(text.encode()))
