/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "CUDA2DPP.h"

using SEC = runtime::CUDA_RUNTIME_API_SECTIONS;

// Map of all CUDA Runtime API functions
const std::map<llvm::StringRef, dppCounter> CUDA_RUNTIME_FUNCTION_MAP = [] {
  std::map<llvm::StringRef,  dppCounter> m;
  m["cudaMemcpyFromSymbol"] = {"ascify::cudaMemcpyFromSymbol", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaMemcpyToSymbol"] = {"ascify::cudaMemcpyToSymbol", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaGetDeviceProperties"] = {"ascify::cudaGetDeviceProperties", CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  // The admitted CUDA SDK aliases the public spelling to this exact symbol.
  // Do not normalize arbitrary version suffixes.
  m["cudaGetDeviceProperties_v2"] = {"ascify::cudaGetDeviceProperties", CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  // Memory management
  // CUDA and ACL memory APIs differ in allocation policy, destination bounds,
  // and async argument order.  Preserve CUDA call shapes and adapt them in the
  // compatibility layer rather than performing unsound name-only lowering.
  m["cudaMalloc"]                                              = {"ascify::cudaMalloc",                                     CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaFree"]                                                = {"ascify::cudaFree",                                       CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaMemcpy"]                                              = {"ascify::cudaMemcpy",                                     CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaMemcpyAsync"]                                         = {"ascify::cudaMemcpyAsync",                                CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaMemset"]                                              = {"ascify::cudaMemset",                                     CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaMemsetAsync"]                                         = {"ascify::cudaMemsetAsync",                                CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaMallocHost"]                                         = {"ascify::cudaMallocHost",                                 CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaProfilerStart"] = {"ascify::cudaProfilerStart", CONV_PROFILER, API_RUNTIME, 0};
  m["cudaProfilerStop"] = {"ascify::cudaProfilerStop", CONV_PROFILER, API_RUNTIME, 0};
  m["cudaLaunchHostFunc"] = {"ascify::cudaLaunchHostFunc", CONV_EXECUTION, API_RUNTIME, 0};
  m["cudaFreeHost"]                                            = {"ascify::cudaFreeHost",                                   CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // Mapped/pinned host memory (CUDA zero-copy) and device flags; see
  // ascify_cuda_compat.hpp for the native ACL mapping contracts.
  m["cudaHostAlloc"] = {"ascify::cudaHostAlloc", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaHostRegister"] = {"ascify::cudaHostRegister", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaHostUnregister"] = {"ascify::cudaHostUnregister", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaHostGetDevicePointer"] = {"ascify::cudaHostGetDevicePointer", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaSetDeviceFlags"] = {"ascify::cudaSetDeviceFlags", CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  // CUDA Driver API line (vectorAddDrv/simpleDrvRuntime): real runtime module
  // loading plus typed <<<>>> launch thunks; see ascify_cuda_compat.hpp.
  m["cuInit"] = {"ascify::cuInit", CONV_DEVICE, API_DRIVER, 0};
  m["cuDeviceGetCount"] = {"ascify::cuDeviceGetCount", CONV_DEVICE, API_DRIVER, 0};
  m["cuDeviceGet"] = {"ascify::cuDeviceGet", CONV_DEVICE, API_DRIVER, 0};
  m["cuDeviceGetName"] = {"ascify::cuDeviceGetName", CONV_DEVICE, API_DRIVER, 0};
  m["cuDeviceGetAttribute"] = {"ascify::cuDeviceGetAttribute", CONV_DEVICE, API_DRIVER, 0};
  m["cuGetErrorString"] = {"ascify::cuGetErrorString", CONV_DEVICE, API_DRIVER, 0};
  m["cuCtxCreate"] = {"ascify::cuCtxCreate", CONV_DEVICE, API_DRIVER, 0};
  m["cuCtxDestroy"] = {"ascify::cuCtxDestroy", CONV_DEVICE, API_DRIVER, 0};
  m["cuCtxSynchronize"] = {"ascify::cuCtxSynchronize", CONV_DEVICE, API_DRIVER, 0};
  m["cuModuleLoadData"] = {"ascify::cuModuleLoadData", CONV_MEMORY, API_DRIVER, 0};
  m["cuModuleUnload"] = {"ascify::cuModuleUnload", CONV_MEMORY, API_DRIVER, 0};
  m["cuModuleGetFunction"] = {"ascify::cuModuleGetFunction", CONV_EXECUTION, API_DRIVER, 0};
  m["cuMemAlloc"] = {"ascify::cuMemAlloc", CONV_MEMORY, API_DRIVER, 0};
  m["cuMemFree"] = {"ascify::cuMemFree", CONV_MEMORY, API_DRIVER, 0};
  m["cuMemcpyHtoD"] = {"ascify::cuMemcpyHtoD", CONV_MEMORY, API_DRIVER, 0};
  m["cuMemcpyDtoH"] = {"ascify::cuMemcpyDtoH", CONV_MEMORY, API_DRIVER, 0};
  m["cuLaunchKernel"] = {"ascify::cuLaunchKernel", CONV_EXECUTION, API_DRIVER, 0};
  // CUDA SDK version macros (cuda.h) rename several Driver entry points at
  // preprocessing time; the AST therefore sees these _v2/_v4 spellings. Map
  // each to the same reviewed implementation.
  m["cuCtxCreate_v4"] = {"ascify::cuCtxCreate", CONV_DEVICE, API_DRIVER, 0};
  m["cuMemAlloc_v2"] = {"ascify::cuMemAlloc", CONV_MEMORY, API_DRIVER, 0};
  m["cuMemFree_v2"] = {"ascify::cuMemFree", CONV_MEMORY, API_DRIVER, 0};
  m["cuMemcpyHtoD_v2"] = {"ascify::cuMemcpyHtoD", CONV_MEMORY, API_DRIVER, 0};
  m["cuMemcpyDtoH_v2"] = {"ascify::cuMemcpyDtoH", CONV_MEMORY, API_DRIVER, 0};
  m["cuCtxDestroy_v2"] = {"ascify::cuCtxDestroy", CONV_DEVICE, API_DRIVER, 0};
  m["cuMemcpyHtoD_v2_ptds"] = {"ascify::cuMemcpyHtoD", CONV_MEMORY, API_DRIVER, 0};
  m["cuMemcpyDtoH_v2_ptds"] = {"ascify::cuMemcpyDtoH", CONV_MEMORY, API_DRIVER, 0};
  // Bounded texture/storage surface, guarded by CudaTextureBoundary.
  m["cudaMallocPitch"] = {"ascify::cudaMallocPitch", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaMallocArray"] = {"ascify::cudaMallocArray", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaMalloc3DArray"] = {"ascify::cudaMalloc3DArray", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaFreeArray"] = {"ascify::cudaFreeArray", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaMemcpy2D"] = {"ascify::cudaMemcpy2D", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaMemset2D"] = {"ascify::cudaMemset2D", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaMemcpyToArray"] = {"ascify::cudaMemcpyToArray", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaMemcpy2DToArray"] = {"ascify::cudaMemcpy2DToArray", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaMemcpy3D"] = {"ascify::cudaMemcpy3D", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaCreateTextureObject"] = {"ascify::cudaCreateTextureObject", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaDestroyTextureObject"] = {"ascify::cudaDestroyTextureObject", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaCreateChannelDesc"] = {"ascify::cudaCreateChannelDesc", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["make_cudaExtent"] = {"ascify::make_cudaExtent", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["make_cudaPos"] = {"ascify::make_cudaPos", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["make_cudaPitchedPtr"] = {"ascify::make_cudaPitchedPtr", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["tex2D"] = {"ascify::tex2D", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["tex2DLayered"] = {"ascify::tex2DLayered", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["texCubemap"] = {"ascify::texCubemap", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // Device management
  // CUDA owns lazy runtime/context creation. Route all lifecycle-sensitive
  // calls through the compatibility manager instead of assuming an ACL
  // context already exists or directly changing a mismatched call shape.
  m["cudaGetDevice"]                                           = {"ascify::cudaGetDevice",                                  CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaSetDevice"]                                           = {"ascify::cudaSetDevice",                                  CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaGetDeviceCount"]                                      = {"ascify::cudaGetDeviceCount",                             CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaDeviceCanAccessPeer"] = {"ascify::cudaDeviceCanAccessPeer", CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaDeviceEnablePeerAccess"] = {"ascify::cudaDeviceEnablePeerAccess", CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaDeviceDisablePeerAccess"] = {"ascify::cudaDeviceDisablePeerAccess", CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  // CANN uses a different argument order and writes int64_t. Keep CUDA's
  // signature at the call site and adapt it in the compatibility shim.
  m["cudaDeviceGetAttribute"]                                  = {"ascify::cudaDeviceGetAttribute",                         CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaDeviceSynchronize"]                                   = {"ascify::cudaDeviceSynchronize",                          CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaDeviceReset"]                                         = {"ascify::cudaDeviceReset",                                CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  // beta3 has no per-kernel attribute API. The compatibility layer exposes the
  // subset needed by OneFlow's dynamic-UB selection code.
  m["cudaFuncGetAttributes"]                                   = {"ascify::cudaFuncGetAttributes",                          CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaFuncSetAttribute"]                                    = {"ascify::cudaFuncSetAttribute",                           CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaDeviceGetStreamPriorityRange"] = {"ascify::cudaDeviceGetStreamPriorityRange", CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaStreamCreateWithPriority"] = {"ascify::cudaStreamCreateWithPriority", CONV_STREAM, API_RUNTIME, SEC::STREAM};
  // Stream management
  m["cudaStreamCreate"]                                        = {"ascify::cudaStreamCreate",                               CONV_STREAM, API_RUNTIME, SEC::STREAM};
  m["cudaStreamCreateWithFlags"]                               = {"ascify::cudaStreamCreateWithFlags",                      CONV_STREAM, API_RUNTIME, SEC::STREAM};
  m["cudaStreamDestroy"]                                       = {"ascify::cudaStreamDestroy",                              CONV_STREAM, API_RUNTIME, SEC::STREAM};
  m["cudaStreamSynchronize"]                                   = {"ascify::cudaStreamSynchronize",                          CONV_STREAM, API_RUNTIME, SEC::STREAM};
  m["cudaStreamQuery"] = {"ascify::cudaStreamQuery", CONV_STREAM, API_RUNTIME, SEC::STREAM};
  m["cudaStreamWaitEvent"] = {"ascify::cudaStreamWaitEvent", CONV_STREAM, API_RUNTIME, SEC::STREAM};
  // Event management
  m["cudaEventCreate"]                                         = {"ascify::cudaEventCreate",                                CONV_EVENT, API_RUNTIME, SEC::EVENT};
  m["cudaEventDestroy"]                                        = {"ascify::cudaEventDestroy",                               CONV_EVENT, API_RUNTIME, SEC::EVENT};
  m["cudaEventRecord"]                                         = {"ascify::cudaEventRecord",                                CONV_EVENT, API_RUNTIME, SEC::EVENT};
  m["cudaEventQuery"]                                          = {"ascify::cudaEventQuery",                                 CONV_EVENT, API_RUNTIME, SEC::EVENT};
  m["cudaEventSynchronize"]                                    = {"ascify::cudaEventSynchronize",                           CONV_EVENT, API_RUNTIME, SEC::EVENT};
  m["cudaEventElapsedTime"]                                    = {"ascify::cudaEventElapsedTime",                           CONV_EVENT, API_RUNTIME, SEC::EVENT};
  // Error handling
  m["cudaGetLastError"]                                        = {"ascify::cudaGetLastError",                               CONV_ERROR_LOG, API_RUNTIME, SEC::ERROR_HANDLING};
  m["cudaPeekAtLastError"]                                     = {"ascify::cudaPeekAtLastError",                            CONV_ERROR_LOG, API_RUNTIME, SEC::ERROR_HANDLING};
  m["cudaGetErrorString"]                                      = {"ascify::cudaGetErrorString",                             CONV_ERROR_LOG, API_RUNTIME, SEC::ERROR_HANDLING};
  // There is no direct CANN counterpart. The shim preserves the CUDA template
  // signature and estimates per-vector-core occupancy from thread and UB limits.
  m["cudaOccupancyMaxActiveBlocksPerMultiprocessor"]            = {"ascify::cudaOccupancyMaxActiveBlocksPerMultiprocessor", CONV_OCCUPANCY, API_RUNTIME, SEC::OCCUPANCY, FULL};
  m["cudaOccupancyMaxPotentialBlockSize"]                       = {"ascify::cudaOccupancyMaxPotentialBlockSize", CONV_OCCUPANCY, API_RUNTIME, SEC::OCCUPANCY, FULL};
  return m;
}();
