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
  m["cudaMemcpyToSymbol"] = {"ascify::cudaMemcpyToSymbol", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaGetDeviceProperties"] = {"ascify::cudaGetDeviceProperties", CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
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
  m["cudaFreeHost"]                                            = {"ascify::cudaFreeHost",                                   CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // Device management
  // CUDA owns lazy runtime/context creation. Route all lifecycle-sensitive
  // calls through the compatibility manager instead of assuming an ACL
  // context already exists or directly changing a mismatched call shape.
  m["cudaGetDevice"]                                           = {"ascify::cudaGetDevice",                                  CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaSetDevice"]                                           = {"ascify::cudaSetDevice",                                  CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaGetDeviceCount"]                                      = {"ascify::cudaGetDeviceCount",                             CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  // CANN uses a different argument order and writes int64_t. Keep CUDA's
  // signature at the call site and adapt it in the compatibility shim.
  m["cudaDeviceGetAttribute"]                                  = {"ascify::cudaDeviceGetAttribute",                         CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaDeviceSynchronize"]                                   = {"ascify::cudaDeviceSynchronize",                          CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaDeviceReset"]                                         = {"ascify::cudaDeviceReset",                                CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  // beta3 has no per-kernel attribute API. The compatibility layer exposes the
  // subset needed by OneFlow's dynamic-UB selection code.
  m["cudaFuncGetAttributes"]                                   = {"ascify::cudaFuncGetAttributes",                          CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaFuncSetAttribute"]                                    = {"ascify::cudaFuncSetAttribute",                           CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  // Stream management
  m["cudaStreamCreate"]                                        = {"ascify::cudaStreamCreate",                               CONV_STREAM, API_RUNTIME, SEC::STREAM};
  m["cudaStreamCreateWithFlags"]                               = {"ascify::cudaStreamCreateWithFlags",                      CONV_STREAM, API_RUNTIME, SEC::STREAM};
  m["cudaStreamDestroy"]                                       = {"ascify::cudaStreamDestroy",                              CONV_STREAM, API_RUNTIME, SEC::STREAM};
  m["cudaStreamSynchronize"]                                   = {"ascify::cudaStreamSynchronize",                          CONV_STREAM, API_RUNTIME, SEC::STREAM};
  // Event management
  m["cudaEventCreate"]                                         = {"ascify::cudaEventCreate",                                CONV_EVENT, API_RUNTIME, SEC::EVENT};
  m["cudaEventDestroy"]                                        = {"ascify::cudaEventDestroy",                               CONV_EVENT, API_RUNTIME, SEC::EVENT};
  m["cudaEventRecord"]                                         = {"ascify::cudaEventRecord",                                CONV_EVENT, API_RUNTIME, SEC::EVENT};
  m["cudaEventSynchronize"]                                    = {"ascify::cudaEventSynchronize",                           CONV_EVENT, API_RUNTIME, SEC::EVENT};
  m["cudaEventElapsedTime"]                                    = {"ascify::cudaEventElapsedTime",                           CONV_EVENT, API_RUNTIME, SEC::EVENT};
  // Error handling
  m["cudaGetLastError"]                                        = {"ascify::cudaGetLastError",                               CONV_ERROR_LOG, API_RUNTIME, SEC::ERROR_HANDLING};
  m["cudaPeekAtLastError"]                                     = {"ascify::cudaPeekAtLastError",                            CONV_ERROR_LOG, API_RUNTIME, SEC::ERROR_HANDLING};
  m["cudaGetErrorString"]                                      = {"ascify::cudaGetErrorString",                             CONV_ERROR_LOG, API_RUNTIME, SEC::ERROR_HANDLING};
  // There is no direct CANN counterpart. The shim preserves the CUDA template
  // signature and estimates per-vector-core occupancy from thread and UB limits.
  m["cudaOccupancyMaxActiveBlocksPerMultiprocessor"]            = {"ascify::cudaOccupancyMaxActiveBlocksPerMultiprocessor", CONV_OCCUPANCY, API_RUNTIME, SEC::OCCUPANCY, FULL};
  return m;
}();
