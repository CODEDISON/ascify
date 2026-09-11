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

// Map of CUDA Runtime API type names to DPP type names
const std::map<llvm::StringRef, dppCounter> CUDA_RUNTIME_TYPE_NAME_MAP = [] {
  std::map<llvm::StringRef, dppCounter> m;
  // Error types
  m["cudaError_t"]               = {"aclError",                          CONV_TYPE, API_RUNTIME, SEC::ERROR_HANDLING};
  m["cudaSuccess"]               = {"ACL_SUCCESS",                       CONV_DEFINE, API_RUNTIME, SEC::ERROR_HANDLING};
  m["cudaErrorInvalidValue"]     = {"ACL_ERROR_RT_PARAM_INVALID",       CONV_DEFINE, API_RUNTIME, SEC::ERROR_HANDLING};
  m["cudaErrorMemoryAllocation"] = {"ACL_ERROR_RT_MALLOC_FAILED",        CONV_DEFINE, API_RUNTIME, SEC::ERROR_HANDLING};
  m["cudaErrorInvalidDevice"]    = {"ACL_ERROR_RT_DEVICE_NOT_FOUND",     CONV_DEFINE, API_RUNTIME, SEC::ERROR_HANDLING};
  // Stream type
  m["cudaStream_t"]              = {"aclrtStream",                       CONV_TYPE, API_RUNTIME, SEC::STREAM};
  // Keep CUDA flag values in the Ascify namespace. CUDA bit 0 is not
  // ACL_STREAM_FAST_LAUNCH even though both numeric values are one.
  m["cudaStreamDefault"]         = {"ascify::cudaStreamDefault",         CONV_DEFINE, API_RUNTIME, SEC::STREAM};
  m["cudaStreamNonBlocking"]     = {"ascify::cudaStreamNonBlocking",     CONV_DEFINE, API_RUNTIME, SEC::STREAM};
  // Event type
  m["cudaEvent_t"]                = {"aclrtEvent",                       CONV_TYPE, API_RUNTIME, SEC::EVENT};
  // Device attributes. Values beyond VECTOR_CORE_NUM drift between public ACL
  // and frozen RTS headers, so generated code names only Ascify constants.
  m["cudaDeviceProp"] = {"ascify::cudaDeviceProp", CONV_TYPE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaDevAttrMultiProcessorCount"]                 = {"ACL_DEV_ATTR_VECTOR_CORE_NUM",               CONV_TYPE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaDevAttrMaxThreadsPerMultiProcessor"]         = {"ascify::cudaDevAttrMaxThreadsPerVectorCore", CONV_TYPE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaDevAttrMaxSharedMemoryPerBlock"]             = {"ascify::cudaDevAttrLocalMemoryPerVectorCore", CONV_TYPE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaDeviceAttr"]                                    = {"aclrtDevAttr",                                        CONV_TYPE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaDevAttrMaxThreadsPerBlock"]                  = {"ascify::cudaDevAttrMaxThreadsPerBlock", CONV_TYPE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaDevAttrWarpSize"]                            = {"ascify::cudaDevAttrWarpSize", CONV_TYPE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaDevAttrMaxSharedMemoryPerBlockOptin"]      = {"ascify::cudaDevAttrLocalMemoryPerVectorCore", CONV_TYPE, API_RUNTIME, SEC::DEVICE_MGMT};
  // Minimal kernel-attribute surface used by layer_norm.cuh.
  m["cudaFuncAttributes"]                            = {"ascify::cudaFuncAttributes",               CONV_TYPE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaFuncAttribute"]                             = {"ascify::cudaFuncAttribute",                CONV_TYPE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaFuncAttributeMaxDynamicSharedMemorySize"]  = {"ascify::cudaFuncAttributeMaxDynamicSharedMemorySize", CONV_DEFINE, API_RUNTIME, SEC::DEVICE_MGMT};
  // Memory copy enums
  m["cudaMemcpyHostToDevice"]    = {"ACL_MEMCPY_HOST_TO_DEVICE",        CONV_DEFINE, API_RUNTIME, SEC::MEMORY};
  m["cudaMemcpyDeviceToHost"]    = {"ACL_MEMCPY_DEVICE_TO_HOST",        CONV_DEFINE, API_RUNTIME, SEC::MEMORY};
  m["cudaMemcpyDeviceToDevice"]  = {"ACL_MEMCPY_DEVICE_TO_DEVICE",      CONV_DEFINE, API_RUNTIME, SEC::MEMORY};
  m["cudaMemcpyHostToHost"]      = {"ACL_MEMCPY_HOST_TO_HOST",          CONV_DEFINE, API_RUNTIME, SEC::MEMORY};
  m["cudaMemcpyKind"]            = {"aclrtMemcpyKind",                  CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  return m;
}();

// Map of CUDA Driver API type names to DPP type names
const std::map<llvm::StringRef, dppCounter> CUDA_DRIVER_TYPE_NAME_MAP = [] {
  std::map<llvm::StringRef, dppCounter> m;
  // Driver API error types
  m["CUresult"]       = {"aclError",              CONV_TYPE, API_DRIVER, 0};
  m["CUDA_SUCCESS"]   = {"ACL_SUCCESS",            CONV_DEFINE, API_DRIVER, 0};
  return m;
}();

// Map of CUDA Driver API functions to DPP functions
const std::map<llvm::StringRef, dppCounter> CUDA_DRIVER_FUNCTION_MAP = [] {
  std::map<llvm::StringRef, dppCounter> m;
  // Driver API functions will be added here
  return m;
}();

// Map of CUDA Complex API types (cuBLAS, cuDNN etc.)
const std::map<llvm::StringRef, dppCounter> CUDA_COMPLEX_TYPE_NAME_MAP = [] {
  std::map<llvm::StringRef, dppCounter> m;
  return m;
}();

// Map of CUDA Complex API functions
const std::map<llvm::StringRef, dppCounter> CUDA_COMPLEX_FUNCTION_MAP = [] {
  std::map<llvm::StringRef, dppCounter> m;
  return m;
}();

// Map of cuFile API types
const std::map<llvm::StringRef, dppCounter> CUDA_FILE_TYPE_NAME_MAP = [] {
  std::map<llvm::StringRef, dppCounter> m;
  return m;
}();

// Map of cuFile API functions
const std::map<llvm::StringRef, dppCounter> CUDA_FILE_FUNCTION_MAP = [] {
  std::map<llvm::StringRef, dppCounter> m;
  return m;
}();
