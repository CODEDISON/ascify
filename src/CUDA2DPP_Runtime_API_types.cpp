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
  m["cudaErrorMemoryAllocation"] = {"ACL_ERROR_RT_MEMORY_ALLOCATION",    CONV_DEFINE, API_RUNTIME, SEC::ERROR_HANDLING};
  m["cudaErrorInvalidDevice"]    = {"ACL_ERROR_RT_INVALID_DEVICEID",     CONV_DEFINE, API_RUNTIME, SEC::ERROR_HANDLING};
  m["cudaErrorNotReady"]         = {"ACL_ERROR_RT_EVENT_NOT_COMPLETE",   CONV_DEFINE, API_RUNTIME, SEC::ERROR_HANDLING};
  m["cudaErrorAssert"]           = {"ascify::cudaErrorAssert",           CONV_DEFINE, API_RUNTIME, SEC::ERROR_HANDLING};
  // Stream type
  m["cudaStream_t"]              = {"aclrtStream",                       CONV_TYPE, API_RUNTIME, SEC::STREAM};
  m["cudaHostFn_t"] = {"ascify::cudaHostFn_t", CONV_TYPE, API_RUNTIME, SEC::STREAM};
  // Keep CUDA flag values in the Ascify namespace. CUDA bit 0 is not
  // ACL_STREAM_FAST_LAUNCH even though both numeric values are one.
  m["cudaStreamDefault"]         = {"ascify::cudaStreamDefault",         CONV_DEFINE, API_RUNTIME, SEC::STREAM};
  m["cudaStreamNonBlocking"]     = {"ascify::cudaStreamNonBlocking",     CONV_DEFINE, API_RUNTIME, SEC::STREAM};
  m["cudaStreamLegacy"] = {"ascify::cudaStreamLegacy", CONV_DEFINE, API_RUNTIME, SEC::STREAM};
  m["cudaStreamPerThread"] = {"ascify::cudaStreamPerThread", CONV_DEFINE, API_RUNTIME, SEC::STREAM};
  // Event type
  m["cudaEvent_t"]                = {"aclrtEvent",                       CONV_TYPE, API_RUNTIME, SEC::EVENT};
  m["cudaEventWaitDefault"] = {"ascify::cudaEventWaitDefault", CONV_DEFINE, API_RUNTIME, SEC::EVENT};
  m["cudaEventWaitExternal"] = {"ascify::cudaEventWaitExternal", CONV_DEFINE, API_RUNTIME, SEC::EVENT};
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
  // Explicit texture adapter types; SDK reflection is separately rejected.
  m["cudaArray"] = {"ascify::cudaArray", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaArray_t"] = {"ascify::cudaArray_t", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaArray_const_t"] = {"ascify::cudaArray_const_t", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaTextureObject_t"] = {"ascify::cudaTextureObject_t", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaChannelFormatDesc"] = {"ascify::cudaChannelFormatDesc", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaChannelFormatKind"] = {"ascify::cudaChannelFormatKind", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaResourceDesc"] = {"ascify::cudaResourceDesc", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaResourceType"] = {"ascify::cudaResourceType", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaTextureDesc"] = {"ascify::cudaTextureDesc", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaTextureAddressMode"] = {"ascify::cudaTextureAddressMode", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaTextureFilterMode"] = {"ascify::cudaTextureFilterMode", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaTextureReadMode"] = {"ascify::cudaTextureReadMode", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaPitchedPtr"] = {"ascify::cudaPitchedPtr", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaExtent"] = {"ascify::cudaExtent", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaPos"] = {"ascify::cudaPos", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaMemcpy3DParms"] = {"ascify::cudaMemcpy3DParms", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaArrayLayered"] = {"0x01", CONV_DEFINE, API_RUNTIME, SEC::MEMORY};
  m["cudaArrayCubemap"] = {"0x04", CONV_DEFINE, API_RUNTIME, SEC::MEMORY};
  m["cudaResourceViewDesc"] = {"ascify::cudaResourceViewDesc", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaChannelFormatKindSigned"] = {"ascify::cudaChannelFormatKindSigned", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaChannelFormatKindUnsigned"] = {"ascify::cudaChannelFormatKindUnsigned", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaChannelFormatKindFloat"] = {"ascify::cudaChannelFormatKindFloat", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaChannelFormatKindNone"] = {"ascify::cudaChannelFormatKindNone", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaResourceTypeArray"] = {"ascify::cudaResourceTypeArray", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaResourceTypeMipmappedArray"] = {"ascify::cudaResourceTypeMipmappedArray", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaResourceTypeLinear"] = {"ascify::cudaResourceTypeLinear", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaResourceTypePitch2D"] = {"ascify::cudaResourceTypePitch2D", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaAddressModeWrap"] = {"ascify::cudaAddressModeWrap", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaAddressModeClamp"] = {"ascify::cudaAddressModeClamp", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaAddressModeMirror"] = {"ascify::cudaAddressModeMirror", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaAddressModeBorder"] = {"ascify::cudaAddressModeBorder", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaFilterModePoint"] = {"ascify::cudaFilterModePoint", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaFilterModeLinear"] = {"ascify::cudaFilterModeLinear", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaReadModeElementType"] = {"ascify::cudaReadModeElementType", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaReadModeNormalizedFloat"] = {"ascify::cudaReadModeNormalizedFloat", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  // Memory copy enums
  m["cudaMemcpyHostToDevice"]    = {"ACL_MEMCPY_HOST_TO_DEVICE",        CONV_DEFINE, API_RUNTIME, SEC::MEMORY};
  m["cudaMemcpyDeviceToHost"]    = {"ACL_MEMCPY_DEVICE_TO_HOST",        CONV_DEFINE, API_RUNTIME, SEC::MEMORY};
  m["cudaMemcpyDeviceToDevice"]  = {"ACL_MEMCPY_DEVICE_TO_DEVICE",      CONV_DEFINE, API_RUNTIME, SEC::MEMORY};
  m["cudaMemcpyHostToHost"]      = {"ACL_MEMCPY_HOST_TO_HOST",          CONV_DEFINE, API_RUNTIME, SEC::MEMORY};
  m["cudaMemcpyKind"]            = {"aclrtMemcpyKind",                  CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  // Host-allocation / registration / device flags keep their CUDA numeric
  // values; the compat implementations interpret them.
  m["cudaHostAllocDefault"] = {"ascify::cudaHostAllocDefault", CONV_DEFINE, API_RUNTIME, SEC::MEMORY};
  m["cudaHostAllocPortable"] = {"ascify::cudaHostAllocPortable", CONV_DEFINE, API_RUNTIME, SEC::MEMORY};
  m["cudaHostAllocMapped"] = {"ascify::cudaHostAllocMapped", CONV_DEFINE, API_RUNTIME, SEC::MEMORY};
  m["cudaHostAllocWriteCombined"] = {"ascify::cudaHostAllocWriteCombined", CONV_DEFINE, API_RUNTIME, SEC::MEMORY};
  m["cudaHostRegisterDefault"] = {"ascify::cudaHostRegisterDefault", CONV_DEFINE, API_RUNTIME, SEC::MEMORY};
  m["cudaHostRegisterPortable"] = {"ascify::cudaHostRegisterPortable", CONV_DEFINE, API_RUNTIME, SEC::MEMORY};
  m["cudaHostRegisterMapped"] = {"ascify::cudaHostRegisterMapped", CONV_DEFINE, API_RUNTIME, SEC::MEMORY};
  m["cudaDeviceScheduleAuto"] = {"ascify::cudaDeviceScheduleAuto", CONV_DEFINE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaDeviceScheduleSpin"] = {"ascify::cudaDeviceScheduleSpin", CONV_DEFINE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaDeviceScheduleYield"] = {"ascify::cudaDeviceScheduleYield", CONV_DEFINE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaDeviceScheduleBlockingSync"] = {"ascify::cudaDeviceScheduleBlockingSync", CONV_DEFINE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaDeviceBlockingSync"] = {"ascify::cudaDeviceBlockingSync", CONV_DEFINE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaDeviceMapHost"] = {"ascify::cudaDeviceMapHost", CONV_DEFINE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaEventDefault"] = {"ascify::cudaEventDefault", CONV_DEFINE, API_RUNTIME, 0};
  m["cudaEventBlockingSync"] = {"ascify::cudaEventBlockingSync", CONV_DEFINE, API_RUNTIME, 0};
  m["cudaEventDisableTiming"] = {"ascify::cudaEventDisableTiming", CONV_DEFINE, API_RUNTIME, 0};
  return m;
}();

// Map of CUDA Driver API type names to DPP type names
const std::map<llvm::StringRef, dppCounter> CUDA_DRIVER_TYPE_NAME_MAP = [] {
  std::map<llvm::StringRef, dppCounter> m;
  // Driver API error types
  m["CUresult"]       = {"aclError",              CONV_TYPE, API_DRIVER, 0};
  m["CUdevice"] = {"ascify::CUdevice", CONV_TYPE, API_DRIVER, 0};
  m["CUcontext"] = {"ascify::CUcontext", CONV_TYPE, API_DRIVER, 0};
  m["CUmodule"] = {"ascify::CUmodule", CONV_TYPE, API_DRIVER, 0};
  m["CUfunction"] = {"ascify::CUfunction", CONV_TYPE, API_DRIVER, 0};
  m["CUdeviceptr"] = {"ascify::CUdeviceptr", CONV_TYPE, API_DRIVER, 0};
  m["CUstream"] = {"ascify::CUstream", CONV_TYPE, API_DRIVER, 0};
  m["CUctxCreateParams"] = {"ascify::CUctxCreateParams", CONV_TYPE, API_DRIVER, 0};
  m["CUdevice_attribute"] = {"ascify::CUdevice_attribute", CONV_TYPE, API_DRIVER, 0};
  m["CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR"] = {"ascify::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR", CONV_DEFINE, API_DRIVER, 0};
  m["CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR"] = {"ascify::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR", CONV_DEFINE, API_DRIVER, 0};
  m["CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT"] = {"ascify::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT", CONV_DEFINE, API_DRIVER, 0};
  m["CU_DEVICE_ATTRIBUTE_CLOCK_RATE"] = {"ascify::CU_DEVICE_ATTRIBUTE_CLOCK_RATE", CONV_DEFINE, API_DRIVER, 0};
  m["CU_DEVICE_ATTRIBUTE_INTEGRATED"] = {"ascify::CU_DEVICE_ATTRIBUTE_INTEGRATED", CONV_DEFINE, API_DRIVER, 0};
  m["CU_DEVICE_ATTRIBUTE_COMPUTE_MODE"] = {"ascify::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE", CONV_DEFINE, API_DRIVER, 0};
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
