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

// Maps CUDA header names to DPP header names
const std::map <llvm::StringRef, dppCounter> CUDA_INCLUDE_MAP {
  {"cuda_fp16.h", {"ascify/half2_compat.hpp", CONV_INCLUDE_CUDA_MAIN_H, API_RUNTIME, 0}},
  // CUDA includes -> Ascend ACL headers
  // Verified against CANN 9.0.0: acl_runtime.h does not exist; use acl/acl.h
  {"cuda.h",                                                {"acl/acl.h",                                                                                                  CONV_INCLUDE_CUDA_MAIN_H,    API_DRIVER, 0}},
  {"cuda_runtime.h",                                        {"acl/acl.h",                                                                                                  CONV_INCLUDE_CUDA_MAIN_H,    API_RUNTIME, 0}},
  // acl_cub/aclcub.hpp not yet available in CANN 9.0.0; cub::BlockReduce needs software re-implementation
  {"cub/cub.cuh",                                          {"acl_cub/aclcub.hpp",                                                                                         CONV_INCLUDE,                API_CUB, 0}},
  // acl/acl_bf16.h does not exist in CANN 9.0.0; use simt_api/asc_bf16.h for bfloat16_t support
  {"cuda_bf16.h",                                          {"simt_api/asc_bf16.h",                                                                                        CONV_INCLUDE,                API_RUNTIME, 0}},
  {"cooperative_groups.h",                                 {"ascify/cooperative_groups_compat.hpp",                                                                       CONV_INCLUDE,                API_RUNTIME, 0}},
  {"math_constants.h",                                    {"simt_api/math_constants.h",                                                              CONV_INCLUDE,                API_RUNTIME, 0}},
};

const std::map<llvm::StringRef, dppCounter> &CUDA_RENAMES_MAP() {
  static std::map<llvm::StringRef, dppCounter> ret;
  if (!ret.empty())
    return ret;
  // First run, so compute the union map.
  ret.insert(CUDA_RUNTIME_FUNCTION_MAP.begin(), CUDA_RUNTIME_FUNCTION_MAP.end());
  ret.insert(CUDA_RUNTIME_TYPE_NAME_MAP.begin(), CUDA_RUNTIME_TYPE_NAME_MAP.end());
  ret.insert(CUDA_DEVICE_TYPE_NAME_MAP.begin(), CUDA_DEVICE_TYPE_NAME_MAP.end());
  ret.insert(CUDA_DEVICE_FUNCTION_MAP.begin(), CUDA_DEVICE_FUNCTION_MAP.end());
  ret.insert(CUDA_DRIVER_TYPE_NAME_MAP.begin(), CUDA_DRIVER_TYPE_NAME_MAP.end());
  ret.insert(CUDA_DRIVER_FUNCTION_MAP.begin(), CUDA_DRIVER_FUNCTION_MAP.end());
  return ret;
};
