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

// Map of CUDA Device type names to DPP type names
const std::map<llvm::StringRef, dppCounter> CUDA_DEVICE_TYPE_NAME_MAP = [] {
  std::map<llvm::StringRef, dppCounter> m;
  // Bfloat16 type (verified against CANN 9.0.0: Ascend uses bfloat16_t, not acl_bfloat16)
  m["nv_bfloat16"]    = {"bfloat16_t",       CONV_DEVICE_TYPE, API_RUNTIME, 0};
  m["nv_bfloat16_2"]  = {"bfloat16x2_t",     CONV_DEVICE_TYPE, API_RUNTIME, 0};
  return m;
}();

// Map of CUDA Device intrinsics/functions/macros to DPP equivalents
const std::map<llvm::StringRef, dppCounter> CUDA_DEVICE_FUNCTION_MAP = [] {
  std::map<llvm::StringRef, dppCounter> m;
  // Math constants (verified: ASCRT_* macros not available in CANN 9.0.0; use standard C limits)
  m["CUDART_INF_F"]      = {"ASCRT_INF_F",     CONV_NUMERIC_LITERAL, API_RUNTIME, 0};
  m["CUDART_INF"]        = {"ASCRT_INF_F",                CONV_NUMERIC_LITERAL, API_RUNTIME, UNSUPPORTED};// double type need todo
  m["CUDART_NAN_F"]      = {"ASCRT_NAN_F",    CONV_NUMERIC_LITERAL, API_RUNTIME, 0};
  m["CUDART_NAN"]        = {"ASCRT_INF_F",                CONV_NUMERIC_LITERAL, API_RUNTIME, UNSUPPORTED};// double type need todo
  m["CUDART_MAX_NORMAL_F"] = {"__FLT_MAX__", CONV_NUMERIC_LITERAL, API_RUNTIME, 0};
  m["CUDART_MIN_NORMAL_F"] = {"__FLT_MIN__", CONV_NUMERIC_LITERAL, API_RUNTIME, 0};
  m["CUDART_NEG_ZERO_F"] = {"(-0.0f)",    CONV_NUMERIC_LITERAL, API_RUNTIME, 0};
  m["CUDART_ZERO_F"]     = {"(0.0f)",     CONV_NUMERIC_LITERAL, API_RUNTIME, 0};
  // Preserve the CUDA shuffle signature. The compatibility shim validates that
  // the mask is a full warp before calling the unmasked beta3 asc_shfl* API.
  m["__shfl_down_sync"]  = {"ascify::shfl_down_sync", CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__shfl_up_sync"]    = {"ascify::shfl_up_sync",   CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__shfl_xor_sync"]   = {"ascify::shfl_xor_sync",  CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__shfl_sync"]       = {"ascify::shfl_sync",      CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__any_sync"]        = {"ascify::any_sync",       CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__all_sync"]        = {"ascify::all_sync",       CONV_DEVICE_FUNC, API_RUNTIME, 0};
  // The facade retains CUDA tile result types while using native CANN groups.
  m["cooperative_groups"] = {"ascify_cg",           CONV_DEVICE_FUNC, API_RUNTIME, 0};
  // Keep synchronization behind the dual-header-family compatibility layer.
  // Public 8.5 deliberately rejects block vote collectives that its aggregate
  // SIMT interface does not expose.
  m["__syncthreads"]     = {"ascify::syncthreads",       CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__syncthreads_and"] = {"ascify::syncthreads_and",   CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__syncthreads_or"]  = {"ascify::syncthreads_or",    CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__syncthreads_count"] = {"ascify::syncthreads_count", CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__syncwarp"]        = {"ascify::syncwarp",    CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__threadfence_block"] = {"ascify::threadfence_block", CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__threadfence"]       = {"ascify::threadfence",       CONV_DEVICE_FUNC, API_RUNTIME, 0};
  // Neither admitted target family has a proven host/system-scope equivalent.
  // The wrapper is intentionally ill-formed only when called.
  m["__threadfence_system"] = {"ascify::threadfence_system", CONV_DEVICE_FUNC, API_RUNTIME, 0};
  // Use the beta3 SIMT math entry points rather than host/compiler builtins.
  m["__expf"]            = {"expf",              CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__logf"]            = {"logf",              CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__log2f"]           = {"log2f",             CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__log10f"]          = {"log10f",            CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__fdividef"]        = {"ascify::fdividef",  CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__fmul_rn"]         = {"__fmul_rn",         CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__fmaf_rn"]          = {"__fmaf_rn",         CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__fadd_rn"]          = {"__fadd_rn",         CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__fsub_rn"]          = {"__fsub_rn",         CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__frsqrt_rn"]        = {"rsqrtf",             CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__fsqrt_rn"]         = {"sqrtf",              CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__frcp_rn"]          = {"__frcp_rn",         CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__float2half_rn"]    = {"__float2half_rn",    CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__half2float"]       = {"__half2float",       CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__float2half2_rn"]   = {"ascify::float2half2_rn", CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__half22float2"]     = {"ascify::half22float2", CONV_DEVICE_FUNC, API_RUNTIME, 0};
  // The installed CANN 9.1 SIMT surface names paired half arithmetic x2.
  // Keep fused __hfma2 distinct from separately rounded multiply/add.
  m["__hadd2"]           = {"__haddx2",          CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__hmul2"]           = {"__hmulx2",          CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__hfma2"]           = {"__hfmax2",          CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__floats2half2_rn"]  = {"__floats2half2_rn", CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__low2float"]       = {"__low2float",       CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__high2float"]      = {"__high2float",      CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__int_as_float"]     = {"__int_as_float",     CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__float_as_int"]     = {"__float_as_int",     CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__longlong_as_double"] = {"__longlong_as_double", CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__double_as_longlong"] = {"__double_as_longlong", CONV_DEVICE_FUNC, API_RUNTIME, 0};
  // Trap -> __trap (CCE directly supports __trap)
  m["__trap"]             = {"__builtin_trap",          CONV_DEVICE_FUNC, API_RUNTIME, 0};
  // The vector core has float math APIs. Do not rewrite the `double` token
  // globally: host signatures and type traits must retain their source
  // semantics. AscifyAction applies the narrower AST rule only to proven
  // by-value scalar double parameters on CUDA global functions.
  m["exp"]                = {"expf",              CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["log"]                = {"logf",              CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["sqrt"]               = {"sqrtf",             CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["rsqrt"]              = {"rsqrtf",            CONV_DEVICE_FUNC, API_RUNTIME, 0};
  // std::max/min -> Ascend SIMT equivalents
//   m["std::max"]           = {"llmax",             CONV_DEVICE_FUNC, API_RUNTIME, 0};
//   m["std::min"]           = {"llmin",             CONV_DEVICE_FUNC, API_RUNTIME, 0};
  // blockIdx/threadIdx dim3 fields (keep same on SIMT)
  m["blockIdx"]           = {"blockIdx",          CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["threadIdx"]          = {"threadIdx",         CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["blockDim"]           = {"blockDim",          CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["gridDim"]            = {"gridDim",           CONV_DEVICE_FUNC, API_RUNTIME, 0};
  // CUB namespace -> aclcub (note: acl_cub library not yet available in CANN 9.0.0)
  m["cub"]                = {"aclcub",            CONV_DEVICE_FUNC, API_CUB, 0};

  // ---------- CUDA qualifier replacements (note: may need Clang-level Attr rewrite) ----------
  // The admitted compiler families require different spellings: public 8.5
  // requires aicore, while coherent 9.1 beta3 rejects device+global.  Let the
  // injected compatibility header select the spelling instead of baking one
  // compiler's attributes into every translated source.
  m["__global__"]         = {"ASCIFY_GLOBAL",        CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__device__"]         = {"__aicore__",          CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__forceinline__"]    = {"ASCIFY_FORCEINLINE", CONV_DEVICE_FUNC, API_RUNTIME, 0};
  // __shared__ -> __ubuf__ (Ascend C Unified Buffer qualifier, defined as empty macro in cpu_stub.hpp)
  m["__shared__"]         = {"__ubuf__",         CONV_DEVICE_FUNC, API_RUNTIME, 0};
  // Preserve CUDA declaration alignment instead of dropping the attribute.
  m["__align__"]          = {"ASCIFY_ALIGN",     CONV_DEVICE_FUNC, API_RUNTIME, 0};
  return m;
}();
