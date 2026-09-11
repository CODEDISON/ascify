#ifndef ASCIFY_DEVICE_CONTRACT_REJECT_HPP
#define ASCIFY_DEVICE_CONTRACT_REJECT_HPP

#include <simt_api/device_warp_functions.h>

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510 && !defined(ASCENDC_CPU_DEBUG)
// The public SDK header also changes assertion macros. Import only its device
// rejection function while preserving the caller's exact macro definitions.
#pragma push_macro("assert")
#pragma push_macro("ascendc_assert")
#pragma push_macro("ASC_INTERNAL_COUNT_ARGS_IMPL")
#undef ASC_INTERNAL_COUNT_ARGS_IMPL
#pragma push_macro("ASC_INTERNAL_COUNT_ARGS")
#undef ASC_INTERNAL_COUNT_ARGS
#pragma push_macro("ASC_INTERNAL_GET_ARG_COUNT")
#undef ASC_INTERNAL_GET_ARG_COUNT
#pragma push_macro("ASC_INTERNAL_ASSERT_IMPL")
#undef ASC_INTERNAL_ASSERT_IMPL
#pragma push_macro("ASC_INTERNAL_ASSERT_FUNC")
#undef ASC_INTERNAL_ASSERT_FUNC
#pragma push_macro("ASC_INTERNAL_ASSERT_1")
#undef ASC_INTERNAL_ASSERT_1
#pragma push_macro("ASC_INTERNAL_ASSERT_2")
#undef ASC_INTERNAL_ASSERT_2
#include <utils/debug/asc_assert.h>
#pragma pop_macro("ASC_INTERNAL_ASSERT_2")
#pragma pop_macro("ASC_INTERNAL_ASSERT_1")
#pragma pop_macro("ASC_INTERNAL_ASSERT_FUNC")
#pragma pop_macro("ASC_INTERNAL_ASSERT_IMPL")
#pragma pop_macro("ASC_INTERNAL_GET_ARG_COUNT")
#pragma pop_macro("ASC_INTERNAL_COUNT_ARGS")
#pragma pop_macro("ASC_INTERNAL_COUNT_ARGS_IMPL")
#pragma pop_macro("ascendc_assert")
#pragma pop_macro("assert")
#endif

namespace ascify {
namespace detail {
__SIMT_DEVICE_FUNCTIONS_DECL__ inline void reject_device_contract() {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510 && !defined(ASCENDC_CPU_DEBUG)
  // DT CANN 9.1.0 implements this as a device-invalid address store. The
  // ordinary O2 compiler and separate-process negative probe verified that
  // synchronization reports 507035. A C/C++ builtin trap instead lowers to
  // an unsupported host abort call in this SIMT backend.
  // Do not use assert(): NDEBUG/ASCENDC_DUMP can disable that macro.
  __asc_simt_vf::__trap();
#else
  // Preserve an actual rejection in host models, including CPU debug, where
  // the SDK trap itself is deliberately a no-op. No device pass is inferred.
  __builtin_trap();
#endif
}
} // namespace detail
} // namespace ascify
#endif
