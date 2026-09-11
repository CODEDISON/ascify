#define __NPU_ARCH__ 3510
#if ASCIFY_TEST_ASSERT_DEFINED
#define assert(...) 31415
#define ascendc_assert(...) 92653
#define ASC_INTERNAL_COUNT_ARGS_IMPL(...) 400
#define ASC_INTERNAL_COUNT_ARGS(...) 401
#define ASC_INTERNAL_GET_ARG_COUNT(...) 402
#define ASC_INTERNAL_ASSERT_IMPL(...) 403
#define ASC_INTERNAL_ASSERT_FUNC(...) 404
#define ASC_INTERNAL_ASSERT_1(...) 405
#define ASC_INTERNAL_ASSERT_2(...) 406
#else
#undef assert
#undef ascendc_assert
#undef ASC_INTERNAL_COUNT_ARGS_IMPL
#undef ASC_INTERNAL_COUNT_ARGS
#undef ASC_INTERNAL_GET_ARG_COUNT
#undef ASC_INTERNAL_ASSERT_IMPL
#undef ASC_INTERNAL_ASSERT_FUNC
#undef ASC_INTERNAL_ASSERT_1
#undef ASC_INTERNAL_ASSERT_2
#endif
#include <ascify/device_contract_reject.hpp>
#include <ascify/device_contract_reject.hpp>
#if ASCIFY_TEST_ASSERT_DEFINED
static_assert(assert(false) == 31415, "assert macro was changed");
static_assert(ascendc_assert(false) == 92653, "ascendc_assert macro was changed");
static_assert(ASC_INTERNAL_COUNT_ARGS_IMPL(false) == 400, "SDK helper macro changed");
static_assert(ASC_INTERNAL_COUNT_ARGS(false) == 401, "SDK helper macro changed");
static_assert(ASC_INTERNAL_GET_ARG_COUNT(false) == 402, "SDK helper macro changed");
static_assert(ASC_INTERNAL_ASSERT_IMPL(false) == 403, "SDK helper macro changed");
static_assert(ASC_INTERNAL_ASSERT_FUNC(false) == 404, "SDK helper macro changed");
static_assert(ASC_INTERNAL_ASSERT_1(false) == 405, "SDK helper macro changed");
static_assert(ASC_INTERNAL_ASSERT_2(false) == 406, "SDK helper macro changed");
#else
#ifdef assert
#error "undefined assert must stay undefined"
#endif
#ifdef ascendc_assert
#error "undefined ascendc_assert must stay undefined"
#endif
#ifdef ASC_INTERNAL_COUNT_ARGS_IMPL
#error "SDK helper macro leaked"
#endif
#ifdef ASC_INTERNAL_COUNT_ARGS
#error "SDK helper macro leaked"
#endif
#ifdef ASC_INTERNAL_GET_ARG_COUNT
#error "SDK helper macro leaked"
#endif
#ifdef ASC_INTERNAL_ASSERT_IMPL
#error "SDK helper macro leaked"
#endif
#ifdef ASC_INTERNAL_ASSERT_FUNC
#error "SDK helper macro leaked"
#endif
#ifdef ASC_INTERNAL_ASSERT_1
#error "SDK helper macro leaked"
#endif
#ifdef ASC_INTERNAL_ASSERT_2
#error "SDK helper macro leaked"
#endif
#endif
int main() {
  ascify::detail::reject_device_contract();
  return __asc_simt_vf::trap_calls == 1 ? 0 : 1;
}
