#ifndef ASCIFY_TEST_SDK_ASSERT_H
#define ASCIFY_TEST_SDK_ASSERT_H
// Model the SDK's macro side effects and direct trap routing, not NPU errors.
#undef assert
#undef ascendc_assert
#define assert(...) SDK_DISABLED_ASSERT
#define ascendc_assert(...) SDK_DISABLED_ASCENDC_ASSERT
#define ASC_INTERNAL_COUNT_ARGS_IMPL(...) 1
#define ASC_INTERNAL_COUNT_ARGS(...) 1
#define ASC_INTERNAL_GET_ARG_COUNT(...) 1
#define ASC_INTERNAL_ASSERT_IMPL(...) 1
#define ASC_INTERNAL_ASSERT_FUNC(...) 1
#define ASC_INTERNAL_ASSERT_1(...) 1
#define ASC_INTERNAL_ASSERT_2(...) 1
namespace __asc_simt_vf {
inline unsigned trap_calls = 0;
inline void __trap() { ++trap_calls; }
}
#endif
