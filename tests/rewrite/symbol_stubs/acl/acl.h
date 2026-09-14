#ifndef ASCIFY_TEST_SYMBOL_ACL_H
#define ASCIFY_TEST_SYMBOL_ACL_H

#include <stddef.h>

#define ACL_MAJOR_VERSION 1
#if defined(ASCIFY_TEST_SYMBOL_OLD_SDK)
#define ACL_MINOR_VERSION 16
#else
#define ACL_MINOR_VERSION 17
#endif

typedef int aclError;
enum aclrtMemcpyKind {
  ACL_MEMCPY_HOST_TO_HOST,
  ACL_MEMCPY_HOST_TO_DEVICE,
  ACL_MEMCPY_DEVICE_TO_HOST,
  ACL_MEMCPY_DEVICE_TO_DEVICE
};
static const aclError ACL_SUCCESS = 0;
static const aclError ACL_ERROR_RT_PARAM_INVALID = 107000;

#if !defined(ASCIFY_TEST_SYMBOL_OLD_SDK) && \
    !defined(ASCIFY_TEST_SYMBOL_NO_SIZE_API)
aclError aclrtGetSymbolSize(const void *symbol, size_t *size);
#endif
#if !defined(ASCIFY_TEST_SYMBOL_OLD_SDK) && \
    !defined(ASCIFY_TEST_SYMBOL_NO_COPY_API)
aclError aclrtMemcpyToSymbol(const void *symbol, const void *source,
                            size_t count, size_t offset, aclrtMemcpyKind kind);
#endif

#endif
