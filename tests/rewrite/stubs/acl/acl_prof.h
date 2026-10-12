#ifndef ASCIFY_TEST_STUB_ACL_PROF_H
#define ASCIFY_TEST_STUB_ACL_PROF_H
#include <acl/acl.h>

typedef void* aclrtContext;
struct aclprofConfig {};
struct aclprofAicoreEvents {};
enum aclprofAicoreMetrics { ACL_AICORE_NONE = 0xff };
static const uint64_t ACL_PROF_ACL_API = 1;
static const uint64_t ACL_PROF_TASK_TIME = 2;
static const aclError ACL_ERROR_PROFILING_FAILURE = 500005;

aclError aclrtGetCurrentContext(aclrtContext*);
aclError aclprofInit(const char*, size_t);
aclError aclprofFinalize();
aclprofConfig* aclprofCreateConfig(uint32_t*, uint32_t, aclprofAicoreMetrics,
                                  const aclprofAicoreEvents*, uint64_t);
aclError aclprofStart(const aclprofConfig*);
aclError aclprofStop(const aclprofConfig*);
aclError aclprofDestroyConfig(const aclprofConfig*);
#endif
