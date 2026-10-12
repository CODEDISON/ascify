#define __aicore__
#define ASCIFY_TEST_CONTROLLABLE_EVENT_EX
#include <ascify/ascify_cuda_compat.hpp>
#include <cassert>
namespace { int creates=0,queries=0;unsigned flags=0;aclError failure=ACL_SUCCESS;
aclrtEventRecordedStatus state=ACL_EVENT_RECORDED_STATUS_COMPLETE;
aclrtEvent seen=nullptr; }
aclError aclrtCreateEventExWithFlag(aclrtEvent* event,unsigned value) {
 ++creates;flags=value;if(failure==ACL_SUCCESS)*event=reinterpret_cast<aclrtEvent>(1);return failure;
}
aclError aclrtQueryEventStatus(aclrtEvent event,aclrtEventRecordedStatus* output) {
 ++queries;seen=event;if(failure==ACL_SUCCESS)*output=state;return failure;
}
int main(){
 aclrtEvent event=nullptr;
 assert(ascify::cudaEventCreate(nullptr)==ACL_ERROR_RT_PARAM_INVALID&&creates==0);
 assert(ascify::cudaEventCreate(&event)==ACL_SUCCESS&&creates==1);
 assert(flags==(ACL_EVENT_TIME_LINE|ACL_EVENT_SYNC|ACL_EVENT_CAPTURE_STREAM_PROGRESS));
 assert(ascify::cudaEventQuery(nullptr)==ACL_ERROR_RT_PARAM_INVALID&&queries==0);
 assert(ascify::cudaEventQuery(event)==ACL_SUCCESS&&queries==1&&seen==event);
 state=ACL_EVENT_RECORDED_STATUS_NOT_READY;
 assert(ascify::cudaEventQuery(event)==ACL_ERROR_RT_EVENT_NOT_COMPLETE&&queries==2);
 failure=507021;
 assert(ascify::cudaEventQuery(event)==failure&&queries==3);
 assert(ascify::cudaEventCreate(&event)==failure&&event==nullptr&&creates==2);
 failure=ACL_SUCCESS;state=static_cast<aclrtEventRecordedStatus>(2);
 assert(ascify::cudaEventQuery(seen)==ACL_ERROR_FEATURE_UNSUPPORTED&&queries==4);
}
