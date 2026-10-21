// Execute the public wrappers against explicit CPU memory/test-double ACL.
// This models lifetime and requested extents; it is not real ACL/NPU evidence.
#define ASCIFY_TEST_CUSTOM_CURRENT_CONTEXT
#define aclrtMalloc unused_malloc
#define aclrtFree unused_free
#define aclrtMemcpy unused_memcpy
#define aclrtGetDevice unused_get_device
#define aclrtResetDevice unused_reset_device
#include <acl/acl.h>
#undef aclrtMalloc
#undef aclrtFree
#undef aclrtMemcpy
#undef aclrtGetDevice
#undef aclrtResetDevice
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

struct Block { size_t bytes; int32_t device; };
static std::map<void *,Block> memory;
static std::mutex memory_mutex, schedule_mutex;
static std::condition_variable schedule;
static thread_local int32_t model_device=0;
static thread_local void *model_context=reinterpret_cast<void *>(0x42);
static bool fail_allocation=false, fail_free=false, fail_reset=false, fail_context=false;
static thread_local bool pause_current_context=false;
static bool context_entered=false, resume_context=false;
static bool recycle_next=false;
static void *recycled=nullptr;
static size_t recycled_bytes=0;
static unsigned successful_allocations=0, successful_frees=0;

inline aclError aclrtGetDevice(int32_t *device) { *device=model_device; return ACL_SUCCESS; }
inline aclError aclrtGetCurrentContext(void **context) {
  if(fail_context) return ACL_ERROR_RT_CONTEXT_NULL;
  if(pause_current_context) {
    std::unique_lock<std::mutex> lock(schedule_mutex);
    context_entered=true;schedule.notify_all();
    assert(schedule.wait_for(lock,std::chrono::seconds(5),[]{return resume_context;}));
  }
  *context=model_context; return ACL_SUCCESS;
}
inline aclError aclrtMalloc(void **out,size_t bytes,aclrtMemMallocPolicy) {
  if(fail_allocation) return ACL_ERROR_RT_MEMORY_ALLOCATION;
  std::lock_guard<std::mutex> lock(memory_mutex);
  void *value=nullptr;
  if(recycled && recycled_bytes==bytes) {value=recycled;recycled=nullptr;}
  else value=std::malloc(bytes);
  assert(value && bytes);std::memset(value,0xa5,bytes);
  assert(memory.emplace(value,Block{bytes,model_device}).second);
  ++successful_allocations;*out=value;return ACL_SUCCESS;
}
inline aclError aclrtFree(void *pointer) {
  std::lock_guard<std::mutex> lock(memory_mutex);
  const auto found=memory.find(pointer);
  if(found==memory.end()) return ACL_ERROR_RT_PARAM_INVALID;
  if(fail_free) return ACL_ERROR_RT_PARAM_INVALID;
  if(recycle_next) {
    assert(!recycled);recycled=pointer;recycled_bytes=found->second.bytes;recycle_next=false;
  } else std::free(pointer);
  memory.erase(found);++successful_frees;return ACL_SUCCESS;
}
static bool range(const void *pointer,size_t bytes) {
  const uintptr_t address=reinterpret_cast<uintptr_t>(pointer);
  for(const auto &entry:memory) {
    const uintptr_t base=reinterpret_cast<uintptr_t>(entry.first);
    if(address>=base && address-base<=entry.second.bytes &&
       bytes<=entry.second.bytes-(address-base)) return true;
  }
  return false;
}
inline aclError aclrtMemcpy(void *dst,size_t capacity,const void *src,size_t bytes,aclrtMemcpyKind kind) {
  std::lock_guard<std::mutex> lock(memory_mutex);
  assert(capacity>=bytes);
  if(kind==ACL_MEMCPY_HOST_TO_DEVICE || kind==ACL_MEMCPY_DEVICE_TO_DEVICE) assert(range(dst,bytes));
  if(kind==ACL_MEMCPY_DEVICE_TO_HOST || kind==ACL_MEMCPY_DEVICE_TO_DEVICE) assert(range(src,bytes));
  std::memmove(dst,src,bytes);return ACL_SUCCESS;
}
inline aclError aclrtResetDevice(int32_t device) {
  if(fail_reset) return ACL_ERROR_RT_PARAM_INVALID;
  std::lock_guard<std::mutex> lock(memory_mutex);
  for(auto it=memory.begin();it!=memory.end();) {
    if(it->second.device!=device) {++it;continue;}
    std::free(it->first);++successful_frees;it=memory.erase(it);
  }
  return ACL_SUCCESS;
}

#define __aicore__
#include <ascify/ascify_cuda_compat.hpp>
#define __NPU_COMPILER_INTERNAL_PURE_SIMT__
#include <ascify/texture_compat.hpp>

static ascify::cudaResourceDesc resource(void *pointer,size_t width=4,size_t height=2,size_t pitch=16) {
  ascify::cudaResourceDesc r{};r.resType=ascify::cudaResourceTypePitch2D;
  r.res.pitch2D={pointer,ascify::cudaCreateChannelDesc<unsigned int>(),width,height,pitch};return r;
}
static ascify::cudaTextureDesc texture() {
  ascify::cudaTextureDesc t{};
  t.addressMode[0]=t.addressMode[1]=t.addressMode[2]=ascify::cudaAddressModeClamp;return t;
}
static aclError create(ascify::cudaTextureObject_t &token,const ascify::cudaResourceDesc &r) {
  auto t=texture();return ascify::cudaCreateTextureObject(&token,&r,&t,nullptr);
}
static void invalid(const ascify::cudaResourceDesc &r) {
  ascify::cudaTextureObject_t token=123;
  assert(create(token,r)!=ACL_SUCCESS && token==123);
}
static void retired(ascify::cudaTextureObject_t token) {
  assert(ascify::tex2D<unsigned int>(token,0,0)==0);
  assert(ascify::cudaDestroyTextureObject(token)==ACL_ERROR_FEATURE_UNSUPPORTED);
}
static void empty() {
  assert(memory.empty());
  for(const auto &r:ascify::detail::runtimeAllocationRegistry().records) assert(!r.pointer);
  for(const auto &r:ascify::detail::textureRegistry().allocations) assert(!r.storage && !r.references);
}

int main() {
  unsigned int *p=nullptr;
  unsigned int values[]={0,16777217U,0xffffffffU,0x80000000U,0x7fc00001U,9,17,25};
  // Allocation predates the first texture: lazy hook installation cannot lose it.
  assert(!ascify::detail::runtimeTextureHooks.load());
  assert(ascify::cudaMalloc(&p,sizeof(values))==ACL_SUCCESS);
  assert(ascify::cudaMemcpy(p,values,sizeof(values),ACL_MEMCPY_HOST_TO_DEVICE)==ACL_SUCCESS);
  ascify::cudaTextureObject_t a=0,b=0;
  // Pause the first create in textureCurrent, before TextureLock and before
  // hook installation. Only the new owner lock can serialize this free.
  std::atomic<bool> free_started{false},free_finished{false};
  aclError create_result=-1,free_result=-1;
  std::thread creator([&]{pause_current_context=true;create_result=create(a,resource(p));});
  {std::unique_lock<std::mutex> lock(schedule_mutex);
    assert(schedule.wait_for(lock,std::chrono::seconds(5),[]{return context_entered;}));}
  assert(!ascify::detail::runtimeTextureHooks.load());
  std::thread freer([&]{free_started=true;schedule.notify_all();free_result=ascify::cudaFree(p);free_finished=true;schedule.notify_all();});
  {std::unique_lock<std::mutex> lock(schedule_mutex);
    assert(schedule.wait_for(lock,std::chrono::seconds(5),[&]{return free_started.load();}));
    assert(!schedule.wait_for(lock,std::chrono::milliseconds(20),[&]{return free_finished.load();}));
    resume_context=true;schedule.notify_all();}
  creator.join();freer.join();
  assert(create_result==ACL_SUCCESS && free_result==ACL_SUCCESS);retired(a);empty();
  assert(ascify::cudaMalloc(&p,sizeof(values))==ACL_SUCCESS);
  assert(ascify::cudaMemcpy(p,values,sizeof(values),ACL_MEMCPY_HOST_TO_DEVICE)==ACL_SUCCESS);
  invalid(resource(values)); // untracked host memory
  invalid(resource(p+1)); // exact base only
  invalid(resource(p,4,3)); // full last row exceeds requested extent
  invalid(resource(p,2,2,10)); // misaligned row pitch, although span fits
  invalid(resource(p,4,2,SIZE_MAX-3)); // checked range overflow
  model_context=reinterpret_cast<void *>(0x43);invalid(resource(p));
  model_context=reinterpret_cast<void *>(0x42);model_device=1;invalid(resource(p));model_device=0;
  assert(create(a,resource(p))==ACL_SUCCESS);
  assert(create(b,resource(p,2,4,8))==ACL_SUCCESS);
  for(unsigned i=0;i<8;++i) {
    assert(ascify::tex2D<unsigned int>(a,i%4,i/4)==values[i]);
    assert(ascify::tex2D<unsigned int>(b,i%2,i/2)==values[i]);
  }
  model_context=reinterpret_cast<void *>(0x43);
  assert(ascify::cudaFree(p)==ACL_ERROR_RT_PARAM_INVALID);
  model_context=reinterpret_cast<void *>(0x42);
  // Force an address reuse while two old descriptors remain live.
  recycle_next=true;auto *old=p;assert(ascify::cudaFree(p)==ACL_SUCCESS);
  assert(ascify::cudaMalloc(&p,sizeof(values))==ACL_SUCCESS && p==old);
  assert(ascify::cudaMemcpy(p,values,sizeof(values),ACL_MEMCPY_HOST_TO_DEVICE)==ACL_SUCCESS);
  ascify::cudaTextureObject_t fresh=0;assert(create(fresh,resource(p))==ACL_SUCCESS);
  retired(a);retired(b);
  assert(ascify::tex2D<unsigned int>(fresh,2,0)==values[2]);
  assert(ascify::cudaDestroyTextureObject(fresh)==ACL_SUCCESS);
  assert(ascify::cudaFree(p)==ACL_SUCCESS);
  assert(ascify::cudaFree(p)==ACL_ERROR_RT_PARAM_INVALID);empty();

  fail_allocation=true;p=reinterpret_cast<unsigned int *>(1);
  assert(ascify::cudaMalloc(&p,32)==ACL_ERROR_RT_MEMORY_ALLOCATION && !p);fail_allocation=false;empty();
  // Optional proof failure must not turn a valid ordinary allocation into failure.
  fail_context=true;assert(ascify::cudaMalloc(&p,32)==ACL_SUCCESS);fail_context=false;
  invalid(resource(p));assert(ascify::cudaFree(p)==ACL_SUCCESS);empty();
  assert(ascify::cudaMalloc(&p,32)==ACL_SUCCESS);assert(create(a,resource(p))==ACL_SUCCESS);
  fail_free=true;assert(ascify::cudaFree(p)!=ACL_SUCCESS);fail_free=false;
  assert(ascify::detail::runtimeAllocationLocked(p)); // failed free retains owner
  retired(a);assert(create(b,resource(p))==ACL_SUCCESS);
  assert(ascify::cudaDestroyTextureObject(b)==ACL_SUCCESS);
  assert(ascify::cudaFree(p)==ACL_SUCCESS);empty();

  std::vector<void *> capacity(129,nullptr);
  for(auto &pointer:capacity) assert(ascify::cudaMalloc(&pointer,32)==ACL_SUCCESS);
  assert(create(a,resource(capacity[127]))==ACL_SUCCESS);invalid(resource(capacity[128]));
  assert(ascify::cudaDestroyTextureObject(a)==ACL_SUCCESS);
  for(auto pointer:capacity) assert(ascify::cudaFree(pointer)==ACL_SUCCESS);empty();

  assert(ascify::cudaMalloc(&p,32)==ACL_SUCCESS);assert(create(a,resource(p))==ACL_SUCCESS);
  fail_reset=true;assert(ascify::cudaDeviceReset()!=ACL_SUCCESS);fail_reset=false;
  assert(ascify::detail::runtimeAllocationLocked(p));
  assert(ascify::cudaDeviceReset()==ACL_SUCCESS);empty();
  assert(ascify::cudaDestroyTextureObject(a)==ACL_ERROR_RT_PARAM_INVALID);

  assert(!recycled && successful_allocations==successful_frees);
}
