// Public-wrapper host memory model. ACL definitions below are test doubles;
// passing this test is neither target execution nor original Samples evidence.
#define ASCIFY_TEST_CUSTOM_CURRENT_CONTEXT
#define aclrtMalloc ascify_unused_stub_malloc
#define aclrtFree ascify_unused_stub_free
#define aclrtMemcpy ascify_unused_stub_memcpy
#define aclrtSynchronizeStream ascify_unused_stub_sync
#include <acl/acl.h>
#undef aclrtMalloc
#undef aclrtFree
#undef aclrtMemcpy
#undef aclrtSynchronizeStream
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

static std::map<void *,size_t> allocations;
static unsigned contexts[2], current_context=0, copies=0, syncs=0;
static unsigned fail_copy_call=0;
static aclError sync_status=ACL_SUCCESS, copy_status=ACL_SUCCESS;
inline aclError aclrtGetCurrentContext(void **context) {
  *context=&contexts[current_context]; return ACL_SUCCESS;
}
static bool device_range(const void *pointer,size_t count) {
  const uintptr_t address=reinterpret_cast<uintptr_t>(pointer);
  for(const auto &allocation:allocations) {
    const uintptr_t base=reinterpret_cast<uintptr_t>(allocation.first);
    if(address>=base && address-base<=allocation.second &&
        count<=allocation.second-(address-base)) return true;
  }
  return false;
}
inline aclError aclrtMalloc(void **pointer,size_t bytes,aclrtMemMallocPolicy) {
  *pointer=std::malloc(bytes); assert(*pointer);
  std::memset(*pointer,0xa5,bytes); assert(allocations.emplace(*pointer,bytes).second);
  return ACL_SUCCESS;
}
inline aclError aclrtFree(void *pointer) {
  assert(allocations.erase(pointer)==1); std::free(pointer); return ACL_SUCCESS;
}
inline aclError aclrtSynchronizeStream(aclrtStream) {
  ++syncs; return sync_status;
}
inline aclError aclrtMemcpy(void *dst,size_t capacity,const void *src,size_t bytes,
    aclrtMemcpyKind kind) {
  ++copies;
  if(copy_status!=ACL_SUCCESS) return copy_status;
  if(copies==fail_copy_call) return ACL_ERROR_RT_PARAM_INVALID;
  assert(dst && src && capacity>=bytes);
  if(kind==ACL_MEMCPY_HOST_TO_DEVICE || kind==ACL_MEMCPY_DEVICE_TO_DEVICE)
    assert(device_range(dst,bytes));
  if(kind==ACL_MEMCPY_DEVICE_TO_HOST || kind==ACL_MEMCPY_DEVICE_TO_DEVICE)
    assert(device_range(src,bytes));
  std::memmove(dst,src,bytes); return ACL_SUCCESS;
}
#define __aicore__
#include <ascify/ascify_cuda_compat.hpp>
#define __NPU_COMPILER_INTERNAL_PURE_SIMT__
#include <ascify/texture_compat.hpp>

static void verify(ascify::cudaArray_t array,const std::vector<unsigned char> &gold) {
  std::vector<unsigned char> actual(160,0);
  ascify::cudaMemcpy3DParms p{};
  p.srcArray=array;
  p.dstPtr=ascify::make_cudaPitchedPtr(actual.data(),32,32,5);
  p.extent=ascify::make_cudaExtent(8,5,1);
  p.kind=ACL_MEMCPY_DEVICE_TO_HOST;
  assert(ascify::cudaMemcpy3D(&p)==ACL_SUCCESS);
  assert(actual==gold);
}
template<class T> static void rectangles() {
  auto channel=ascify::cudaCreateChannelDesc<T>();
  ascify::cudaArray_t array=nullptr;
  assert(ascify::cudaMallocArray(&array,&channel,8,5)==ACL_SUCCESS);
  std::vector<unsigned char> gold(160,0xa5), source(60);
  for(unsigned i=0;i<60;++i) source[i]=static_cast<unsigned char>(i*37+11);
  unsigned before_sync=syncs, before_copy=copies;
  assert(ascify::cudaMemcpy2DToArray(array,4,1,source.data(),13,8,3,
      ACL_MEMCPY_HOST_TO_DEVICE)==ACL_SUCCESS);
  assert(syncs==before_sync+1 && copies==before_copy+3);
  for(unsigned r=0;r<3;++r) for(unsigned b=0;b<8;++b)
    gold[(r+1)*32+4+b]=source[r*13+b];
  verify(array,gold);

  void *device=nullptr;
  assert(ascify::cudaMalloc(&device,source.size())==ACL_SUCCESS);
  assert(ascify::cudaMemcpy(device,source.data(),source.size(),ACL_MEMCPY_HOST_TO_DEVICE)==ACL_SUCCESS);
  before_sync=syncs; before_copy=copies;
  assert(ascify::cudaMemcpy2DToArray(array,8,0,device,20,8,3,
      ACL_MEMCPY_DEVICE_TO_DEVICE)==ACL_SUCCESS);
  assert(syncs==before_sync+1 && copies==before_copy+3);
  for(unsigned r=0;r<3;++r) for(unsigned b=0;b<8;++b)
    gold[r*32+8+b]=source[r*20+b];
  verify(array,gold);

  // All parameter/ownership failures precede any row write or stream sync.
  before_copy=copies; before_sync=syncs;
  const auto bad=ACL_ERROR_RT_PARAM_INVALID;
  assert(ascify::cudaMemcpy2DToArray(array,28,0,source.data(),13,8,1,ACL_MEMCPY_HOST_TO_DEVICE)==bad);
  assert(ascify::cudaMemcpy2DToArray(array,9,0,source.data(),13,8,1,ACL_MEMCPY_HOST_TO_DEVICE)==ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(ascify::cudaMemcpy2DToArray(array,8,0,source.data(),13,7,1,ACL_MEMCPY_HOST_TO_DEVICE)==ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(ascify::cudaMemcpy2DToArray(array,0,4,source.data(),13,8,2,ACL_MEMCPY_HOST_TO_DEVICE)==bad);
  assert(ascify::cudaMemcpy2DToArray(array,0,0,source.data(),7,8,1,ACL_MEMCPY_HOST_TO_DEVICE)==bad);
  assert(ascify::cudaMemcpy2DToArray(array,SIZE_MAX-3,0,source.data(),13,8,1,ACL_MEMCPY_HOST_TO_DEVICE)==bad);
  assert(ascify::cudaMemcpy2DToArray(array,0,SIZE_MAX,source.data(),13,8,1,ACL_MEMCPY_HOST_TO_DEVICE)==bad);
  assert(ascify::cudaMemcpy2DToArray(array,0,0,source.data(),SIZE_MAX,8,3,ACL_MEMCPY_HOST_TO_DEVICE)==bad);
  assert(ascify::cudaMemcpy2DToArray(array,0,0,reinterpret_cast<void *>(UINTPTR_MAX-3),13,8,1,ACL_MEMCPY_HOST_TO_DEVICE)==bad);
  assert(ascify::cudaMemcpy2DToArray(array,0,0,nullptr,13,8,1,ACL_MEMCPY_HOST_TO_DEVICE)==bad);
  assert(ascify::cudaMemcpy2DToArray(nullptr,0,0,source.data(),13,8,1,ACL_MEMCPY_HOST_TO_DEVICE)==bad);
  assert(ascify::cudaMemcpy2DToArray(reinterpret_cast<ascify::cudaArray_t>(device),0,0,source.data(),13,8,1,ACL_MEMCPY_HOST_TO_DEVICE)==bad);
  assert(ascify::cudaMemcpy2DToArray(array,0,0,source.data(),13,8,1,ACL_MEMCPY_DEVICE_TO_HOST)==ACL_ERROR_FEATURE_UNSUPPORTED);
  current_context=1;
  assert(ascify::cudaMemcpy2DToArray(array,0,0,source.data(),13,8,1,ACL_MEMCPY_HOST_TO_DEVICE)==bad);
  current_context=0;
  assert(copies==before_copy && syncs==before_sync);
  verify(array,gold);

  before_copy=copies;
  sync_status=ACL_ERROR_RT_PARAM_INVALID;
  assert(ascify::cudaMemcpy2DToArray(array,0,0,source.data(),13,8,1,ACL_MEMCPY_HOST_TO_DEVICE)==sync_status);
  sync_status=ACL_SUCCESS;
  assert(copies==before_copy); verify(array,gold);
  copy_status=ACL_ERROR_RT_PARAM_INVALID;
  assert(ascify::cudaMemcpy2DToArray(array,0,0,source.data(),13,8,3,ACL_MEMCPY_HOST_TO_DEVICE)==copy_status);
  copy_status=ACL_SUCCESS; verify(array,gold);

  // A later ACL failure propagates after already completed writes. The
  // adapter promises no rollback, and must stop before subsequent rows.
  before_copy=copies; fail_copy_call=copies+2;
  assert(ascify::cudaMemcpy2DToArray(array,20,1,source.data(),13,8,3,ACL_MEMCPY_HOST_TO_DEVICE)==ACL_ERROR_RT_PARAM_INVALID);
  fail_copy_call=0;
  assert(copies==before_copy+2);
  for(unsigned b=0;b<8;++b) gold[32+20+b]=source[b];
  verify(array,gold);

  assert(ascify::cudaMemcpy2DToArray(array,32,5,nullptr,0,0,0,ACL_MEMCPY_HOST_TO_DEVICE)==ACL_SUCCESS);
  assert(ascify::cudaFree(device)==ACL_SUCCESS);
  assert(ascify::cudaFreeArray(array)==ACL_SUCCESS);
  assert(ascify::cudaMemcpy2DToArray(array,0,0,source.data(),13,8,1,ACL_MEMCPY_HOST_TO_DEVICE)==bad);
  assert(allocations.empty());
  assert(ascify::cudaMalloc3DArray(&array,&channel,ascify::make_cudaExtent(8,5,2),ascify::cudaArrayLayered)==ACL_SUCCESS);
  assert(ascify::cudaMemcpy2DToArray(array,0,0,source.data(),13,8,1,ACL_MEMCPY_HOST_TO_DEVICE)==bad);
  assert(ascify::cudaFreeArray(array)==ACL_SUCCESS);
}
int main() { rectangles<unsigned int>(); rectangles<float>(); assert(allocations.empty()); }
