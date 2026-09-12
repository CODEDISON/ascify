// Native frontend fixture: each variant is a legal CUDA input. Rejected
// variants must retain the entire helper transaction and publish no guard.
#ifndef ASCIFY_TEMPLATE_CASE
#define ASCIFY_TEMPLATE_CASE 0
#endif

#if ASCIFY_TEMPLATE_CASE == 12
static const char* _cudaGetErrorEnum(int) { return "project status"; }
#endif
#include <helper_cuda.h>

#if ASCIFY_TEMPLATE_CASE == 4
namespace project {
struct Value {};
cudaError_t cudaFree(Value*) { return cudaSuccess; }
cudaError_t cudaMemcpy(Value*, Value*, size_t, cudaMemcpyKind) {
  return cudaSuccess;
}
}
#elif ASCIFY_TEMPLATE_CASE == 5
cudaError_t cudaFree(float*) { return cudaSuccess; }
#elif ASCIFY_TEMPLATE_CASE == 6
#define __is_same(...) true
#elif ASCIFY_TEMPLATE_CASE == 7
#define static_assert(...)
#elif ASCIFY_TEMPLATE_CASE == 9
#define ASCIFY_TEMPLATE_BODY_BEGIN {
#endif

#if ASCIFY_TEMPLATE_CASE == 12
template<class T> int project_status(T*) { return 0; }
#endif

// The definition's parameter is T; using the canonical prototype's U when
// rendering the guard would produce an invalid output declaration.
template<class U>
void template_memory(U*, U*, size_t, int, const char**);

template<class T>
void template_memory(T* destination, T* source, size_t bytes,
                     int argc, const char** argv)
#if ASCIFY_TEMPLATE_CASE == 8
#define T int
#endif
#if ASCIFY_TEMPLATE_CASE == 9
ASCIFY_TEMPLATE_BODY_BEGIN
#else
{
#endif
  int device;
  device = findCudaDevice(argc, argv);
  (void)device;
  checkCudaErrors(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaFree(destination));
#if ASCIFY_TEMPLATE_CASE == 12
  checkCudaErrors(project_status(source));
#endif
  getLastCudaError("template helper fixture");
}

#if ASCIFY_TEMPLATE_CASE == 6
#undef __is_same
#elif ASCIFY_TEMPLATE_CASE == 7
#undef static_assert
#elif ASCIFY_TEMPLATE_CASE == 8
#undef T
#elif ASCIFY_TEMPLATE_CASE == 9
#undef ASCIFY_TEMPLATE_BODY_BEGIN
#endif

#if ASCIFY_TEMPLATE_CASE == 10
template<>
void template_memory<float>(float*, float*, size_t, int, const char**) {}
#elif ASCIFY_TEMPLATE_CASE == 11
extern template void template_memory<float>(float*, float*, size_t,
                                           int, const char**);
#endif

void instantiate_memory(int argc, const char** argv) {
#if ASCIFY_TEMPLATE_CASE != 2
  template_memory<float>(nullptr, nullptr, 0, argc, argv);
#if ASCIFY_TEMPLATE_CASE != 1
  template_memory<int>(nullptr, nullptr, 0, argc, argv);
#endif
#endif
#if ASCIFY_TEMPLATE_CASE == 3
  template_memory<char>(nullptr, nullptr, 0, argc, argv);
#elif ASCIFY_TEMPLATE_CASE == 4
  template_memory<project::Value>(nullptr, nullptr, 0, argc, argv);
#endif
}
