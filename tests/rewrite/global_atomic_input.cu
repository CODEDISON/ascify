#include <cuda.h>

__device__ int global_atomic_state;

__global__ void GlobalAtomicCoverage(
    int* signed_values,
    unsigned int* unsigned_values,
    int* old_values,
    int index) {
  old_values[0] = atomicAdd(&signed_values[index], 1);
  old_values[1] = atomicSub(signed_values + index, 2);
  old_values[2] = atomicExch(&signed_values[index], 3);
  old_values[3] = atomicMax(signed_values + index, 4);
  old_values[4] = atomicMin(&signed_values[index], 5);
  old_values[5] = atomicCAS(signed_values + index, 5, 6);
  old_values[6] = atomicAnd(&signed_values[index], 7);
  old_values[7] = atomicOr(signed_values + index, 8);
  old_values[8] = atomicXor(&signed_values[index], 9);
  old_values[9] = atomicInc(unsigned_values + index, 10U);
  old_values[10] = atomicDec(&unsigned_values[index], 10U);
}

__device__ int GenericHelperAtomic(int* unknown_address) {
  return atomicAdd(unknown_address, 1);
}

#define ASCIFY_TEST_ATOMIC_MACRO(address) atomicAdd((address), 1)

__global__ void RejectedAtomicProvenance(
    int* global_values,
    float* float_values,
    int index) {
  __shared__ int shared_value;
  int local_value = 0;
  int* local_alias = global_values;
  atomicAdd(&shared_value, 1);
  atomicAdd(&local_value, 1);
  atomicAdd(local_alias + index, 1);
  atomicAdd(float_values + index, 1.0f);
  atomicAdd(&global_atomic_state, 1);
  ASCIFY_TEST_ATOMIC_MACRO(global_values + index);
}
