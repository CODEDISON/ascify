#define __aicore__
#define __global__
#include <ascify/ascify_cuda_compat.hpp>
int main() { int value=0; return ascify::atomic_add_generic(&value,1); }
