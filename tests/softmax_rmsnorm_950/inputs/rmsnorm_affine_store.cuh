/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Conversion input mirroring the forward adapter in
// oneflow/user/kernels/rms_norm_gpu_kernel.cu.  It intentionally contains no
// Ascify target markers; the dav-c310 recipe must prove and add them.
// Linear offsets use 64 bits, matching the standalone A800 benchmark adapter.
#ifndef ASCIFY950_RMSNORM_AFFINE_STORE_INPUT_CUH_
#define ASCIFY950_RMSNORM_AFFINE_STORE_INPUT_CUH_

#include "oneflow/core/cuda/layer_norm.cuh"

#include <cstdint>

namespace ascify950_recipe_fixture {

template<typename SRC, typename DST, bool affine>
struct RmsNormAffineStore {
  RmsNormAffineStore(DST* dst, const DST* weight, int64_t row_size)
      : dst(dst), weight(weight), row_size(row_size) {}

  template<int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    oneflow::cuda::layer_norm::Pack<DST, N> dst_pack;
    oneflow::cuda::layer_norm::Pack<DST, N> weight_pack;
    const int64_t offset = (row * row_size + col) / N;
    const int64_t weight_offset = col / N;
    if (affine) {
      weight_pack.storage =
          *(reinterpret_cast<const oneflow::cuda::layer_norm::PackType<DST, N>*>(
                weight)
            + weight_offset);
    }
#pragma unroll
    for (int index = 0; index < N; ++index) {
      if (affine) {
        dst_pack.elem[index] =
            static_cast<DST>(src[index]) * weight_pack.elem[index];
      } else {
        dst_pack.elem[index] = static_cast<DST>(src[index]);
      }
    }
    *(reinterpret_cast<oneflow::cuda::layer_norm::PackType<DST, N>*>(dst)
      + offset) = dst_pack.storage;
  }

  DST* dst;
  const DST* weight;
  int64_t row_size;
};

}  // namespace ascify950_recipe_fixture

#endif  // ASCIFY950_RMSNORM_AFFINE_STORE_INPUT_CUH_
