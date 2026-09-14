# Third-party notices

The root [LICENSE](LICENSE) contains the Apache License, Version 2.0. Files with
separate license headers retain those terms; the root license does not replace
the MIT or NVIDIA notices reproduced below. Preserve the applicable notices when
redistributing source or installed artifacts.

## Inherited translator source

Ascify's initial repository history identifies its translator skeleton as based
on [HIPIFY](https://github.com/ROCm/HIPIFY) (initial commit `4d84836`). The inherited
translator files in `src/` retain their original Advanced Micro Devices copyright
and MIT permission notices. This is upstream attribution; Ascify's product target
is CUDA-to-Ascend. An exact upstream import revision is not recorded here.

The original file headers are authoritative for each file's copyright and terms.
The retained MIT notice reads:

Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

## LLVM and Clang resource headers

Ascify links against the LLVM/Clang installation selected at build time. When
`ASCIFY_INSTALL_CLANG_HEADERS` is enabled, installation also copies matching
Clang resource headers from that toolchain. Those headers retain their own
upstream notices. Record and distribute the notices applicable to the selected
LLVM/Clang package when packaging its libraries or resource headers.

The complete [Clang license](LICENSES/Clang.txt) and
[LLVM license](LICENSES/LLVM.txt) shipped with the LLVM 23.1.0 development
packages used for the native build check are included here and installed under
`share/licenses/ascify/`. They include the Apache-2.0 terms, LLVM exceptions,
and retained legacy notices. A different toolchain may require additional
notices from that distribution.

## OneFlow CUDA conversion fixtures

The following files are derived from OneFlow
(`https://github.com/Oneflow-Inc/oneflow`) at commit
`25c8978c1c8b1371ef6aa4187dae4495bd233c35`:

- `tests/fixtures/oneflow/oneflow/core/cuda/softmax.cuh`
- `tests/fixtures/oneflow/oneflow/core/cuda/layer_norm.cuh`
- `tests/fixtures/oneflow/oneflow/core/cuda/rms_norm.cuh`
- `tests/softmax_rmsnorm_950/inputs/rmsnorm_affine_store.cuh`

Copyright 2020 The OneFlow Authors. All rights reserved.

These files are licensed under the Apache License, Version 2.0. A copy of the
license is provided in this repository's `LICENSE` file. The original OneFlow
copyright and Apache-2.0 license headers are retained in each fixture.

`rms_norm.cuh` contains the local `rows_per_access` tail-handling patch
documented in `tests/fixtures/oneflow/README.md`.

`rmsnorm_affine_store.cuh` is a reduced conversion fixture derived from the
forward adapter in `oneflow/user/kernels/rms_norm_gpu_kernel.cu`; it contains
only the load/store contract needed by the Ascify semantic matcher.

## NVIDIA CUDA Samples helper fixtures

The following complete fixtures are copied byte-for-byte from NVIDIA CUDA
Samples at commit `b7c5481c556c3fe98db060207ecaa41a4b9a9abc`:

- `tests/rewrite/fixtures/nvidia_samples/Common/helper_cuda.h`
- `tests/rewrite/fixtures/nvidia_samples/Common/exception.h`
- `tests/rewrite/fixtures/nvidia_samples/Common/helper_functions.h`
- `tests/rewrite/fixtures/nvidia_samples/Common/helper_image.h`
- `tests/rewrite/fixtures/nvidia_samples/Common/helper_string.h`
- `tests/rewrite/fixtures/nvidia_samples/Common/helper_timer.h`
- `tests/rewrite/fixtures/mutated_dependency_nvidia_samples/Common/helper_cuda.h`

The following negative-test fixtures are modified or reduced derivatives of
`Common/helper_cuda.h` or `Common/helper_string.h` from that same commit:

- `tests/rewrite/fixtures/altered_nvidia_samples/Common/helper_cuda.h`
- `tests/rewrite/fixtures/altered_find_nvidia_samples/Common/helper_cuda.h`
- `tests/rewrite/fixtures/mutated_find_nvidia_samples/Common/helper_cuda.h`
- `tests/rewrite/fixtures/mutated_dependency_nvidia_samples/Common/helper_string.h`

The nearby `cublas_v2.h` and `user_helpers/helper_cuda.h` files are
project-authored stubs and are not copies of NVIDIA CUDA Samples.

Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of NVIDIA CORPORATION nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
