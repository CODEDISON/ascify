# Ascify

**Translate CUDA C/C++ source for Ascend.**

English · [简体中文](README.zh-CN.md) · [Documentation](docs/README.md) · [Contributing](CONTRIBUTING.md)

Ascify uses Clang to analyze CUDA source and rewrite supported APIs and kernel
constructs to Ascify compatibility headers and Ascend ACL interfaces. The
`ascify-clang` executable produces source that you compile, link, and validate
with your Ascend target toolchain.

```text
CUDA source  →  ascify-clang  →  Ascend-compatible source  →  CANN build and device tests
```

## What it supports

- **Source migration:** CUDA runtime and device constructs within the documented
  compatibility surface, with conversion statistics and optional JSON receipts.
- **Local headers:** Optional recursive conversion with checked dependency
  provenance and transactional output publication.
- **Ascend execution paths:** A general SIMT compatibility path and an explicit
  SIMD+SIMT Hybrid path for recognized row-wise Softmax, RMSNorm, and LayerNorm.
- **Conservative boundaries:** Unsupported patterns retain their original source
  or fail explicitly, according to the relevant conversion contract.

See the [compatibility reference](docs/compatibility.md) for supported CUDA APIs and device constructs.

## CUDA Samples on Ascend950DT

On 24 November 2026 (UTC), one Ascify installation translated **38/38** unmodified
CUDA Samples v13.0 vector samples (20 compute, 18 runtime). **31/38** compiled and
linked with CANN 9.1.0. **29/38** met their recorded criteria in four consecutive
device runs, including the `clock` demonstration. Five of those 29 used an
adjusted build or link step, and two required a project-specific Driver adapter.

| v13.0 result | CUDA Samples | Build or device check |
|---|---|---|
| 4/4 · compute | `binaryPartitionCG`, `fp16ScalarProduct`, `scalarProd`, `simpleTemplates`, `simpleVoteIntrinsics`, `template`, `threadFenceReduction`, `vectorAdd` | Compiled and linked the translated sample. |
| 4/4 · runtime | `graphMemoryNodes`, `simpleCallback`, `simpleCudaGraphs`, `simpleMultiCopy`, `simpleMultiGPU`, `simpleP2P`, `simpleStreams`, `simpleZeroCopy`, `streamOrderedAllocation` | Compiled and linked the translated sample. Graph replay was sequential. |
| 4/4 · runtime | `asyncAPI`, `cudaOpenMP`, `simpleMPI` | `asyncAPI` used the profiling library and a writable `ASCIFY_PROFILE_DIR`. OpenMP used one visible device. MPI used two processes on one host. |
| 4/4 · standalone build | `shfl_scan` | Used `--standalone-device-executable` for the scan and integral-image checks. |
| 4/4 · demonstration | `clock` | The program ran and reported zero cycles. It has no numerical assertion. |
| 4/4 · adjusted build | `scan`, `sortingNetworks` | Compiled the translated files together as one translation unit. |
| 4/4 · adjusted link | `histogram`, `mergeSort`, `reduction` | Changed how host and device objects were linked into shared libraries. |
| 4/4 · Driver adapter | `simpleDrvRuntime`, `vectorAddDrv` | Required project-specific typed kernel-launch code and a forwarding header. |
| Failed on device | `cdpSimpleQuicksort`, `warpAggregatedAtomicsCG` | The sort produced an out-of-order pair. The atomic sample reported an out-of-bounds device read (355). |
| Failed to build | `cudaCompressibleMemory`, `reductionMultiBlockCG`, `simpleOccupancy` | A kernel pointer in a cooperative launch or occupancy query was rewritten to a Graph adapter symbol. |
| Failed to build | `simpleAWBarrier` | Pinning the architecture macro caused overlapping rewrites. Other profiles emitted code that lacked a sample helper. |
| Failed to build | `simpleCooperativeGroups`, `streamOrderedAllocationP2P` | A device `printf` became an undeclared stub in the first sample. A host-lambda `printf` became a device stub in the second. |
| Failed to build | `vectorAddMMAP` | The target compiler reported repeated default arguments and pointer arithmetic on `void*`. |

The tested v13.0 CUDA Samples revision was `3f1c5096`. The installed
`ascify-clang` SHA-256 was
`b24012d79874b79bd3d15821ffac0139ad6fd65a09e9dccc2fe73571dd91d0de`.
Outside this 38-sample set, both CUDA source files for `cdpAdvancedQuicksort`
translated, and the target compiler rejected an inline type.

| CUDA Samples v13.4 (revision `5443602d`) | Result on the same installation |
|---|---|
| `localityDomains` | Built and ran four times. The stream and allocation both reported domain 0. The program reported 64 SMs per domain and a split resource with `smCount = 0`. |
| `localityDomainsDrv` | Host C++ translation succeeded. Undeclared Driver types and a missing compatibility namespace stopped target compilation. |

## Quick start

You need CMake 3.16.8+, a C++17 compiler, Ninja, matching LLVM/Clang development
files, and a CUDA Toolkit directory for parsing. Conversion does not require a
GPU or NPU. Python 3.9+ is required for repository checks and test tooling.

```bash
git clone https://github.com/CODEDISON/ascify.git
cd ascify

export LLVM_BUILD_DIR=/path/to/llvm
./build.sh
cmake --install build
```

`LLVM_BUILD_DIR` may be an LLVM development installation or build tree containing
both `LLVMConfig.cmake` and `ClangConfig.cmake`. For a checkout, you can instead set
`LLVM_PROJECT_PATH=/path/to/llvm-project` to use its `build/` directory.

Convert the included FP32 vector-add example:

```bash
mkdir -p .work/examples

ascify_install/bin/ascify-clang examples/vector_add.cu \
  --cuda-path=/path/to/cuda \
  --target-policy=dav-c310-vec \
  --migration-receipt="$PWD/.work/examples/vector_add.receipt.json" \
  -o "$PWD/.work/examples/vector_add.cu.dpp" \
  -- -std=c++17
```

The installed executable locates its private Clang parsing resources in
`libexec/ascify/clang/<major>/`. Custom layouts can pass
`--clang-resource-directory=/path/to/clang/resource`. Internal
`CUDA2DPP` filenames and the `.dpp` output suffix are retained Ascend compatibility
names. The [English guide](docs/user-guide.en.md) and
[中文使用手册](docs/user-guide.zh-CN.md) cover dependency setup, result checks,
target compilation, and troubleshooting.

## Conversion modes

| Mode | Selection | Scope |
|---|---|---|
| Conservative defaults | `--target-policy=portable --simt-math=precise` | Default source conversion policy |
| Target SIMT | `--target-policy=dav-c310-vec` | Target-specific compatibility and guarded rewrites |
| SIMD+SIMT Hybrid | Add `--simt-math=fast --target-recipe=dav-3510-rowwise-simd-v1` to target SIMT | Explicit, proved row-wise recipes with whole-operator SIMT fallback on selector misses |

Hybrid output needs the separately built runtime in
`runtime/dav_3510/rowwise/`. Start with the
[row-wise conversion guide](docs/rowwise-simd-conversion.md) for supported
shapes, ABI requirements, and target validation.

On 950PR, a rebuild of OneFlow FP16 targets from revision `610b517` reached
81.53% of the frozen A800 baseline for RMSNorm plain and 80.86% for affine,
both above the original 80% target. Softmax passed its correctness suite
and measured 44.40% and 45.13% in two full runs, down from 92.17% in
September. With the November translator `e199603`, both operators stopped
at kernel occupancy analysis.
See the [950PR performance and conversion results](docs/validation-matrix.md#oneflow-fp16-hybrid-on-950pr).

## Documentation

| Task | Guide |
|---|---|
| Install and convert your first source | [English](docs/user-guide.en.md) · [中文](docs/user-guide.zh-CN.md) |
| Look up CLI options and conversion boundaries | [Source conversion reference](docs/conversion-reference.md) |
| Understand the implementation and supported APIs | [Architecture](docs/architecture.md) · [Compatibility](docs/compatibility.md) |
| Add a capability or fix a bug | [Contributing](CONTRIBUTING.md) |
| Prepare a release | [Release process](docs/release-process.md) |

## Development

```bash
python3 tools/check_repository.py --root .
ASCIFY_BINARY= sh tests/run_release_checks.sh
```

The host suite checks repository and semantic contracts. Native build, installed
CLI, actual CUDA conversion, and device checks have separate scopes; see
[Contributing](CONTRIBUTING.md) for the commands and required evidence.

| Directory | Responsibility |
|---|---|
| `src/` | Clang analysis, rewrite rules, CLI, and statistics |
| `include/ascify/`, `acl_cub/` | Public compatibility headers consumed by generated Ascend source |
| `frontend_compat/` | Versioned profiles used when parsing CUDA input |
| `runtime/` | Device implementations linked by generated Hybrid code |
| `tests/`, `examples/` | Fixed fixtures, checks, and CUDA examples |
| `docs/`, `tools/` | Documentation and development tools |

For example, a converted `cudaMalloc` call uses `ascify::cudaMalloc` from
`<ascify/ascify_cuda_compat.hpp>`. The `ascify/` prefix names the public header
namespace; compile generated code with `-I<install-prefix>/include`. The
translator implementation itself lives in `src/`. Its private Clang parsing
headers are installed under `libexec/ascify/clang/<major>/include`, separate from
the public compatibility headers.

## License and attribution

See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Individual file notices retain their terms, including inherited MIT-licensed
translator code and third-party CUDA fixtures. Copyright and source attribution
are preserved in the repository and installed notices.
