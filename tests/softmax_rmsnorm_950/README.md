# Softmax / RMSNorm test harness

> Scope: `run_910_conversion_v3.sh` exercises the earlier in-header
> `TrySoftmax` / `TryRmsNorm` recipe and does not pass the explicit
> `--target-recipe=dav-3510-rowwise-simd-v1` option or build the four v1 mixed
> runtime DSOs. For the externally
> linked SIMD+SIMT path, follow
> [the explicit conversion guide](../../docs/rowwise-simd-conversion.md), then
> use this directory's build/smoke harness with
> `ROWWISE_SIMD_RUNTIME_DIR` set to the complete runtime `lib` directory.

This harness validates forward FP16 Softmax and RMSNorm (plain and affine).
`layernorm_hybrid_check.cce` additionally validates the registered LayerNorm
route with fixed boundary, tail, cross-AIV, in-place, invalid-domain, and
profile cases. It excludes LogSoftmax, backward kernels, and A800 comparison.

`direct` means an unedited Ascify-generated header whose proved wrapper calls
the versioned `ascify::target::dav_c310::rowwise_simd_v1` implementation. `native` is a thin
control entry into the same implementation. A passing comparison shows that a
clean conversion reproduces the selected target recipe; it does not claim
that Ascify synthesizes the low-level kernel from arbitrary CUDA.

## Prerequisites and output paths

Run the examples from the repository root. Conversion requires a built
Ascify executable, LLVM/Clang resource headers, and CUDA parsing headers.
The replay scripts use Bash, Python 3.9 or newer, `sha256sum`, and `flock`.
Device checks additionally require a compatible CANN package, its CCEC compiler,
`npu-smi`, and an Ascend device supported by the selected recipe. The legacy
formal validator specifically expects `Ascend950PR` in the device snapshot. See the
[build guide](../../docs/user-guide.en.md) for translator dependencies.

Choose a writable output directory. The following uses the scripts' ignored
default; `WORK_ROOT` can also point outside the checkout:

```bash
export REPO_ROOT="$(pwd -P)"
export WORK_ROOT="${REPO_ROOT}/.work/softmax_rmsnorm_950"
```

The harness writes generated headers, build outputs, locks, manifests, CSV
results, and logs below `WORK_ROOT`. Use the same work root when staging and
running one experiment; conversion and device hosts may use different absolute
paths. Set `CANN_ROOT`, `LLVM_BUILD_DIR`, and `CUDA_ROOT` to your own installations.

The three OneFlow conversion inputs are versioned in
`tests/fixtures/oneflow/`; their origin, local RMSNorm patch, and SHA256 values
are recorded in [the fixture manifest](../fixtures/oneflow/README.md).

## Host-only gate

```bash
ASCIFY_BINARY= ASCIFY_CUDA_PATH= ASCIFY_CLANG_RESOURCE_DIRECTORY= \
  sh tests/run_release_checks.sh
```

This runs the static rewrite contracts and registered Python suites. To also
re-translate the golden fixtures, provide the translator and parsing dependencies:

```bash
ASCIFY_BINARY=/path/to/ascify-clang \
ASCIFY_CUDA_PATH=/path/to/cuda \
ASCIFY_CLANG_RESOURCE_DIRECTORY=/path/to/install/libexec/ascify/clang/23 \
sh tests/run_release_checks.sh
```

## Build and convert

The legacy replay script retains its `run_910_conversion_v3.sh` filename, but
conversion runs on the host where Ascify and its dependencies are installed.
For an LLVM 23 build with the default install directory layout:

```bash
export LLVM_BUILD_DIR=/path/to/llvm-prefix
export CUDA_ROOT=/path/to/cuda
export BUILD_DIR="${WORK_ROOT}/build"
export INSTALL_ROOT="${WORK_ROOT}/install"

./build.sh
cmake --install "${BUILD_DIR}"

export ASCIFY_BINARY="${INSTALL_ROOT}/bin/ascify-clang"
export CLANG_RESOURCE_DIRECTORY="${INSTALL_ROOT}/libexec/ascify/clang/23"
```

The resource root contains `include/__clang_cuda_runtime_wrapper.h`; adjust
`23` and `libexec` to the resource version and install layout of your build.
Pass it explicitly because the legacy replay script predates this layout.

Create a new evidence set. The script refuses to overwrite an existing set:

```bash
WORK_ROOT="${WORK_ROOT}" \
ASCIFY_BINARY="${ASCIFY_BINARY}" \
CUDA_ROOT="${CUDA_ROOT}" \
CLANG_RESOURCE_DIRECTORY="${CLANG_RESOURCE_DIRECTORY}" \
tests/softmax_rmsnorm_950/scripts/run_910_conversion_v3.sh
```

The command fixes:

```text
--target-policy=dav-c310-vec
--simt-math=fast
-Iinputs
-std=c++17
```

It packages the exact converter binary, recipe source/header, four inputs,
four outputs, argv, logs, and hashes. The required generated topology is:

| Unit | target include | load marker | store marker | `TrySoftmax` | `TryRmsNorm` |
|---|---:|---:|---:|---:|---:|
| `softmax` | 1 | 1 | 1 | 3 | 0 |
| `layer_norm` | 0 | 1 | 1 | 0 | 0 |
| `rmsnorm` | 1 | 0 | 0 | 0 | 3 |
| `rmsnorm_adapter` | 0 | 0 | 1 | 0 | 0 |

Softmax calls must be in the Warp, BlockSMem, and BlockUncached direct launch
wrappers. RMSNorm calls must be after the complete `GetNumBlocks` geometry and
error-return block and before the unique launch. Dispatchers, LogSoftmax, and
all other locations must contain no recipe call.

Run the fail-close mutation matrix against the same commit and inputs:

```bash
python3 -B tests/rewrite/check_dav_c310_rowwise_mutations.py \
  --ascify "${ASCIFY_BINARY}" \
  --softmax tests/fixtures/oneflow/oneflow/core/cuda/softmax.cuh \
  --rms-norm tests/fixtures/oneflow/oneflow/core/cuda/rms_norm.cuh \
  --layer-norm tests/fixtures/oneflow/oneflow/core/cuda/layer_norm.cuh \
  --recipe-source src/DavC310TargetRecipe.cpp \
  --recipe-header src/DavC310TargetRecipe.h \
  --cuda-path "${CUDA_ROOT}" \
  --clang-resource-directory "${CLANG_RESOURCE_DIRECTORY}" \
  --include-dir tests/fixtures/oneflow \
  --work-dir "${WORK_ROOT}/recipe_mutations" \
  --converter-cwd "${REPO_ROOT}"
```

The mutation matrix contains 29 Softmax, 9 RMSNorm, and 7 LayerNorm cases.
A successful run reports 45/45 passed.

## Use the conversion bundle on a device host

If conversion and device execution use separate hosts, copy the complete
`conversion/evidence_v3` directory to `${WORK_ROOT}/conversion/evidence_v3`
on the device host using your preferred file-transfer method. Preserve its
manifest, relative paths, and file bytes. Use the same source commit to build
the target checks; the checkout and work-root paths can differ between hosts.
The generated headers are consumed without manual edits.

## Stage, build, verify, and benchmark on the device host

Set the work root containing the conversion bundle and your CANN package:

```bash
export REPO_ROOT="$(pwd -P)"
export WORK_ROOT="${REPO_ROOT}/.work/softmax_rmsnorm_950"
export CANN_ROOT=/path/to/cann

mkdir -p "${WORK_ROOT}/generated"
cp -a "${WORK_ROOT}/conversion/evidence_v3/outputs/." \
  "${WORK_ROOT}/generated/"

export ASCIFY_BINARY_SHA256="$(
  sha256sum "${WORK_ROOT}/conversion/evidence_v3/converter/ascify-clang" |
  awk '{print $1}'
)"

FORMAL_TAG=repro_v3 \
ASCIFY_BINARY_SHA256="${ASCIFY_BINARY_SHA256}" \
CANN_ROOT="${CANN_ROOT}" \
WORK_ROOT="${WORK_ROOT}" \
tests/softmax_rmsnorm_950/scripts/run_formal_recipe_v3.sh
```

The formal runner selects its device automatically after building, ignoring
a caller-provided `DEVICE`. It selects and locks a healthy
device with zero compute and HBM-bandwidth utilization. If a candidate becomes
busy between the initial check and the check made under its project lock, the
selector rejects that candidate and continues with the next discovered device.

The replay script uses:

- direct and native `correctness.csv`: 42 Softmax and 18 RMSNorm cases each;
- direct and native `unified_tune.csv`: 5 Softmax and 10 RMSNorm cases each;
- `WARMUP=20`, `SAMPLES=50`, `INNER_REPEATS=20`;
- direct-A, native, direct-B ordering with no interleaved build or correctness;
- a bounded process-cleanup poll between phases, followed by the unchanged
  strict idle pre-snapshot;
- direct A/B spread at most `1.05`;
- every shape direct/native geometric center at least `0.90`;
- Softmax, RMSNorm plain, and RMSNorm affine group geomean at least `0.95`.

The binary bundle and run manifests are written to
`${WORK_ROOT}/results/manifests/`.

Derive ordinary arithmetic throughput and separate SFU call rates:

```bash
python3 -B tests/softmax_rmsnorm_950/tools/derive_work_metrics.py \
  "${WORK_ROOT}/results/perf_history.csv" \
  --shape-manifest tests/softmax_rmsnorm_950/shapes/unified_tune.csv \
  --output "${WORK_ROOT}/results/perf_metrics_v1.csv"
```

The calculation does not count `exp` or `rsqrt` as FLOPs. See
[probes/README.md](probes/README.md) for independent hardware measurements.

Build and run only the independent LayerNorm hybrid check when converted
Softmax/RMSNorm headers are not being staged:

```bash
export CANN_ROOT=/path/to/cann
export ROWWISE_SIMD_RUNTIME_DIR=/path/to/rowwise-simd-v1/lib

tests/softmax_rmsnorm_950/scripts/build.sh layernorm-check production-fast
LAYERNORM_CHECK_PATHS=direct \
  tests/softmax_rmsnorm_950/scripts/run_layernorm_hybrid_checks.sh
```

The program checks five valid domains, including exact in-place input/output,
requires a non-divisible column count to be rejected before launch, and prints
ACL-event median/minimum/p90 latency for a 256 x 4096 profile case.

To validate the complete generated-code route instead of calling only the
runtime ABI, stage the Ascify-generated header under its preserved include
topology and build the same oracle with the generated-wrapper entry enabled:

```bash
mkdir -p "${WORK_ROOT}/generated/oneflow/core/cuda"
cp /path/to/ascify-output/layer_norm.cuh \
  "${WORK_ROOT}/generated/oneflow/core/cuda/layer_norm.cuh"

GENERATED_ROOT="${WORK_ROOT}/generated" \
  tests/softmax_rmsnorm_950/scripts/build.sh \
  layernorm-generated-check production-fast

tests/softmax_rmsnorm_950/scripts/run_layernorm_hybrid_checks.sh
```

This second binary instantiates the converted `DirectLoad`/`DirectStore` and
ordinary top-level `DispatchLayerNorm`; the injected generic facade then
selects the same versioned runtime from the proved warp branch. It checks five
representative generated-path domains through 1024 columns. The companion
direct-ABI binary additionally covers the runtime's 4096- and 8192-column
boundaries. The runner acquires one healthy-device lock and checks both paths
sequentially on that device. See
[the explicit conversion guide](../../docs/rowwise-simd-conversion.md).
