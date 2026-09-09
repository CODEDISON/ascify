# Explicit FP16 workload input profile

The `oneflow-fp16-no-device-fp64-v1` profile prepares conversion inputs for
FP16 input/output with FP32 accumulation. It isolates unused device FP64
helper definitions that the target compiler otherwise diagnoses while parsing
the complete OneFlow headers. It does not add FP64 support to Ascify.

The vendored OneFlow fixtures remain unchanged. The preparer validates the
three fixture SHA256 values from `tests/fixtures/oneflow/README.md`, then
creates a separate include root and `profile.json`. The manifest records the
source/profile hashes and every exact edit. Output must be a new directory.

Only six explicit device `double` specializations change:

| Header | Specializations |
|---|---|
| `softmax.cuh` | `Inf<double>`, `Exp<double>`, `Div<double>`, `Log<double>` |
| `layer_norm.cuh` | `Div<double>`, `Rsqrt<double>` |

Their function bodies become same-signature `= delete` declarations. Actual
direct calls, template instantiations, deduced calls, and function-address
uses remain compile-time errors; they cannot silently resolve to the float
specialization. Every other byte is retained, including float definitions,
host-side doubles, epsilon signatures, alignment expressions, and the existing
RMSNorm row-tail fix. This is a workload input selection, not a general filter
for every possible double expression.

From the repository root, explicitly prepare the profile and select it for
the pure SIMT (`dav-fast`) conversion baseline:

```sh
python3 -B tests/softmax_rmsnorm_950/scripts/prepare_fp16_input_profile.py \
  --output-root "$PWD/.work/oneflow-fp16-input-v1"

INPUT_ROOT="$PWD/.work/oneflow-fp16-input-v1" \
CONVERSION_SET_ID=ascify_fp16_no_device_fp64_v1 \
bash tests/softmax_rmsnorm_950/scripts/run_910_conversion_v3.sh
```

Set the usual `ASCIFY_BINARY`, `CUDA_ROOT`, `CLANG_RESOURCE_DIRECTORY`, and
`WORK_ROOT` for the DT installation. Keep `profile.json` alongside the
conversion evidence; the existing conversion runner records the actual
profile input bytes in its input hashes. The default conversion input remains
the original vendored fixture. No generated `.cce`/`.cuh` output is edited.

`run_910_conversion_v3.sh` does not select the mixed SIMD+SIMT target recipe.
For the hybrid workload, pass the same prepared source root to the dedicated
`tools/generate_rowwise_cce.py --mode simd-simt` pipeline. The input profile is
independent of this mode selection: profile preparation or pure SIMT
conversion must not be reported as hybrid compilation or execution.

Host verification:

```sh
python3 -B tests/softmax_rmsnorm_950/scripts/test_prepare_fp16_input_profile.py -v
```

The test compiles the actual prepared math helper regions with a host C++
compiler: float helpers compile and retain their values, while every deleted
double helper and indirect/template use is rejected. It also verifies exact
reversibility of the six edits, unchanged source identities, deterministic
profile generation, and refusal of changed sources or existing output paths.
This does not replace DT Ascify/CCEC validation or final PR correctness and
performance testing. No target compilation result is claimed by the profile.
