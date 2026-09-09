# Explicit FP16 input profile

`prepare_fp16_input_profile.py` produces separate, versioned conversion inputs
for FP16 input/output and FP32 accumulation. It verifies the original three
OneFlow fixture hashes before writing a new directory and `profile.json`.
It does not edit the original fixtures or any generated CCE output.

```sh
python3 -B tests/softmax_rmsnorm_950/scripts/prepare_fp16_input_profile.py \
  --input-root tests/fixtures/oneflow \
  --output-root /path/to/new/fp16_inputs
```

Pass this output directory as the converter's input/include root. The RMSNorm
store adapter remains the current, separately converted input in
`tests/softmax_rmsnorm_950/inputs/rmsnorm_affine_store.cuh`.

The current profile is `oneflow-fp16-no-device-fp64-arithmetic-v2`:

| Helper specialization | Profile behavior |
|---|---|
| Softmax `Exp<double>`, `Div<double>`, `Log<double>` | Deleted declaration; actual arithmetic calls fail compilation |
| LayerNorm `Div<double>`, `Rsqrt<double>` | Deleted declaration; actual arithmetic calls fail compilation |
| Softmax `Inf<double>` | Original function returning the `CUDART_INF` positive-infinity constant is retained |
| All float helpers | Unchanged |
| RMSNorm kernel fixture, including its existing tail fix | Unchanged |

The previous v1 deleted six specializations, including `Inf<double>`. That
prevented the existing recipe from proving the infinity helper family:
`provenPositiveInfinityCall` requires both original float and double
positive-infinity bodies. Consequently all three Softmax wrapper proofs were
lost. Retaining the original unused constant-return function restores that
proof without changing the recipe or adding device FP64 arithmetic support.

The v2 boundary is intentionally narrower than v1: five double **arithmetic**
specializations are rejected. A call to `Inf<double>` is no longer rejected by
this profile. Do not claim that all six double calls are deleted, or that this
profile implements FP64 operators. A deleted specialization also prevents
implicit deduction, dependent instantiation or taking its function address
from silently using a float implementation.

```sh
python3 -B tests/softmax_rmsnorm_950/scripts/test_prepare_fp16_input_profile.py -v
```

Host tests exercise the actual prepared helper declarations, preserved infinity
constants, five deleted arithmetic calls, source identity, deterministic output
and refusal to overwrite prior evidence. Full conversion, CCEC compilation and
device correctness/performance are separate validation stages.
