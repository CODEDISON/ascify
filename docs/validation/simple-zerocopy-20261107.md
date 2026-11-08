# simpleZeroCopy — strict CUDA Samples v13.0 original acceptance (2026-11-07)

## Scope

- Original `Samples/0_Introduction/simpleZeroCopy` (single `.cu`, default argv:
  `cudaHostAlloc` mapped path, 1048576 floats) converted and validated unchanged.
- Original predicate preserved: `exit(errorNorm/refNorm < 1.e-6f ? EXIT_SUCCESS : EXIT_FAILURE)`.

## Native contracts added

- `cudaHostAlloc(Default/Portable/Mapped)`, `cudaHostRegister/-Unregister`,
  `cudaHostGetDevicePointer`, `cudaSetDeviceFlags` over the native
  `aclrtHostRegister`/`aclrtHostGetDevicePointer` mapping (identity mapping,
  verified on device); `cudaFreeHost` unregisters ledger ranges first.
  WriteCombined and unknown bits fail closed.
- `cudaDeviceProp.canMapHostMemory` reports the real native capability.
- NVIDIA helper `checkCudaCapabilities(major, minor)` calls are proven against
  the frozen helper profile and rewritten to `::ascify::sampleCheckCudaCapabilities`,
  which compares the reviewed capability-epoch surface from
  `cudaGetDeviceProperties` (never a fabricated SM readout) and keeps the
  original informational printf shape.

## Evidence (lean; raw artifacts stay on the validation host)

- Fresh native build: Release/O3, CTest 4/4, host release checks rc0.
- Converter binary `6e40fa6a49d119001e55a8006a3664c6a7ff28af77df1b70bc2d9049ef05891c`;
  helper transaction: include removed, check_rewrites=19, get_last_rewrites=1.
- Owned O2 single-TU target (compile/link rc0; `-DCUDART_VERSION=13040`
  faithfully restores the SDK 13.4 preprocessing for the retained main-file
  `CUDART_VERSION` gates); program `963a49ea72f3c1597adb7c6f997f76728e82381dd9b4c902d471be7ec2c24b`-class identity recorded in the acceptance receipt.
- Original sample run on Ascend 950DT: rc0, original predicate passed, all four
  original stdout lines printed, single run, no retries, device healthy and
  idle afterwards. The kernel dereferences the mapped device pointers
  (true zero-copy); no explicit staging copies.

Lean acceptance receipt:
`projects/ascify_generalization60_20261101/zerocopy_original_v1/root_original_acceptance_v1.json`.
