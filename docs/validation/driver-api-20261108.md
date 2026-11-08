# vectorAddDrv & simpleDrvRuntime — strict v13.0 original acceptance (2026-11-08)

## Scope

Both original `Samples/0_Introduction` Driver-API samples (default argv,
`Result = PASS` 50000-float 1e-7 predicate, exit code 0) converted and
validated unchanged in a single shared session.

## Driver bridge added (shared by both samples)

- `ascify::cu*` surface over real runtime module loading:
  `cuModuleLoadData` derives the ELF64 image size and really loads through
  `aclrtCreateBinary`/`aclrtBinaryLoad`; `cuModuleGetFunction` cross-checks
  the registry entry against the loaded module via `aclrtBinaryGetFunction`
  with the compiled `__device_stub__` symbol; epoch device attributes; the
  CUDA context is the runtime device binding (`CUctxCreateParams` inert).
- Kernel launch goes through a typed `<<<>>>` thunk registered beside the
  kernel (the proven code-generation path); the runtime handle ABI
  (`KernelArgs*`) expects a different metadata layout and is documented as
  incompatible with these SIMT kernels.
- `helper_cuda_drvapi.h` frozen profile: `__checkCudaErrors` macro shape and
  include transaction; `findCudaDeviceDRV`/`findFatbinPath` fully ported.
- SDK `_v2/_v4` version-macro spellings mapped; preprocessor-skipped branch
  tokens stay dead in the target under the same undefined conditions.
- Synchronous `cuMemcpyHtoD/DtoH` drain the device first, preserving the
  CUDA legacy default-stream ordering (this fixed an observed intermittent
  4-block loss without the drain).
- `sampleFindCudaDevice` generalized: multi-visible environments select
  logical device zero (documented uniform epoch performance tier); zero
  visible devices still fails.

## Evidence (lean; raw artifacts stay on the validation host)

- Fresh native build (Release O3, CTest 4/4, host release checks rc0);
  converter binary `127622e89fba6e31…`.
- Owned merged single-TU target per sample (convert/compile/link rc0);
  `vectorAdd_kernel64.fatbin` is the compiled module image really loaded at
  runtime. Programs `24ffbbe6929e3917…` (vectorAddDrv) and
  `36b076bc79a31156…` (simpleDrvRuntime).
- Single device run each, original predicate and all original stdout lines,
  device healthy and idle afterwards, no retries.

Lean acceptance receipts:
`projects/ascify_generalization60_20261101/drvbridge_original_v1/{vectorAddDrv,simpleDrvRuntime}/root_original_acceptance_v1.json`.
