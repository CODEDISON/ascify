# simpleStreams — strict v13.0 original acceptance (2026-11-08)

Original `Samples/0_Introduction/simpleStreams` (default argv, nreps=100,
16M-int array) converted and validated unchanged: `correct_data` full-array
exact value 2500 (`c*nreps*niterations`) with exit code 0, all original
stdout sections (device info, memcopy/kernel/non-streamed/4-streams timing)
printed. Single device run, no retries, device healthy and idle afterwards.

## Contracts added

- `cudaEventCreateWithFlags` over `aclrtCreateEvent`
  (Default/BlockingSync/DisableTiming; any other bit fails closed), flag
  constants mapped and admitted.
- `gpuDeviceInit` / `gpuGetMaxGflopsDeviceId` fully ported under the
  documented uniform epoch performance tier (the old GFLOPS forbid pin is
  now a tier pin).
- `cudaEventRecord` on the legacy default stream drains the device first,
  preserving CUDA's implicit barrier against blocking streams — this fixed
  the observed `correct_data` ordering (75/2500 without it).

Lean acceptance receipt:
`projects/ascify_generalization60_20261101/quickwins_v1/simpleStreams/root_original_acceptance_v1.json`.
