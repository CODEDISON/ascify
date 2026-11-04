# Original binaryPartitionCG validation

The complete CUDA Samples v13.0 `binaryPartitionCG` target passed conversion,
CANN compilation and execution on 950DT with CANN 9.1.0. The original eight
source files, one translation unit, 102400 elements, occupancy query, single
nonblocking stream and algorithm were preserved.

The original occupancy query selected 128 blocks of 1024 threads. One original
program invocation returned zero and printed its final `Done` message. The
actual copied input was checked independently against all three synchronized
outputs:

| Result | Exact expected and observed value |
|---|---:|
| Odd count | 50945 |
| Odd sum | 1272565 |
| Even sum | 1233938 |

A deferred observer retained the original asynchronous transfers and sync. It
captured the 409600-byte input and the 4-byte and 8-byte outputs, reading output
memory only after the original successful stream synchronization. No extra
synchronization or mock provider was used for acceptance. Bound inputs and
execution evidence remained unchanged; the device was healthy and idle before
and after execution.

The converter completes a proved four-byte tail hole in this kernel's launch
ABI with an anonymous default unsigned parameter. The original four argument
expressions and offsets remain unchanged. Actual AArch64 disassembly shows a
48-byte host copy and a zero fifth argument. See the
[bounded ABI contract](../dynamic-kernel-abi-tail.md).

Validated production source: `e26769084285e293179228a237eebcdf55e9bec8`.
The fresh O3 translator, CTest4, existing 307 baseline cases and required CG,
reduction, coordinate and ABI regressions passed on the same translator. The
complete original target was compiled and linked at O2. Native CUDA compilation
also validated the original source for its nine declared architectures; no
NVIDIA execution or performance comparison is claimed.

Artifact hashes and detailed validation boundaries are in the
[machine-readable record](binary-partition-cg-20261103.json). This acceptance
covers this original target; other algorithms and launch shapes require their
own validation.
