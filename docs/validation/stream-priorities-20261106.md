# Original StreamPriorities validation

The unchanged CUDA Samples v13.0 `StreamPriorities` passed ordinary conversion,
a complete one-object O2 target build, and one execution on Ascend950DT with
CANN 9.1.0. The original single CU, RDC, two named nonblocking priority streams,
four events, four 1024-block × 512-thread launches, and all four host and four
device 256 MiB buffers were preserved.

Both original full-buffer `memcmp` checks passed. A passive observer captured
the two actual synchronous D2H outputs before returning to the caller. The
independent oracle checked every value: **134,217,728 signed int32 values**,
536,870,912 bytes in total. Each complete output equals `int32[i]`. The observer
added no ACL calls, synchronization, or kernel tracing hooks. Actual stream
priorities were 7 and 0; both original event timings completed. No performance
ordering or preemption result is claimed.

Validated source: `271c4b5abce37214027a0bb050b93a877378ee69`.
Translator SHA-256: `8d9537f115ad2d06eaeb9cff2e5b4fb7b541810dcb4a155fa41c8de08a0e7b3b`.
The [acceptance receipt](stream-priorities-20261106.json) records source, target,
process, numerical, and evidence identities.

The translator's 42 O3 compilation instances, four CTests, 307 existing software
cases, supplemental legacy pair, and 30 additional dynamic/ABI checks passed.
The new priority wrappers also passed nine CPU stages and a real native empty
stream test that read back priorities 0 and 7. The observer passed 17 CPU stages
including error, wrong extent, and incomplete capture cases. CPU fixtures do
not supply the original device result.

The original CMake file was preserved, including its aarch64 skip. Validation
used the fixed complete NPU target adapter; it does not claim that upstream
CMake built this sample. This acceptance raises the historical core union to
35/75 and the strict v13.0 union to 42. `StreamPriorities` is outside the original
38-name vector scope, which remains 19/38. Other samples retain their own
product evidence. Complete builds, objects, raw logs, and captures remain on
the validation host. See the [priority contract](../stream-priorities.md).
