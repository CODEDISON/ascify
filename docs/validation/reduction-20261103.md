# Original CUDA Samples reduction

The complete original CUDA Samples v13.0 `reduction` target passed conversion,
CANN O2 compilation, linking and NPU correctness checks with source
`0806abf9be7565f7c2c79766329ab2d38911fcf7`. Both original translation units,
all14 source files, all10 kernel families and int/float/double were retained.
The generated device image contains213 entries,71 per dtype.

All45 original CLI configurations passed. Each kept16,777,216 input elements,
the original warmup and100 benchmark iterations. Coverage includes all10
families for each dtype, kernel9 with64/128/512/1024 requested threads, and
original multi-stage GPU-final reduction for all3dtypes. Actual source-selected
thread/block counts are recorded, including runtime grid-limit adjustment.
This does not exercise every retained runtime instance or all possible inputs.

Each configuration passed the original `Test passed` predicate and an independent
oracle over its actual full input. The input observer sealed both original H2D
copies: the full input and the output-initialization prefix. Every expected D2H
record was checked with process/directory identity, framing, checksums and
complete-population seals. The oracle checks source-owned block populations,
exact integer sums or `math.fsum` with rational rounding bounds, and all-lane
integer/finite bounds. It assumes no random-number sequence and does not call
the target implementation. Original tolerances were preserved.

The root review verified7237 raw files. Six numerically successful attempts
with external occupancy were retained and excluded; affected configurations
were rerun as complete processes. Four busy preflights made zero program calls.
Device health and idle checks are snapshots. The newer runner checks the
immediate process-exit epoch before offline CPU verification and retains a
later diagnostic idle snapshot; final health is still required. No continuous
exclusive-device or performance claim follows.

The same translator passed a fresh native O3 build/install, CTest4/4,307 logical
baseline cases and the additional source/semantic suites. Prior binary64
shared-pointer ABI probes were reused only after an explicit no-ABI-change
resource review; they were not rerun on this binary. Upstream NVCC CMake and
CUDA-device reference execution were not reproduced.

The [masked accumulation decision](../decisions/003-masked-shuffle-accumulation.md)
describes completion of source-undefined inactive-lane reads while preserving
defined neighbor/self operations. Signed-overflow inputs remain outside the
defined source domain. The failed predecessor and incomplete first observer
attempt were retained; neither was counted or silently retried unchanged.

The historical strict-v13.0 vector union is now16/38, and the core union31/75.
Only `reduction` has this binary's complete original NPU acceptance; the other
historical names retain their own product evidence. This publication merges
the tested code with earlier public documentation and adds this record. The
publication commit does not represent another NPU run.

[Structured results, fingerprints and all45 configurations](reduction-20261103.json)
provide the exact tested scope.
