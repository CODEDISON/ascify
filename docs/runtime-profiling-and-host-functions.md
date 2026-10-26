# Runtime collection and host functions

`cudaProfilerStart` and `cudaProfilerStop` migrate to an explicitly owned CANN
collection session. Set `ASCIFY_PROFILE_DIR` to an existing writable directory.
Without this opt-in, both return `ACL_ERROR_FEATURE_UNSUPPORTED`; they do not
pretend to control an external profiling tool. The admitted use is a dedicated
single-context process, without another ACL profiler. CANN collection is
process-wide, whereas CUDA controls an attached tool for the current context.

Start obtains the actual current context and logical device and calls
`aclprofInit`, `aclprofCreateConfig` and `aclprofStart`. Collection includes ACL
API calls and task timing. Stop calls `aclprofStop`, `aclprofDestroyConfig` and
`aclprofFinalize`. There is no invented device ID, synchronous device barrier,
or swallowed explicit-call error. Repeated Start on the active owned session
and repeated Stop on the inactive session are idempotent. A lock serializes
calls; another current context/device is rejected while resources are owned.

A failed Stop/Destroy/Finalize retains the resource and exact failed stage so
Stop can retry. Start failure returns its original error; unsuccessful rollback
remains owned and its error is returned by a following Stop/Start. A null config
has no ACL error-code result and maps to `ACL_ERROR_PROFILING_FAILURE`. An exit
callback registered after successful Start performs best-effort cleanup before
runtime teardown; applications must explicitly Stop to observe flush errors.

The profiler header projection accepts the verified physical CUDA header as a
direct unconditional literal include. On the LLVM23 frontend it additionally
admits one proved positive main-file availability block: a direct literal
`#if __has_include(<cuda_profiler_api.h>)`, optionally preceded by the exact
`defined(__has_include) &&` test, followed by the direct include, optionally one
fresh caller flag defined to literal `1`, and the matching `#endif`. The block
must be top-level, without an else, nested condition, extra include or code.
The actual lookup entry, active physical SDK read, builtin history and complete
source ranges must agree. The flag retains its source name and definition;
subsequent uses are limited to direct `ifdef`/`ifndef` or `if defined` controls.

For an explicit Host C++ original root, the frontend's own unmodified main-file
mapping may carry a dedicated witness for the canonical path, physical UID/size,
original bytes and SHA256. These identities are rechecked around mapping, at
the condition, at proof sealing and before publishing the rewritten buffer.
Only the root may use this witness; selected headers, contextual/joint inputs,
caller remaps, VFS overlays, PCH and modules do not acquire it. SDK lookup,
entry and content checks retain their physical identity requirements.
The witness is optional for ordinary Host C++ reads: failed capture keeps the
original read path and leaves the conditional-profiler exception unavailable.

That proved query becomes `__has_include(<acl/acl_prof.h>)`, with an explicit
compile-time error alternative if the target header is absent. A transactional
`#line` restores the valid original next-line position after the block; caller
line directives/markers and explicit position-macro observations remain outside
this admission. Header-guard reflection, other retained header inquiries, token
pastes, macro-stack controls, shadow headers and unproved source contexts are
still refused. No fake header or capability macro is introduced, and the source
profiling region must not silently disappear when CCEC does not search CUDA.

A successful source-conversion receipt may include `cuda_profiler_availability`
with rule `positive-physical-sdk-probe-v1`, source/SDK/range evidence and
`backend_status: unverified`. This is not a backend or runtime success claim.
Complete build acceptance must also prove the actual CCEC-selected canonical
CANN profiler-header dependencies and the real `msprofiler` provider, with
unchanged input identities. See the [frontend boundary checks](../tests/rewrite/check_conditional_profiler.py).

The three-argument typed and `void**` `cudaMallocHost` overloads support zero
flags through real `aclrtMallocHost`. Nonzero Portable/Mapped/WriteCombined
flags return unsupported without allocating. Both preserve the existing null
output/error contract and free through `aclrtFreeHost`.

`cudaLaunchHostFunc` queues the same function and user-data pointer through
`aclrtLaunchHostFunc` on an explicit stream, for the reviewed legacy SIMT /
ACL 1.17.0 family. The runtime owns the callback thread. Earlier stream work
completes before the callback, and later work waits for it. Callbacks must not
call CUDA/CANN runtime APIs. Null functions, default streams and unreviewed
header families are rejected. Linux's physically verified empty `CUDART_CB`
macro is removed only at a direct main-file expansion; user definitions,
nested macro spellings and nonempty calling conventions are retained.

Direct physically owned `cudaDeviceGetAttribute` requests for CUDA compute
capability major/minor use nonnumeric unsupported-selector tags. They return
`ACL_ERROR_FEATURE_UNSUPPORTED` without changing the output; null outputs and
negative devices return `ACL_ERROR_RT_PARAM_INVALID`. Enum arithmetic, `sizeof`,
stored selectors, templates and macro selector uses keep their CUDA spelling.
No fabricated CUDA capability number is published. Display-only adaptation is
a separate source proof and does not turn these unsupported queries into success.

For the reviewed legacy / ACL 1.17.0 family, `cudaEventCreate` uses real
`aclrtCreateEventExWithFlag` with timeline, synchronization and stream-progress
flags. This matches CUDA's initially complete event and rerecord behavior.
`cudaEventQuery` maps the actual complete/not-ready state to success or
`ACL_ERROR_RT_EVENT_NOT_COMPLETE` (`cudaErrorNotReady`); query errors propagate.
An unknown state or unreviewed header family returns unsupported. The state
query adds no synchronization barrier and maintains no shadow completion state.

Programs that use the profiler helpers must link the real CANN `msprofiler`
library, which exports the `aclprof*` control APIs in the reviewed installation.

Evidence: CUDA 13.4 `cuda_runtime.h`, `driver_types.h` and `cuda_profiler_api.h`;
CANN 9.1 physical `acl/acl_prof.h` and `acl/acl_rt.h`; the 950DT-supported
official `runtime/0_simple_callback` example; CANN API profiling workflow.
Host tests verify forwarding, failures, ownership, retry and concurrency.
Original CUDA Samples and real collector artifacts are separate device gates.
