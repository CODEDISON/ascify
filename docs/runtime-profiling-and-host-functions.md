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
direct unconditional literal include. Conditional includes, header-guard
observations, retained header inquiries, token pastes and macro-stack pragmas
are refused. In particular, a retained `__has_include(cuda_profiler_api.h)` must
not silently disable the target profiling region. The v13.3 original matrixMul
uses an unconditional include; the v13.4 conditional form is not admitted yet.

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
