# ADR-0010: Lazily own the CUDA-to-ACL runtime lifecycle

- Status: Accepted; clean build, translated fixtures and target regression are release gates
- Date: 2026-08-24
- Scope: CUDA-shaped runtime lifecycle implemented by the installed ACL compatibility layer
- Callback ordering is superseded by [ADR-0024](0024-order-cleanup-after-lazy-device-initialization.md); the ownership and binding contracts below remain in effect.

## Context

Legal CUDA Runtime programs do not call an explicit process initializer. The
runtime initializes itself on the first relevant API, and a program that never
calls `cudaSetDevice` uses logical device zero. NVIDIA's `vectorAdd` sample is
one such program: its first context-dependent operation is `cudaMalloc`.

ACL has a different contract. The CANN 9.1 beta3 headers used by the 950PR
target state that:

- `aclInit` can be called only once in a process;
- `aclrtSetDevice` implicitly creates the default context and default stream;
- selecting the same device repeatedly still requires only one
  `aclrtResetDevice`;
- `aclFinalize` disables subsequent ACL service use.

The previous Ascify map sent device-management calls directly to similarly
named ACL functions and memory shims directly to `aclrtMalloc`/copy/set. That
left a null current context when a CUDA source relied on normal lazy runtime
initialization. It also produced a wrong call shape for CUDA's no-argument
`cudaDeviceReset()` and an `int*`/`uint32_t*` mismatch for
`cudaGetDeviceCount`.

This is a language-runtime contract difference, not a `vectorAdd` or operator
pattern. The rule must also coexist with embedding applications that initialized
ACL themselves, explicit CUDA device selection, early returns, repeated API
calls, multiple translated translation units, and `cudaDeviceReset` followed by
new CUDA work.

## Decision

Keep lifecycle state in the installed, header-only CUDA compatibility layer and
route every currently admitted context-dependent Runtime API through it.

### One process initialization attempt with explicit ownership

An inline C++17 `RuntimeManager` performs `aclInit(nullptr)` on the first
admitted Runtime API. A successful call makes the manager the ACL initialization
owner. `ACL_ERROR_REPEAT_INITIALIZE` means an embedding component owns ACL; it is
treated as a usable borrowed initialization, but Ascify will not call
`aclFinalize` for it. Any other initialization failure is cached and returned,
so later calls do not violate ACL's one-initialization rule by retrying blindly.

The manager is an inline object rather than a function-local static. The
compatibility header is inserted before translated source declarations, so all
translated translation units share one state object. The object's ordinary
destructor is only an idempotent final guard; it is not the primary ACL cleanup
hook. Multiple shared-object copies and unconverted global constructors remain
an integration boundary and are not claimed here.

### Register cleanup after ACL initialization

Constructing the inline manager before `main` also registers its ordinary
destructor before the first `aclInit`. CANN creates its own process-exit state
during initialization. On coherent CANN 9.1 beta3, calling `aclFinalize` from
the older manager destructor therefore ran after CANN teardown and terminated
an otherwise correct program during normal process exit.

After `aclInit` returns success or repeat-initialize, the manager now registers
one dedicated `atexit` callback. LIFO exit order runs this newer callback before
CANN teardown. The callback closes the manager, resets tracked bindings, and
finalizes only an owned ACL initialization; the later ordinary destructor sees
the closed state and performs no ACL operation.

If callback registration fails after an owned initialization, Ascify calls
`aclFinalize` immediately while CANN state is still live. A successful rollback
returns `ACL_ERROR_BAD_ALLOC`; a real rollback error is propagated. A borrowed
initialization is never finalized and registration failure returns
`ACL_ERROR_BAD_ALLOC` before binding a device. The failed result is cached, so
later calls do not repeat initialization or registration.

### Per-thread current-device restoration

Before a context-dependent call, the shim first asks `aclrtGetDevice` for the
calling thread's current device. An existing context is preserved, including
one selected by an embedding application. Only `ACL_ERROR_RT_CONTEXT_NULL` or
`ACL_ERROR_RT_NO_DEVICE` triggers `aclrtSetDevice`.

Each host thread starts with CUDA's logical default device zero. A successful
`cudaSetDevice(d)` updates that thread's selected device. A successful
`cudaDeviceReset()` destroys the active ACL device context but retains the
selected device number, so the next CUDA operation recreates device `d`, not
device zero. Invalid and unrelated ACL failures are propagated rather than
being disguised as a missing context.

`cudaGetDeviceCount` initializes ACL but does not force a device context. This
preserves the useful zero-device query case. Allocation, copy/set, stream,
event, synchronization, current-device, reset, and occupancy shims ensure a
current context before delegating.

### Device cleanup without host-side dynamic allocation

Every device successfully bound by the compatibility layer is recorded once.
The registry is a fixed array of 64 device identifiers protected by an
`atomic_flag` lock. It uses neither C++ containers nor `malloc/new` on CCEC's
host path. This bound is well above the seven-device frozen 950PR environment,
but it remains explicit: a 65th distinct binding is immediately rolled back and
returns `ACL_ERROR_BAD_ALLOC`. If that rollback itself returns an ACL error, the
real rollback error is returned instead of being hidden by Ascify's capacity
status.

This replaces an earlier `malloc`-backed registry. Although that version passed
CCEC syntax and object generation, the third clean `vectorAdd` target run
terminated with `std::bad_alloc` on its first CUDA Runtime path after target
syntax and link had both succeeded. That run did not prove which downstream
component threw, but it did invalidate dynamic allocation in the lifecycle hot
path as a delivery design. The fixed registry removes that dependency rather
than suppressing the exception.

A successful explicit `cudaDeviceReset` removes that device from the registry.
At process teardown, the post-initialization callback resets all remaining
owned device bindings once and then calls `aclFinalize` only if its own
`aclInit` succeeded. Callback repetition and the later object destructor are
idempotent. Exit callbacks cannot report teardown errors, so explicit
`cudaDeviceReset` remains the path when the program needs a returned cleanup
status.

### Error propagation

Initialization, missing-context recovery, set-device, and reset failures are
returned from the CUDA-shaped wrapper that triggered them. A thread-local
pending lifecycle status also participates in `cudaPeekAtLastError` and
`cudaGetLastError`: peek preserves it and get consumes it. This covers failures
that happen before ACL has a valid thread-level last-error state.

Short-circuit contracts established by ADR-0002 remain intact. In particular,
`cudaFree(nullptr)` and valid zero-byte copy/set operations succeed without
initializing ACL or submitting an ACL operation.

## Alternatives considered

### Inject `aclInit`, device zero, reset, and finalize into every `main`

Rejected as the primary rule. It initializes device zero before an explicit
later `cudaSetDevice(d)`, misses libraries and calls made from global
constructors, duplicates work in multi-translation-unit programs, and cannot
coordinate with an embedding application's ACL ownership. It also changes
failure control flow if injected code returns from `main` before the source's
own CUDA error handling.

### Eagerly initialize in a compatibility-header global constructor

Rejected. Merely loading a translated shared object would acquire device zero,
even if no CUDA API is called. Initialization order against an external ACL
owner would also be uncontrolled.

### Finalize only from the inline manager destructor

Rejected after device diagnosis. The destructor is registered before
initialization-time CANN teardown and therefore runs too late under process-exit
LIFO ordering. Moving cleanup into a callback registered after `aclInit`
preserves lazy initialization and makes the ordering explicit without catching
exceptions or bypassing normal exit.

### Keep direct CUDA-to-ACL identifier replacements

Rejected. Direct replacement cannot synthesize CUDA's lazy default context,
does not preserve `cudaDeviceReset()` arity, and does not adapt the device-count
output width.

### Retry `aclInit` after every failure

Rejected. The active ACL contract permits one process initialization and does
not establish that arbitrary failures are safely retryable. The first failure
is returned consistently instead of risking partial double initialization.

## Consequences

- A translated `vectorAdd`-style program can reach its first `cudaMalloc`
  without a source-level `cudaSetDevice`; the shim initializes ACL and binds
  logical device zero first.
- Explicit source `cudaSetDevice`, `cudaGetDevice`, `cudaGetDeviceCount`,
  `cudaDeviceSynchronize`, and no-argument `cudaDeviceReset` now retain CUDA
  call shapes.
- Stream and event calls no longer assume that an unrelated earlier operation
  happened to establish an ACL context.
- Repeated initialization and cleanup have a single, testable ownership rule;
  one post-initialization exit callback performs primary cleanup and the later
  destructor is a no-op.
- More than 64 distinct compatibility-owned device bindings fail closed; the
  implementation makes no silent cleanup claim beyond that explicit bound.
- A raw `<<<...>>>` launch that is literally the first CUDA operation has no
  compatibility function call to trigger the manager. The header exports
  `ascify::cudaRuntimeEnsureReady()` for a future target-proven launch guard;
  this ADR does not claim that launch-first case until the frontend inserts and
  CCEC validates that guard.
- Process-exit cleanup errors cannot be returned. Programs requiring cleanup
  evidence should call translated `cudaDeviceReset` explicitly.
- Failure to register the callback fails closed before device binding; owned
  initialization is rolled back immediately and borrowed initialization is not
  finalized.
- Each release must bind host, target, and device results to the exact source
  and binary under test. Results from an earlier source snapshot do not
  substitute for a clean release gate.

## Verification

- `tests/rewrite/runtime_lifecycle_compat_test.cpp` runs eight isolated process
  cases: owned initialization, borrowed/repeated initialization, failed
  initialization, 64-slot exhaustion, explicit reset before exit cleanup,
  owned and borrowed callback-registration failure, and rollback-error
  propagation. It checks allocation-free bounded admission, registration only
  after successful initialization, default-device creation, explicit device
  switching, reset/rebind behavior, unique reverse cleanup, finalize ownership,
  callback/destructor idempotence, downstream-call suppression, and peek/get
  error behavior.
- `tests/rewrite/simt_compat_input.cu` and `simt_compat.expected` require the
  CUDA-shaped lifecycle, stream, and event calls to lower to `ascify::` shims
  and forbid the old direct device mappings.
- `tests/rewrite/check_simt_compat.sh` compiles/runs host compatibility tests and
  statically rejects a return to direct lifecycle mapping, dynamic active-device
  nodes, or `std::vector`/`std::mutex` state.
- Every release source must pass the complete host-only gate and a clean
  translated-fixture gate. Target syntax, link, and device-correctness results
  must be recorded separately with their source and binary identities. Those
  build-specific records are intentionally outside this ADR so the accepted
  lifecycle contract is not confused with one build or device run.
