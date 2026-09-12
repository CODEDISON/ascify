# ADR 0026: Private device copy and shared atomic boundaries

## Status

Implemented candidate; local LLVM 23 transformation and host contracts passed.
Target compilation and device validation remain pending.

## Context

The unchanged CUDA Samples `shfl_scan` closure calls
`memcpy(&out, result, sizeof(out))` in `shfl_integral_image.cuh:158`.
Both operands are thread-private local objects: sixteen unsigned integers and
a record containing four `uint4` vectors, each 64 bytes. The DT target compiler
currently resolves the emitted name to the host C library and rejects the
device call. This call does not need a GM or UBUF copy API.

The separate `histogram256.cu:44` atomic call reaches shared storage through
two helper parameters and an immutable kernel alias. The frozen sample uses
six warps, 192 threads, and 1536 unsigned counters (6144 bytes). A historical
CANN beta3 experiment compiled a native UBUF atomic, but did not establish
the current DT API or runtime atomicity. It is not current acceptance evidence.

## Decision

Add an independent AST planner for the exact private-copy semantic domain.
It requires the trusted global C `memcpy` builtin or Clang's trusted static-inline
device wrapper. That wrapper must consist of exactly one return of
`__builtin_memcpy` with its three parameters forwarded directly in order,
without source macros or additional evaluation. Both entry paths check every source-backed
redeclaration against the system-file identities captured when files entered
the frontend. Host calls, namesakes, macros, templates, explicit casts, aliases,
parameters, shared objects, dynamic lengths, and unequal sizes receive no edit.

Admission requires standard preprocessing; the legacy mode that retains
excluded branches makes no private-copy edit. The admitted call uses
`&destination`, a distinct local source array, and
`sizeof(destination)` inside a non-template device function. Both complete
objects must be trivially copyable, non-volatile, and 64 bytes. The generated
internal helper copies unsigned-character object representations and returns
the original destination pointer. It does not allocate, synchronize, cast an
address space, or call a host memory API. The target template also checks the
actual destination and source sizes, so a target ABI size difference fails
compilation instead of producing a truncated copy.

The common Ascify transaction layer remains responsible for macro collisions,
overlapping replacements, and including the authenticated compatibility header.
No generic `memcpy` name mapping is added.

Shared atomic lowering remains separate and unadmitted until the current DT
native API and address provenance are established. A GM cast, a non-atomic
read/modify/write, a warp-only scope applied to shared block data, or a software
spinlock is not an acceptable substitute. Any helper-body rewrite must prove
all relevant parameter flows, aliases, callers and escapes; a successful
compiler invocation alone cannot prove those properties.

## Validation

The host copy contract tests all 256 starting byte patterns, a word bit-pattern
case, the returned pointer, destination canaries, and a wrong target-size
rejection. CUDA fixtures cover admitted bare and returned-value calls and the
unadmitted provenance/size/template/macro/host/name domains. The native device
probe checks two copies in each of 256 lanes across four blocks, for 32768
copied bytes in total. Target compilation, device results, and original sample
counts must be recorded separately; none are claimed by the host contract.
