# ADR-005: Use native priorities for named nonblocking streams

Status: Accepted

## Context

CUDA and CANN both expose stream priorities, but their numeric endpoints and
flag meanings differ. Copying a CUDA stream flag into ACL would enable an
unrelated fast-launch option. CUDA blocking streams also require implicit
default-stream ordering that this runtime profile does not provide.

## Decision

Query CANN's actual range, preserve its ordering and endpoints, clamp signed
requests, and create only named nonblocking priority streams with native flags
0. Restrict this adaptation to the reviewed legacy SIMT / ACL 1.17.0 ABI and
authenticate the original CUDA SDK declarations. Propagate native failures and
reject unsupported versions, ranges, and ordering requests.

## Consequences

Callers can use native priority without a fabricated CUDA range or an ordering
emulation layer. Blocking priority streams remain unsupported. Correct output
and successful native priority readback do not imply preemption or a particular
performance ordering. See the [public contract](../stream-priorities.md).
