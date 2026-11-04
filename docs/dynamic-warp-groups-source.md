# Dynamic warp groups

Ascify supports a bounded register-group source domain on `dav-c310-vec` with
ordinary C++17 CUDA, recipe `none`, and the explicitly selected
`ascify-admitted-v1` parsing profile. The converter proves the operations in
the current translation unit before emitting the target implementation.
Declaring a group in the parsing profile does not itself authorize its use.

## Operations

| Source operation | Admitted domain |
|---|---|
| Grid coordinates | Direct local `this_grid()` value; rank and size on a one-dimensional launch |
| Coalesced group | Direct local `coalesced_threads()` value |
| Binary partition | An earlier local coalesced group or a genuine local tile32 partitioned from the same function's thread block |
| Labeled partition | An integer label and an earlier local coalesced group |
| Queries | Group rank and size with the checked source signatures |
| Broadcast | int32 or uint32 from literal group rank zero |
| Reduction | int32 with the exact admitted sum or maximum functor |

Values must be direct, non-static, non-volatile locals in ordinary device or
kernel functions. Copies, references, addresses, assignments, escaping values,
layout inquiries and caller templates do not acquire this contract. A static
parent must use a literal tile size of 32 and the verified SDK conversion.
A separate existing full-warp contract supplies tile synchronization; dynamic
groups do not add memory-order or partial-warp barriers.

The backend checks membership and participation at register exchanges. Other
active groups may coexist. Static32 partition requires a complete active warp.
Integer sums use defined uint32 modular operations in the backend; callers
still need a defined CUDA source domain, including signed-overflow bounds.

## Source and dependency checks

The converter inventories group types, values, references, calls, dependent
lookups and unevaluated observations throughout the caller translation unit,
including retained headers. Only exact witnessed nodes receive an exception
from the existing generic audits. SDK records, functions and redeclarations
must belong to the physically verified parsing projection. A retained caller
header must join the selected source closure before its operations can be
admitted.

Spelling, qualification, macro history and conditional source remain checked.
Caller definitions inside the SDK namespace, foreign aliases, representation
observations, inactive protected-name uses and caller conditionals are refused.
An altered profile, a wrapped profile namespace or a macro expansion that
erases an exposed group name cannot bypass activation.

Frozen sample helpers may provide their existing conditional-token evidence;
actual helper conversion and transactional publication must still complete.
Physically verified compiler resources and native headers supply only their
specific input roles. An input ledger does not grant AST or macro ownership.
The backend is resolved from the matching product layout and checked again at
publication. Selected-header conversion reconstructs the proof in the strict
joint parse; AST pointers do not cross that boundary.

The parsing profile's additional reduce overload is handled without weakening
older scalar reduction checks. An unselected overload candidate is accepted
only with its complete pinned declaration and redeclaration shape; concrete
calls still require their original exact type, functor and group witnesses.

## Validation and limits

The frozen source passed a fresh native O3 build, installation and all four
CTest checks. The same translator completed the existing 307 baseline cases,
scalar reduction and forwarding checks, Uniform47, coordinate, NTTP and
19 dynamic-group cases. Positive CPU semantic checks and atomic refusal
checks cover the corresponding bounded source domains.

The complete original CUDA Samples v13.0 `binaryPartitionCG` source also passed
native CUDA device compilation for its nine declared architectures, conversion
and complete CANN O2 compilation and linking. Its original NPU program passed
with 102400 elements and the original occupancy-selected 128-by-1024 launch;
all three outputs matched independent exact arithmetic over the actual input.
See the [original-target validation](validation/binary-partition-cg-20261103.md)
and the [bounded launch ABI completion](dynamic-kernel-abi-tail.md).
Native CUDA source compilation is distinct from execution on an NVIDIA device.

The original `warpAggregatedAtomicsCG` architecture-selected regions and
runtime capability gate remain unresolved. This feature does not drop those
regions or fabricate a CUDA architecture value. Other group algorithms,
launch shapes and SDK/header families require their own target validation.
