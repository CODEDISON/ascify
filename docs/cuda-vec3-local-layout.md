# CUDA vec3 local record lowering

This first profile preserves authenticated CUDA 13.4.1 `float3`, `uint3` and
`int3` as distinct plain C++ records with size 12, alignment 4 and member offsets
0/4/8. Native DPP vectors retain their original definitions. The frontend uses
declaration identity, physical SDK paths and complete provider hashes; a name
or matching source buffer alone does not authorize a rewrite.

The CUDA uint3 definition has one legitimate earlier forward declaration in
Clang's builtin-variable header. This non-definition is accepted only in the
executable-owned resource directory, with exact builtin and including-wrapper
hashes, physical file/UniqueID checks, no overridden buffers, and an empty
top-level struct AST with no attributes. A caller-selected copied ResourceDir
or caller uint3 redeclaration does not gain this authority. The forward proof
is recorded in the migration receipt; whole builtin-vector conversions remain
outside the local profile.

The profile applies with `--target-policy=dav-c310-vec`, no target recipe and
`--default-preprocessor`. It admits automatic scalar locals, local aliases,
literal layout inquiries, trivial record copies and calls to defined ordinary
inline functions whose whole observed vec3 use is within the same conversion
closure. SDK three-component factories are replaced only after checking both
their declarations and the actual AST of their three assignments and return.
User inline overload names remain unchanged. The generated records remain in
the global namespace to preserve the CUDA records' associated lookup namespace.

Every old ABI-audit use needs an individual proof entry. There is no exemption
for an entire canonical type. Unsupported use, a macro observation, a dormant
raw vec3 spelling, source-token ambiguity or a conflicting edit discards the
entire plan. Existing output and local-header bundles retain their transaction
rules. Migration receipts record the local-only rule, provider identity,
source-buffer hashes and individual source sites after successful conversion.

Arrays, pointers to vec3, GM storage, fields and enclosing records, kernel
parameters of vec3 type, unknown external functions, indirect calls, templates,
generic lambdas, RTTI, volatile/atomic use, component-address escape and explicit
casts remain outside this profile. Layout equality does not grant a calling
convention or memory-access proof. Whole `threadIdx` values, `dim3(uint3)`, native
vector builtins and three-component texture results are separate boundaries.
Scalar `.x/.y/.z` access and already-supported float/float4 texture results keep
their existing treatment. `double3` and FP64 are not covered.

Component references are conservatively rejected at their first binding,
including local aliases, call/constructor reference parameters, reference
returns and reference aggregates. The origin check follows expression wrappers
such as casts, assignments, comma and conditional expressions; it does not grant
an escaped alias a later exemption. This also refuses some safe temporary
reference bindings. Scalar value reads, by-value scalar calls, and a reference
to a separately copied scalar remain allowed. Whole vec3 references still need
the existing closed inline/local proof.

Quoted headers already selected by recursive conversion use the existing
original-position context transaction, including token/macro/input equality.
A selected leaf must be successfully reparsed and rewritten before publication.
Angled headers are **not** silently selected. In the frozen CUDA Samples,
recursiveGaussian and simpleTexture3D include `<helper_math.h>`; its complete
overload set therefore remains rejected until a separately reviewed full-header
entry adapter exists. No include, overload, source TU or dependency is removed.
This stage targets the local accumulators in imageDenoising first and makes no
claim that any of the five related complete projects now builds or passes.

`alignedTypes` is a separate case: its `RGB32_misaligned` and explicitly
16-aligned `RGB32` are user records rather than CUDA uint3. Their layout must
remain unchanged. The previously recorded failure was in helper/device-property
support and is not attributed to this rewrite.

Validation has three distinct levels:

- `check_cuda_vec3_layout_host.sh` checks actual C++ record layout, aggregates,
  typed copy/assignment, field values, array stride and guard bytes around
  4/8-offset placements. Array checks validate the representation, not frontend
  array admission or device memory.
- `check_cuda_vec3_local_conversion.py` has 17 positive and 44 negative real
  frontend cases, checks complete output preservation and receipt boundaries,
  and optionally compiles positive outputs with real CCEC. It runs no device.
- `check_vec3_abi_boundary.py` retains the existing provider/CUB matrix, changing
  only seven now-covered local cases. Unknown genuine compiler providers retain
  their former generation behavior and receive no new layout safety claim.

Real device readback with independent scalar references and 12-byte guard/stride
tests is required before admitting GM, by-value kernel or cross-TU boundaries.
Complete original sample build and numerical verification remain required for
any sample count. A source conversion, model test or compiler pass is not such
an acceptance.
