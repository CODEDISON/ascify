# Grid-constant value parameters

The `dav-c310-vec` target can specialize CUDA's `__grid_constant__` qualifier
when the AST proves that only ordinary parameter values are observed. It removes
the qualifier token, retaining the const by-value parameter type, kernel body,
and launch. This does not implement CUDA's grid-wide parameter-object identity,
address-sharing or lifetime contract.

The supported kernel is a defined non-template free function. Every declaration
must appear in the main translation unit and explicitly use the CUDA-owned
`__grid_constant__` macro, including its transitive macro dependencies. User
redefinitions, aliases, direct attributes, inherited or additional attributes,
and declarations in headers are rejected. Every qualified parameter must be a
const, non-volatile, trivial standard-layout aggregate defined in the same main
file, with exactly one ordinary non-mutable constant-length array of 32-bit
`int`, with total size at most 32000 bytes. Bases, methods, other members,
macro-defined record/field declarations, layout attributes and non-default
layout are excluded. The size ceiling bounds the admitted ABI envelope by the
largest validated payload; it is not a claim about the target's hardware limit.
Exactly one parameter may carry the qualifier. The kernel has at most four
parameters, with the record first or last; the others may contain up to two
32-bit `int`/`unsigned int` values and one 64-bit pointer to those element types.
The aligned source-layout sum, including tail
padding, must not exceed 32024 bytes. This is a conservative admission bound
derived from the validated parameter shapes, not a complete CCEC kernel ABI
proof. Additional aggregate, reference and wide scalar arguments are rejected.

Every use must directly select that array, subscript it, and read a scalar
value. The scalar value may participate in arithmetic or be passed by value.
Parameter/array/element address-taking, reference binding, array decay escaping
the subscript, whole-object copies, captures and reflection are rejected. The
same proof covers all redeclarations before any qualifier is changed. A failed
proof emits `Ascify grid-constant boundary` and publishes no output.

The frontend does not claim a universal target parameter-size limit. Existing
950DT / CANN 9.1 `dav-c310-vec` validation covered int-array lengths 1, 26, 27,
1024, 1536, 1537 and 8000, plus a scalar control, in 96 launches with signed
patterns, both argument orders, two launch geometries and independent raw-byte
checking. This establishes value transport at those tested points. It does not
certify other target versions, arbitrary layouts or address-observing programs;
target compilation and original program correctness checks remain required.

Run `tests/rewrite/check_grid_constant_parameters.py` with `--binary`, `--clang`,
`--cuda`, `--resource` and a fresh `--evidence-dir`. It first checks every source
with a real CUDA frontend, then checks admitted transformations and precise
semantic rejections. `--syntax-only` stops after source validation. Every
subprocess is bounded and all inputs, commands and diagnostics are retained.
