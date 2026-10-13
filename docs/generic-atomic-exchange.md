# Generic integer atomic exchange

On the admitted CANN 9.1 pure SIMT target, a resolved SDK int32/uint32
`atomicExch` can use `asc_atomic_exch` without an address-space cast. The
operation replaces the addressed value atomically and returns its previous
value. Generic addresses keep their actual GM or UB space. Public 8.5 and
mixed SIMD/SIMT families reject this new helper at compile time.

The frontend changes only a direct unqualified SDK callee in a pure device
free function or CUDA kernel. It admits a current fixed int/uint pointer
parameter, the address of a current fixed int/uint lvalue-reference parameter,
or the address of a function-local shared scalar. Local aliases, automatic
nonshared scalars, fields, pointer arithmetic, explicit casts, wider/floating
types, host/device owners, class methods and macros remain unchanged.

A free function primary template may contain an already resolved nondependent
call whose address parameter and value have fixed int/uint types. This proof
uses the primary definition, never an observed specialization; dependent
addresses or values remain unchanged. Existing global-kernel pointer lowering
and generic atomic-add contracts are unchanged.

`tests/rewrite/check_generic_exch_conversion.py` first validates every fixture
with the real CUDA frontend and then checks admitted, retained and protected
macro outcomes. Source-only mode does not demonstrate conversion or target
execution. Separate GM/UB device evidence is required for numerical acceptance.
