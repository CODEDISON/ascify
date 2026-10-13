# Attribute-only CUDA guards

For `dav-c310-vec` with ordinary preprocessing, Ascify can preserve CUDA host
and device callability when a function's two attributes are enclosed in a
literal `#ifdef __CUDACC__` branch. The original compiler-owned CUDA macro must
make the branch active. The complete body must contain only `__host__` followed
by `__device__`; whitespace and comments are allowed. No `else`, `elif`, nested
directive, alias, extra declaration or attribute is admitted.

Both expanded attributes must belong to the same defined free function or
primary free-function template. The attribute macros and every macro dependency
must have authenticated CUDA SDK or owned compiler provenance. Methods,
class-scope friends, inherited attributes and template specializations are
excluded. User or command-line modification of `__CUDACC__`, including a later
modification in the same translation unit, retains all such guards.

An admitted opener becomes
`#if 1 /* Ascify: proven CUDA host/device attributes */`. The closing directive,
attributes and function body stay in place; the existing device-attribute
translation changes `__device__` to `__aicore__`. Only the physically translated
main file is edited, including a local header when the recursive translator
processes that header as its own main file. All other CUDA conditionals retain
their source text. No global CUDA macro definition is injected.

The rule stages its opener replacements together after preprocessing and AST
proof. A conflict with another semantic replacement retains the staged set.
This does not generalize arbitrary CUDA conditional code, alter CPU algorithms,
or establish that a complete Sample can compile or execute.

`tests/rewrite/check_cuda_attribute_guards.py` checks source-valid admission and
retention cases. A separate CANN 9.1 `dav-c310-vec` compile pair reproduced the
original host-only failure and verified that opening only the attribute branch
allows both host and device callers, including a defined host symbol. Numerical
or original Sample execution is a separate gate.
