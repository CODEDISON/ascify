# Device float frexp

Ascify can translate the CUDA SDK's resolved `float frexp(float, int*)` call to
the CANN 9.1 legacy SIMT `frexpf` implementation. This changes only the callee;
the argument expressions, float conversion, output pointer and evaluation count
stay unchanged. It does not narrow a double overload or alter host math.

The proof requires standard preprocessing, a direct unqualified call written in
the current main file, the exact SDK/owned-compiler declaration chain and a
non-template device or global function. Calls in host/device functions, templates,
lambdas, macros, user overloads and qualified or indirect calls stay unchanged.
Retained source may consequently remain unsupported by the target compiler.
The generated helper and its native function name belong to the authenticated
compatibility header's protected identifier set.

The target declaration is `float frexpf(float, int*)` in
`include/simt_api/math_functions.h`; its implementation is in the corresponding
`impl/simt_api/math_functions_impl.h`. It returns the signed normalized fraction
and writes the binary exponent. The CANN 950 documentation specifies zero
exponents for zero, infinities and NaNs. The public 8.5 header family is not
admitted for this new operation and rejects the helper at compile time.

`tests/rewrite/check_float_frexp_conversion.py` checks source-valid positive and
retained cases plus protected-macro publication refusals with the real frontend.
Its `--syntax-only` mode establishes source validity only. Host contract tests
and source conversion do not establish target numerical behavior; native CCEC
and device validation are separate gates.
