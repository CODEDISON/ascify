# Explicit host C++ translation units

Some CUDA projects also compile ordinary C++ files. For example, the original
CUDA Samples v13.0 `imageDenoising/bmploader.cpp` declares its own `uchar4` and
does not include CUDA. Parsing that file as CUDA injects SDK vector types and
causes a redefinition before translation can start.

`--input-language=auto|cuda|c++` selects the parsing contract. `auto` is the
default: an explicit compiler `-x c++` selects host C++, and `-x cuda` selects
CUDA. With no explicit language, parsing remains CUDA for compatibility,
including inputs named `.cpp`. File extensions do not select host mode.
`--input-language=c++` or `cuda` must agree with an explicit compiler `-x`;
conflicts fail instead of silently changing a compiler command. Only these two
languages are admitted. Multiple `-x` options use the last option before the
original input. Language options after the input are rejected as ambiguous.

The compiler command comes from CommonOptionsParser, including its fixed
command after `--`, compilation database and `--extra-arg[-before]` adjusters.
An original translation unit must have exactly one command and one input.
Unexpanded response files are rejected, so hidden language options cannot
bypass this contract. The selected command and language are inherited by
local-header discovery, standalone header conversion and contextual proof;
temporary `.ascify` names and a selected header's `.cuh` suffix never select a
different language. Normalization removes only original input/language tokens
and substitutes the current parse input with the single selected language.
Other command options, working directory, include order and macros remain.

Host mode runs the same Clang AST matcher, preprocessing callbacks, source
identity checks, edits and publication path. It does not copy C++ around the
translator. Clang's language is C++, not CUDA host-only: no automatic
`cuda_runtime.h`, `__CUDACC__` or device attributes are introduced. Host mode
uses normal conditional preprocessing and rejects `--amap`. An explicit
`#include <cuda_runtime.h>` is still parsed and Runtime API calls still use
normal semantic conversion. `--cuda-path` locates its include directory in
host mode; it does not force that header into a TU.

Explicit compiler CUDA/device flags, forwarded frontend arguments and command
line CUDA language predefines conflict with host mode and fail. They are not
silently removed. A `.cu` root requested as host is rejected, even when its
CUDA code is hidden by conditional macros. For a genuine mixed project,
translate `.cu` roots as CUDA and `.cpp` roots as explicitly declared C++;
preserve their complete source sets and linked declarations.

Example: `ascify-clang bmploader.cpp --input-language=c++ -o bmploader.cpp.dpp`
or `ascify-clang bmploader.cpp -o bmploader.cpp.dpp -- -x c++ -std=c++17`.
Target compilation and original numerical validation remain separate gates.
This feature alone establishes no new sample pass or device capability.

The precedence rules follow the Clang [driver input scan](https://github.com/llvm/llvm-project/blob/main/clang/lib/Driver/Driver.cpp)
and [Tooling argument adjusters](https://github.com/llvm/llvm-project/blob/main/clang/lib/Tooling/ArgumentsAdjusters.cpp).
For a fixed command, `--extra-arg-before=-xc++` precedes the original arguments;
`--extra-arg=-xc++` after the input is rejected. For a database command ending
in `-- input`, Tooling inserts extra arguments before that separator. A fixed
command after the tool's `--` takes precedence over database compiler flags;
the existing `-p` option still supplies the enumerated original input files.
