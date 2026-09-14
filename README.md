# Ascify

**Translate CUDA C/C++ source for Ascend.**

English · [简体中文](README.zh-CN.md) · [Documentation](docs/README.md) · [Contributing](CONTRIBUTING.md)

Ascify uses Clang to analyze CUDA source and rewrite supported APIs and kernel
constructs to Ascify compatibility headers and Ascend ACL interfaces. The
`ascify-clang` executable produces source that you compile, link, and validate
with your Ascend target toolchain.

```text
CUDA source  →  ascify-clang  →  Ascend-compatible source  →  CANN build and device tests
```

## What it supports

- **Source migration:** CUDA runtime and device constructs within the documented
  compatibility surface, with conversion statistics and optional JSON receipts.
- **Local headers:** Optional recursive conversion with checked dependency
  provenance and transactional output publication.
- **Ascend execution paths:** A general SIMT compatibility path and an explicit
  SIMD+SIMT Hybrid path for recognized row-wise Softmax, RMSNorm, and LayerNorm.
- **Conservative boundaries:** Unsupported patterns retain their original source
  or fail explicitly, according to the relevant conversion contract.

Generated source is one migration stage. It does not establish target compilation,
linking, numerical correctness, or performance. Full CUDA Samples application
coverage is ongoing work; see the [validation matrix](docs/validation-matrix.md)
for measured populations, tested commits, and remaining gaps.

## Quick start

You need CMake 3.16.8+, a C++17 compiler, Ninja, matching LLVM/Clang development
files, and a CUDA Toolkit directory for parsing. Conversion does not require a
GPU or NPU. Python 3.9+ is required for repository checks and test tooling.

```bash
git clone https://github.com/Edconeone/ascify.git
cd ascify

export LLVM_BUILD_DIR=/path/to/llvm
./build.sh
cmake --install build
```

`LLVM_BUILD_DIR` may be an LLVM development installation or build tree containing
both `LLVMConfig.cmake` and `ClangConfig.cmake`. For a checkout, you can instead set
`LLVM_PROJECT_PATH=/path/to/llvm-project` to use its `build/` directory.

Convert the included FP32 vector-add example:

```bash
mkdir -p .work/examples

ascify_install/bin/ascify-clang examples/vector_add.cu \
  --cuda-path=/path/to/cuda \
  --target-policy=dav-c310-vec \
  --migration-receipt="$PWD/.work/examples/vector_add.receipt.json" \
  -o "$PWD/.work/examples/vector_add.cu.dpp" \
  -- -std=c++17
```

The installed executable locates its private Clang parsing resources in
`libexec/ascify/clang/<major>/`. Custom layouts can pass
`--clang-resource-directory=/path/to/clang/resource`. Internal
`CUDA2DPP` filenames and the `.dpp` output suffix are retained Ascend compatibility
names. The [English guide](docs/user-guide.en.md) and
[中文使用手册](docs/user-guide.zh-CN.md) cover dependency setup, result checks,
target compilation, and troubleshooting.

## Conversion modes

| Mode | Selection | Scope |
|---|---|---|
| Conservative defaults | `--target-policy=portable --simt-math=precise` | Default source conversion policy |
| Target SIMT | `--target-policy=dav-c310-vec` | Target-specific compatibility and guarded rewrites |
| SIMD+SIMT Hybrid | Add `--simt-math=fast --target-recipe=dav-3510-rowwise-simd-v1` to target SIMT | Explicit, proved row-wise recipes with whole-operator SIMT fallback on selector misses |

Hybrid output needs the separately built runtime in
`runtime/dav_3510/rowwise/`. Start with the
[row-wise conversion guide](docs/rowwise-simd-conversion.md) for supported
shapes, ABI requirements, and target validation.

## Documentation

| Task | Guide |
|---|---|
| Install and convert your first source | [English](docs/user-guide.en.md) · [中文](docs/user-guide.zh-CN.md) |
| Look up CLI options and conversion boundaries | [Source conversion reference](docs/conversion-reference.md) |
| Inspect coverage and performance evidence | [Validation matrix](docs/validation-matrix.md) |
| Add a capability or fix a bug | [Contributing](CONTRIBUTING.md) · [Architecture decisions](docs/decisions/) |
| Prepare a release | [Release process](docs/release-process.md) |

## Development

```bash
python3 tools/check_repository.py --root .
ASCIFY_BINARY= sh tests/run_release_checks.sh
```

The host suite checks repository and semantic contracts. Native build, installed
CLI, actual CUDA conversion, and device checks have separate scopes; see
[Contributing](CONTRIBUTING.md) for the commands and required evidence.

| Directory | Responsibility |
|---|---|
| `src/` | Clang analysis, rewrite rules, CLI, and statistics |
| `include/ascify/`, `acl_cub/` | Public compatibility headers consumed by generated Ascend source |
| `frontend_compat/` | Versioned profiles used when parsing CUDA input |
| `runtime/` | Device implementations linked by generated Hybrid code |
| `tests/`, `examples/` | Fixed fixtures, checks, and CUDA examples |
| `docs/`, `tools/` | Documentation and development tools |

For example, a converted `cudaMalloc` call uses `ascify::cudaMalloc` from
`<ascify/ascify_cuda_compat.hpp>`. The `ascify/` prefix names the public header
namespace; compile generated code with `-I<install-prefix>/include`. The
translator implementation itself lives in `src/`. Its private Clang parsing
headers are installed under `libexec/ascify/clang/<major>/include`, separate from
the public compatibility headers.

## License and attribution

See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Individual file notices retain their terms, including inherited MIT-licensed
translator code and third-party CUDA fixtures. Copyright and source attribution
are preserved in the repository and installed notices.
