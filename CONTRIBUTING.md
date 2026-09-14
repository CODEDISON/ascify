# Contributing

Ascify accepts focused changes to CUDA-to-Ascend conversion, target compatibility,
tests, documentation, and developer tooling. Start from a reproducible input and
state the stage you are improving: source generation, target compilation, linking,
correctness, or performance.

## Local checks

Run the repository check and host release gate before committing (C++17 compiler,
Clang, Bash, CMake, ripgrep (`rg`), and Python 3.9 or newer required).
On Ubuntu or Debian, install the host test dependencies with:

```bash
sudo apt-get update
sudo apt-get install -y build-essential clang cmake git ninja-build python3 ripgrep
```

Then run:

```bash
python3 tools/check_repository.py --root .
ASCIFY_BINARY= sh tests/run_release_checks.sh
```

For translator golden checks, build `ascify-clang` and provide its CUDA parsing
dependencies:

```bash
ASCIFY_BINARY=build/ascify-clang \
ASCIFY_CUDA_PATH=/path/to/cuda \
ASCIFY_CLANG_RESOURCE_DIRECTORY="$PWD/ascify_install/libexec/ascify/clang/23" \
sh tests/run_release_checks.sh
```

The same host gate is registered with CTest and used by GitHub CI:

```bash
cmake -S . -B build/host-ci -DASCIFY_CLANG_TESTS_ONLY=ON \
  -DASCIFY_TEST_BINARY= -DASCIFY_TEST_CUDA_PATH= \
  -DASCIFY_TEST_CLANG_RESOURCE_DIRECTORY=
ASCIFY_BINARY= ctest --test-dir build/host-ci --output-on-failure
```

The host job excludes real translation and NPU execution. Native build and
installation checks exercise those host-side product entry points separately.
Use absolute paths for the binary and parsing dependencies when running real
translator checks through CTest.

Public headers for generated source are installed under `include/ascify/` and
`include/acl_cub/`. The translator's private Clang parsing headers are installed
under `libexec/ascify/clang/<major>/include`. Explicit resource-directory settings
point to that directory's parent; the examples use Clang 23.

After a native build and installation, check the public CLI and convert the
included FP32 example with the installed executable:

```bash
python3 tests/engineering/test_cli_surface.py --binary "$PWD/build/ascify-clang"
python3 tests/engineering/check_install.py \
  --binary "$PWD/ascify_install/bin/ascify-clang" \
  --cuda-path /path/to/cuda
```

The installation check tests default resource discovery. For a layout with
external Clang resources, add `--resource-dir /path/to/clang/resource`. Keep a
separate run without that override when validating the self-contained resource
layout. These checks cover source conversion, not CANN compilation or device
execution. See the [release process](docs/release-process.md) for each level.

## Changes to conversion or runtime behavior

Changes to a target recipe require:

1. positive and adversarial negative rewrite coverage;
2. a clean native build, actual conversion, and mutation replay on the named
   integration environment (DT for phase-one closeout; preserve the original
   identity of historical 910C runs);
3. 950PR direct/native correctness and performance gates;
4. an ADR update when the semantic proof, runtime domain, or acceptance
   threshold changes.

## Repository conventions

Do not commit `.work`, generated headers, binaries, raw benchmark CSV, device
locks, or evidence bundles. Commit only source, fixed fixtures, expected
outputs, and small evidence summaries bound to immutable hashes.

Keep changes focused. Preserve source and fixture paths during release
engineering, and avoid repository-wide formatting. Update the relevant guide,
Unreleased changelog, and ADR when behavior or its contract changes. A bug or
coverage report should identify the input, source commit, dependency versions,
command, first failing stage, and actual diagnostic; redact private connection
details. Do not describe generated source as a working migrated application.

Preserve upstream copyright and license notices. Historical translator attribution
belongs in third-party notices; product options, build dependencies, and generated interfaces
must describe the supported CUDA-to-Ascend path. Do not remove or reformat frozen
CUDA fixtures to satisfy a branding or style check.

Use the issue templates for reproducible bugs and scoped feature requests. In a
pull request, explain the problem and resulting behavior, record the checks you
ran, and call out affected compatibility boundaries. See
[ADR-0029](docs/decisions/0029-cuda-to-ascend-product-contract.md) for the current
product and engineering contracts.
