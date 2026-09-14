# Ascify User Guide

English | [简体中文](user-guide.zh-CN.md)

## 1. What Ascify does

Ascify translates CUDA C/C++ source for Ascend. It uses Clang to read CUDA source files, then rewrites supported CUDA APIs and kernel constructs to Ascify compatibility headers and Ascend ACL interfaces.

The full flow has two parts:

```text
CUDA source
  -> ascify-clang converts the source
  -> Ascend-compatible source
  -> CANN or your target project builds, links, and tests the code
```

These two results are not the same:

| Result | Meaning |
|---|---|
| Ascify returns 0 and writes a non-empty file | Source conversion is complete |
| The new code passes build, link, and device tests | The target program has passed its tests |

Source conversion does not need an NVIDIA GPU or an Ascend NPU. The conversion machine does need a full CUDA Toolkit directory. To build and run the new code, you also need CANN, the Ascify headers, the target libraries, and an Ascend device.

## 2. Requirements

### 2.1 Required software

| Software | Use | Note |
|---|---|---|
| Git | Download the source | Use a recent version |
| Bash | Run `build.sh` and `run.sh` | Both files are Bash scripts |
| CMake | Set up the build | Ascify needs version 3.16.8 or later |
| Ninja | Build the project | `build.sh` uses Ninja by default |
| ripgrep (`rg`) | Check expected compiler diagnostics | Required by the host release checks |
| C/C++ compiler | Build Ascify | It must support the current machine |
| LLVM and Clang development files | Provide the Clang front end, libraries, and CMake files | The `clang` command alone is often not enough |
| CUDA Toolkit | Provide CUDA headers and `libdevice` | An NVIDIA GPU is not required |

You do not need Python to build Ascify. Repository checks, conversion wrappers, and tests require Python 3.9 or newer. The host release checks also need Clang and ripgrep.

The following setup is recorded in historical validation. See the [validation matrix](validation-matrix.md) for its result attribution and current candidate status:

| Item | Tested value |
|---|---|
| System | Linux AArch64 |
| LLVM/Clang | 23.0.0git, commit `caf619642a6dbb216969a9450d33dbac5a8d30df` |
| CUDA Toolkit | 12.8 |
| CMake | 4.3.4 |
| Ninja | 1.13.0 |

Other versions may work. This table only lists versions that the project has tested.

### 2.2 Install the basic tools

On Ubuntu or Debian, run:

```bash
sudo apt-get update
sudo apt-get install -y build-essential clang cmake git ninja-build python3 ripgrep
```

On another Linux system, install the same tools with its package manager. Then run:

```bash
git --version
cmake --version
ninja --version
python3 --version
rg --version
```

If CMake is older than 3.16.8, update it first. LLVM may need a newer CMake version. Check the build requirements for your LLVM version.

## 3. Prepare LLVM and Clang

You can set `LLVM_BUILD_DIR` directly to an LLVM development installation
containing `LLVMConfig.cmake` and `ClangConfig.cmake`. `build.sh` does not require
an LLVM source checkout. The following steps use an existing source build.

### 3.1 Use an existing LLVM build directory

The fastest setup uses an existing `llvm-project` build. It must have these files:

```text
llvm-project/
├── clang/
├── llvm/
└── build/
    ├── bin/clang
    ├── bin/clang++
    ├── lib/cmake/clang/ClangConfig.cmake
    └── lib/cmake/llvm/LLVMConfig.cmake
```

Set the paths:

```bash
export LLVM_PROJECT_PATH=/path/to/llvm-project
export LLVM_BUILD_DIR="$LLVM_PROJECT_PATH/build"
```

Check the files and tools:

```bash
test -x "$LLVM_BUILD_DIR/bin/clang"
test -x "$LLVM_BUILD_DIR/bin/clang++"
test -f "$LLVM_BUILD_DIR/lib/cmake/llvm/LLVMConfig.cmake"
test -f "$LLVM_BUILD_DIR/lib/cmake/clang/ClangConfig.cmake"

"$LLVM_BUILD_DIR/bin/clang" --version
"$LLVM_BUILD_DIR/bin/llvm-config" --host-target
```

The last command must show a target triple for the current machine. If it shows nothing, the CMake compiler check may fail. The check may also fail if LLVM does not have the host target.

### 3.2 Build LLVM from source

If the machine has no usable LLVM build, you can build it from source. Building LLVM can take a lot of time and use a lot of memory and disk space. Read the [LLVM CMake guide](https://llvm.org/docs/CMake.html) before you start.

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

# Optional: use the LLVM commit that Ascify has tested.
git checkout caf619642a6dbb216969a9450d33dbac5a8d30df

case "$(uname -m)" in
  x86_64)
    LLVM_HOST_TARGET=X86
    LLVM_HOST_TRIPLE=x86_64-unknown-linux-gnu
    ;;
  aarch64|arm64)
    LLVM_HOST_TARGET=AArch64
    LLVM_HOST_TRIPLE=aarch64-unknown-linux-gnu
    ;;
  *) echo "Choose an LLVM target for this machine" >&2; exit 1 ;;
esac

cmake -S llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_TARGETS_TO_BUILD="${LLVM_HOST_TARGET};NVPTX" \
  -DLLVM_DEFAULT_TARGET_TRIPLE="${LLVM_HOST_TRIPLE}"

cmake --build build --parallel 2
cd ..
```

Build the full default target to produce the Clang development libraries used by Ascify; building only the `clang` executable does not prepare all of these dependencies. Then set these paths:

```bash
export LLVM_PROJECT_PATH=/path/to/llvm-project
export LLVM_BUILD_DIR="$LLVM_PROJECT_PATH/build"
```

You can also use LLVM development packages from your Linux system. `LLVM_BUILD_DIR` must identify the prefix containing LLVM and Clang CMake configurations, development libraries, and matching parsing resources. The host compilers may live elsewhere: select them with `ASCIFY_CC`/`ASCIFY_CXX` or `CC`/`CXX`. Without overrides, the script tries Clang in the prefix, then CMake compiler discovery. If `ClangConfig.cmake` is missing, install the matching Clang development package or use a complete LLVM source build.

## 4. Prepare the CUDA Toolkit

Ascify uses the CUDA Toolkit to read CUDA source files. The conversion machine does not need an NVIDIA GPU.

Install a CUDA Toolkit that supports the current machine, or use an existing full Toolkit directory. See the [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/). A common path is `/usr/local/cuda`:

```bash
export CUDA_PATH=/usr/local/cuda

test -f "$CUDA_PATH/include/cuda.h"
test -f "$CUDA_PATH/include/cuda_runtime.h"
test -d "$CUDA_PATH/nvvm"
```

`CUDA_PATH` must point to the Toolkit root, not its `include` directory. A partial directory with headers but no `nvvm` or `libdevice` may not work.

## 5. Download, build, and install Ascify

### 5.1 Download the source

```bash
git clone https://github.com/CODEDISON/ascify.git
cd ascify

git rev-parse HEAD
git status --short --branch
```

Save the commit value. You can use it to repeat the same conversion later.

### 5.2 Build

Set `LLVM_BUILD_DIR` to the development installation or source build prepared in section 3. This variable is sufficient; you do not also need `LLVM_PROJECT_PATH`:

```bash
export LLVM_BUILD_DIR=/path/to/llvm
export ASCIFY_BUILD_JOBS=2

./build.sh
```

`build.sh` uses these defaults:

- Build tool: Ninja.
- Build type: Release.
- Build directory: `build/`.
- Install directory: `ascify_install/`.
- C/C++ compilers: `ASCIFY_CC`/`ASCIFY_CXX`, then `CC`/`CXX`, then Clang in the LLVM prefix, then CMake host compiler discovery.
- Linker: selected by the compiler; set `ASCIFY_LINKER` to override it.

Check the new program:

```bash
test -x build/ascify-clang
build/ascify-clang --help >/dev/null
```

If the `clang` in the LLVM build cannot build a program for the current machine, use the system compiler:

```bash
export ASCIFY_CC=/usr/bin/clang
export ASCIFY_CXX=/usr/bin/clang++
./build.sh
```

You can change the build directory, install directory, job count, and build tool:

```bash
export BUILD_DIR="$PWD/build-release"
export INSTALL_ROOT="$PWD/ascify-install-release"
export ASCIFY_BUILD_JOBS=4
export CMAKE_GENERATOR=Ninja
./build.sh
```

If you change these paths, use the new paths in the next steps too.

### 5.3 Install

If you used the default build directory, run:

```bash
cmake --install build

test -x ascify_install/bin/ascify-clang
test -f ascify_install/libexec/ascify/clang/23/include/__clang_cuda_runtime_wrapper.h
test -f ascify_install/include/ascify/ascify_cuda_compat.hpp
```

The install command copies the Ascify program, compatibility headers, front-end
profiles, Clang resource headers, and license notices. The two header roles use
separate installed locations:

| Installed path | Consumer |
|---|---|
| `include/ascify/` and `include/acl_cub/` | Generated Ascend source; compile with `-I<install-prefix>/include` |
| `libexec/ascify/clang/<major>/include/` | The translator's private Clang parsing resources |

The commands here use the tested Clang major version `23`. Substitute your build's
resource version when using another toolchain. The resource-directory option points
to `libexec/ascify/clang/<major>`, the parent of its `include/` directory. The installed
binary discovers that location automatically unless an explicit override is supplied.

Do not copy only `ascify-clang` to another machine. The program also needs matching resource headers. It may also need LLVM shared libraries.

## 6. Run the first conversion

The next command converts the FP32 example `examples/vector_add.cu`. It uses pure SIMT mode and writes a JSON receipt.

```bash
export CUDA_PATH=/usr/local/cuda
export CLANG_RESOURCE_DIRECTORY="$PWD/ascify_install/libexec/ascify/clang/23"
export ASCIFY_BINARY="$PWD/ascify_install/bin/ascify-clang"

mkdir -p .work/examples

./run.sh examples/vector_add.cu \
  --target-policy=dav-c310-vec \
  --simt-math=precise \
  --target-recipe=none \
  --migration-receipt="$PWD/.work/examples/vector_add.receipt.json" \
  -o "$PWD/.work/examples/vector_add.cu.dpp" \
  -- -std=c++17
```

If the CUDA Toolkit is not at `/usr/local/cuda`, use its real path. `run.sh` reads these environment variables and calls `ASCIFY_BINARY`. Installed resource headers can also be discovered automatically; `CLANG_RESOURCE_DIRECTORY` explicitly overrides that location.

The `--` at the end splits Ascify options from Clang options. Here, `-std=c++17` goes to Clang.

### 6.1 Check the result

If the conversion command did not report an error, check the output files:

```bash
test -s .work/examples/vector_add.cu.dpp
test -s .work/examples/vector_add.receipt.json
```

Then check the main changes and the receipt status:

```bash
grep -E 'ascify_cuda_compat|acl/acl.h|aclrtMalloc' \
  .work/examples/vector_add.cu.dpp

grep -Eq '"status"[[:space:]]*:[[:space:]]*"succeeded"' \
  .work/examples/vector_add.receipt.json
```

The `succeeded` value only means that source conversion is complete. It does not mean that the target build, link, device run, result check, or speed test has passed.

### 6.2 Convert without the install step

If you built Ascify but did not run the install command, use the LLVM resource directory:

```bash
export ASCIFY_BINARY="$PWD/build/ascify-clang"
export CLANG_RESOURCE_DIRECTORY="$("$LLVM_BUILD_DIR/bin/clang" -print-resource-dir)"
export CUDA_PATH=/usr/local/cuda

mkdir -p .work/examples

./run.sh examples/vector_add.cu \
  --target-policy=dav-c310-vec \
  --simt-math=precise \
  --target-recipe=none \
  -o "$PWD/.work/examples/vector_add.cu.dpp" \
  -- -std=c++17
```

The install directory is better for regular use. It is also easier to record and check.

## 7. Convert your own source

### 7.1 Convert one file

```bash
./run.sh /path/to/project/kernel.cu \
  --target-policy=dav-c310-vec \
  --simt-math=precise \
  --target-recipe=none \
  -o "$PWD/.work/examples/kernel.cu.dpp" \
  -- -std=c++17 -I/path/to/project/include -DMY_FEATURE=1
```

Pass the include paths, macros, language version, and sysroot that the source needs. Put these options after `--`.

The source may build with `nvcc` but still fail in Ascify. Ascify will fail if an `-I` path, a `-D` value, or a generated header is missing.

### 7.2 Convert local headers with the source

Ascify converts only the input file by default. You can tell it to convert local headers that use quoted includes:

- `--local-headers`: convert local headers that the input file includes directly.
- `--local-headers-recursive`: keep following quoted local includes.

```bash
./run.sh /path/to/project/kernel.cu \
  --local-headers-recursive \
  --target-policy=dav-c310-vec \
  --simt-math=precise \
  -o "$PWD/.work/examples/kernel.cu.dpp" \
  -- -std=c++17 -I/path/to/project/include
```

Ascify writes the new headers to `.work/examples/kernel.cu.dpp.headers/`. This option does not process every system header. It does not add support for more CUDA APIs. See the [local header guide](local-header-closure.md) for the full rules.

### 7.3 Run `ascify-clang` directly

You can run the program without `run.sh`:

```bash
"$ASCIFY_BINARY" /path/to/input.cu \
  --cuda-path="$CUDA_PATH" \
  --clang-resource-directory="$CLANG_RESOURCE_DIRECTORY" \
  --target-policy=dav-c310-vec \
  --simt-math=precise \
  --target-recipe=none \
  -o /path/to/output.cu.dpp \
  -- -std=c++17
```

## 8. Choose a conversion mode

For the first conversion, use `dav-c310-vec + precise + target-recipe=none`. Check the basic flow first. Try another mode after the basic flow works.

| Goal | Options | Note |
|---|---|---|
| Safe default mode | `--target-policy=portable --simt-math=precise` | These are the default values |
| Pure SIMT for dav-c310 | `--target-policy=dav-c310-vec --target-recipe=none` | Does not turn on a Hybrid recipe |
| fast-SIMT | Add `--simt-math=fast` | Needs target tests |
| 950PR row-wise SIMD+SIMT Hybrid | Add `--target-recipe=dav-3510-rowwise-simd-v1` | Handles only Softmax, RMSNorm, and LayerNorm source that passes the built-in checks |

Hybrid mode needs all three options:

```bash
--target-policy=dav-c310-vec
--simt-math=fast
--target-recipe=dav-3510-rowwise-simd-v1
```

Ascify rejects the setup if any option is missing. Hybrid output also needs separate target support libraries. See the [SIMD+SIMT conversion guide](rowwise-simd-conversion.md) for the build steps.

`--frontend-compat=ascify-admitted-v1` supports a small, fixed set of front-end rules. It is not a general CUDA compatibility switch. Start with the default value, `--frontend-compat=none`, for normal input.

## 9. Common options

| Option | Use |
|---|---|
| `-o <file>` | Write one output file |
| `-o-dir <dir>` | Write several inputs to one directory |
| `-inplace` | Change the input file; makes a backup by default |
| `-no-backup` | Do not make a backup |
| `-examine` | Check and print statistics without writing converted source |
| `-print-stats` | Print conversion statistics |
| `-print-stats-csv` | Write conversion statistics as CSV |
| `-o-stats <file>` | Set the statistics output file |
| `-cuda-gpu-arch=sm_XX` | Set the CUDA GPU architecture |
| `--local-headers` | Convert directly included local headers |
| `--local-headers-recursive` | Convert all linked local header levels |
| `--migration-receipt=<file>` | Write a JSON source conversion receipt |
| `--target-policy=portable\|dav-c310-vec` | Choose the target policy |
| `--simt-math=precise\|fast` | Choose the SIMT math mode |
| `--target-recipe=none\|dav-3510-rowwise-simd-v1` | Choose the Hybrid recipe |
| `--frontend-compat=none\|ascify-admitted-v1` | Choose the front-end setting |

Show all options:

```bash
"$ASCIFY_BINARY" --help
```

## 10. Run tests

### 10.1 Run the host tests

```bash
python3 tools/check_repository.py --root .
ASCIFY_BINARY= sh tests/run_release_checks.sh
```

The first command checks repository contracts and documentation links. The second explicitly clears any previously exported `ASCIFY_BINARY` and runs host checks for conversion rules and Python tooling without invoking the real translator.

### 10.2 Run tests with the real program

```bash
ASCIFY_BINARY="$PWD/build/ascify-clang" \
ASCIFY_CUDA_PATH="$CUDA_PATH" \
ASCIFY_CLANG_RESOURCE_DIRECTORY="$PWD/ascify_install/libexec/ascify/clang/23" \
sh tests/run_release_checks.sh
```

These tests also convert real CUDA test files, so they take more time.

## 11. Build and run the converted code

Ascify does not make a full CANN project for every CUDA project. After source conversion, you still need to:

1. Install and set up the CANN Toolkit on the target machine.
2. Add `ascify_install/include` to the include path.
3. Add the host, runtime, and CANN libraries that the project needs.
4. Build the program with options for the target chip.
5. Check numbers, errors, stability, and speed on an Ascend device.

If you use `dav-3510-rowwise-simd-v1`, you must also build the four shared libraries in `runtime/dav_3510/rowwise/`. Keep both the linker-name files and the SONAME files. See the [SIMD+SIMT conversion guide](rowwise-simd-conversion.md) for the full steps.

None of these results can prove a successful device run by itself:

- Ascify returns 0.
- The JSON receipt shows `succeeded`.
- The output file is not empty.
- The build or link step passes.
- The program runs once with no separate result check.

## 12. Common problems

### No LLVM development prefix is set

Set the LLVM development prefix or source build directory:

```bash
export LLVM_BUILD_DIR=/absolute/path/to/llvm
```

### `Host compiler is not executable`

Check your explicit `ASCIFY_CC`, `ASCIFY_CXX`, `CC`, and `CXX` settings. They must name executable host compilers, either by path or as commands available in `PATH`.

### CMake cannot find LLVM or Clang

The error often names `LLVMConfig.cmake` or `ClangConfig.cmake`. Run:

```bash
find "$LLVM_BUILD_DIR" -path '*/cmake/llvm/LLVMConfig.cmake' -print
find "$LLVM_BUILD_DIR" -path '*/cmake/clang/ClangConfig.cmake' -print
```

If these files do not exist, install the development packages or use a full LLVM build directory.

### The CMake compiler check fails

Check that LLVM supports the current machine:

```bash
"$LLVM_BUILD_DIR/bin/clang" --version
"$LLVM_BUILD_DIR/bin/llvm-config" --host-target
```

You can also set `ASCIFY_CC` and `ASCIFY_CXX` to use the system compiler.

### Ninja is not found

Install `ninja-build`. You can also use Makefiles:

```bash
export CMAKE_GENERATOR="Unix Makefiles"
./build.sh
```

### `LLVM/resource config failed`

`CLANG_RESOURCE_DIRECTORY` must point to the directory above `include/`. Check this file:

```bash
test -f "$CLANG_RESOURCE_DIRECTORY/include/__clang_cuda_runtime_wrapper.h"
```

The default private resource path for the guide's Clang 23 build is:

```bash
export CLANG_RESOURCE_DIRECTORY="$PWD/ascify_install/libexec/ascify/clang/23"
```

### CUDA headers or `libdevice` are not found

Check `CUDA_PATH`:

```bash
test -f "$CUDA_PATH/include/cuda.h"
test -f "$CUDA_PATH/include/cuda_runtime.h"
test -d "$CUDA_PATH/nvvm"
```

If the machine has more than one CUDA version, set one full path.

### Project headers are not found

Put the project include paths, macros, and language version after `--`:

```bash
./run.sh input.cu -o output.cu.dpp -- \
  -std=c++17 -I/path/to/include -DMY_MACRO=1
```

### No output file is created

First, check the Ascify exit code and error output. Common causes are a source error, a missing header, a bad output path, or a receipt path that is the same as an input or output path.

Check for old files too. An old file can make a failed conversion look successful.

### The Hybrid recipe is rejected

Check these three values: `dav-c310-vec`, `fast`, and `dav-3510-rowwise-simd-v1`. You must set all three.

The source must still pass the Ascify structure checks. Unsupported source stays on the SIMT path. Do not edit the output to skip the checks.

## 13. Save information for a bug report

When you report a problem, run these commands and save their output:

```bash
git rev-parse HEAD
cmake --version | head -n 1
ninja --version
"$LLVM_BUILD_DIR/bin/clang" --version
"$LLVM_BUILD_DIR/bin/llvm-config" --host-target
readlink -f "$CUDA_PATH"
"$ASCIFY_BINARY" --help > ascify-help.txt
```

Also save the full command, standard output, error output, input file hash, output file hash, and JSON receipt. This information helps you find the step that failed.
