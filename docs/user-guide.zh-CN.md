# Ascify 使用手册

[English](user-guide.en.md) | 简体中文

## 1. Ascify 是什么

Ascify 是一个 CUDA C/C++ 到昇腾（Ascend）的源码转换工具。它使用 Clang 读取 CUDA 源码，将支持的 CUDA API 和核函数结构转换为 Ascify 兼容层和 Ascend ACL 接口。

完整流程有两个部分：

```text
CUDA 源码
  -> ascify-clang 转换源码
  -> Ascend 兼容源码
  -> CANN 或目标工程编译、链接和设备测试
```

请分清下面两个结果：

| 结果 | 含义 |
|---|---|
| Ascify 返回 0，并生成非空文件 | 源码转换完成 |
| 生成的代码通过编译、链接和设备测试 | 目标程序通过验证 |

源码转换不需要 NVIDIA GPU，也不需要昇腾 NPU。但转换机器上要有完整的 CUDA Toolkit 目录。要编译和运行生成的代码，你还需要 CANN、Ascify 头文件、目标库和昇腾设备。

## 2. 环境要求

### 2.1 必需软件

| 软件 | 用途 | 说明 |
|---|---|---|
| Git | 下载源码 | 使用近期版本即可 |
| Bash | 运行 `build.sh` 和 `run.sh` | 这两个脚本使用 Bash |
| CMake | 配置项目 | Ascify 要求 3.16.8 或更高版本 |
| Ninja | 构建项目 | `build.sh` 默认使用 Ninja |
| C/C++ 编译器 | 编译 Ascify | 编译器要支持当前主机 |
| LLVM 和 Clang 开发文件 | 提供 Clang 前端、库和 CMake 配置 | 只有 `clang` 命令通常不够 |
| CUDA Toolkit | 提供 CUDA 头文件和 `libdevice` | 不要求有 NVIDIA GPU |

编译 Ascify 不需要 Python。仓库检查、转换封装脚本和测试需要 Python 3.9 或更高版本。

以下是历史验证中记录的环境；其结果归属和当前候选状态见 [验证矩阵](validation-matrix.md)：

| 项目 | 测试值 |
|---|---|
| 系统 | Linux AArch64 |
| LLVM/Clang | 23.0.0git，commit `caf619642a6dbb216969a9450d33dbac5a8d30df` |
| CUDA Toolkit | 12.8 |
| CMake | 4.3.4 |
| Ninja | 1.13.0 |

其他版本也可能可用。上表只给出已经测试过的版本。

### 2.2 安装基础工具

Ubuntu 或 Debian 用户可以运行：

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git ninja-build python3
```

其他 Linux 发行版请安装同类软件包。安装后运行：

```bash
git --version
cmake --version
ninja --version
python3 --version
```

如果 CMake 低于 3.16.8，请先升级。LLVM 也可能要求更高版本的 CMake。请按你所用 LLVM 版本的要求准备环境。

## 3. 准备 LLVM 和 Clang

可以直接把 `LLVM_BUILD_DIR` 设置为包含 `LLVMConfig.cmake` 和
`ClangConfig.cmake` 的 LLVM 开发安装前缀。`build.sh` 不要求存在 LLVM 源码目录。
下面以已有源码构建目录为例。

### 3.1 使用已有的 LLVM 构建目录

最快的做法是使用已经编译好的 `llvm-project`。目录中要有这些文件：

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

设置路径：

```bash
export LLVM_PROJECT_PATH=/path/to/llvm-project
export LLVM_BUILD_DIR="$LLVM_PROJECT_PATH/build"
```

检查文件和工具：

```bash
test -x "$LLVM_BUILD_DIR/bin/clang"
test -x "$LLVM_BUILD_DIR/bin/clang++"
test -f "$LLVM_BUILD_DIR/lib/cmake/llvm/LLVMConfig.cmake"
test -f "$LLVM_BUILD_DIR/lib/cmake/clang/ClangConfig.cmake"

"$LLVM_BUILD_DIR/bin/clang" --version
"$LLVM_BUILD_DIR/bin/llvm-config" --host-target
```

最后一条命令要显示当前主机的 target triple。如果没有输出，CMake 的编译器检查可能会失败。如果 LLVM 没有当前主机的后端，这项检查也可能失败。

### 3.2 从源码编译 LLVM

如果机器上没有合适的 LLVM，你可以从源码编译。LLVM 的编译会使用很多时间、内存和磁盘。开始前请看 [LLVM CMake 构建说明](https://llvm.org/docs/CMake.html)。

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

# 可选：使用 Ascify 已测试过的 LLVM commit。
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
  *) echo "请为当前主机选择 LLVM target" >&2; exit 1 ;;
esac

cmake -S llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_TARGETS_TO_BUILD="${LLVM_HOST_TARGET};NVPTX" \
  -DLLVM_DEFAULT_TARGET_TRIPLE="${LLVM_HOST_TRIPLE}"

cmake --build build --parallel 2
cd ..
```

编译完整默认目标会生成 Ascify 链接所需的 Clang 开发库；仅编译 `clang` 可执行文件不足以准备全部开发依赖。编译完成后设置：

```bash
export LLVM_PROJECT_PATH=/path/to/llvm-project
export LLVM_BUILD_DIR="$LLVM_PROJECT_PATH/build"
```

你也可以使用 Linux 发行版提供的 LLVM 开发包。`LLVM_BUILD_DIR` 应指向包含 LLVM 与 Clang CMake 配置、开发库和匹配解析资源的前缀。主机编译器可以位于其他目录，通过 `ASCIFY_CC`/`ASCIFY_CXX` 或 `CC`/`CXX` 指定；未指定时，脚本优先查找该前缀中的 Clang，再由 CMake 查找。如果缺少 `ClangConfig.cmake`，请安装匹配的 Clang 开发包，或使用完整的 LLVM 源码构建目录。

## 4. 准备 CUDA Toolkit

Ascify 使用 CUDA Toolkit 来解析 CUDA 源码。转换机器不需要 NVIDIA GPU。

请安装适合当前主机的 CUDA Toolkit。你也可以使用已有的完整 Toolkit 目录。安装方法请看 [CUDA Linux 安装手册](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)。常见路径是 `/usr/local/cuda`：

```bash
export CUDA_PATH=/usr/local/cuda

test -f "$CUDA_PATH/include/cuda.h"
test -f "$CUDA_PATH/include/cuda_runtime.h"
test -d "$CUDA_PATH/nvvm"
```

`CUDA_PATH` 要指向 Toolkit 根目录，不要指向 `include` 目录。只有头文件、没有 `nvvm` 或 `libdevice` 的目录可能无法工作。

## 5. 下载、编译和安装 Ascify

### 5.1 下载源码

```bash
git clone https://github.com/Edconeone/ascify.git
cd ascify

git rev-parse HEAD
git status --short --branch
```

请保存 commit 值。以后可以用它重复同一次转换。

### 5.2 编译

先将 `LLVM_BUILD_DIR` 设为第 3 节准备的开发安装前缀或源码构建目录。只需要这个变量，不要求同时设置 `LLVM_PROJECT_PATH`：

```bash
export LLVM_BUILD_DIR=/path/to/llvm
export ASCIFY_BUILD_JOBS=2

./build.sh
```

`build.sh` 默认使用下面的设置：

- 构建工具：Ninja。
- 构建类型：Release。
- 构建目录：`build/`。
- 安装目录：`ascify_install/`。
- C/C++ 编译器：优先使用 `ASCIFY_CC`/`ASCIFY_CXX` 或 `CC`/`CXX`；未设置时使用 LLVM 前缀中的 Clang，再由 CMake 查找主机编译器。
- 链接器：由编译器选择；需要指定时设置 `ASCIFY_LINKER`。

检查生成的程序：

```bash
test -x build/ascify-clang
build/ascify-clang --help >/dev/null
```

如果 LLVM 中的 `clang` 不能编译当前主机程序，请使用系统编译器：

```bash
export ASCIFY_CC=/usr/bin/clang
export ASCIFY_CXX=/usr/bin/clang++
./build.sh
```

你也可以改构建目录、安装目录、并行任务数和生成器：

```bash
export BUILD_DIR="$PWD/build-release"
export INSTALL_ROOT="$PWD/ascify-install-release"
export ASCIFY_BUILD_JOBS=4
export CMAKE_GENERATOR=Ninja
./build.sh
```

如果你改了这些路径，后面的命令也要使用新路径。

### 5.3 安装

如果你使用默认构建目录，请运行：

```bash
cmake --install build

test -x ascify_install/bin/ascify-clang
test -f ascify_install/libexec/ascify/clang/23/include/__clang_cuda_runtime_wrapper.h
test -f ascify_install/include/ascify/ascify_cuda_compat.hpp
```

安装命令会复制 Ascify 程序、兼容头、前端配置、Clang 解析头和许可证声明。
两类头文件分别安装：

| 安装路径 | 使用方 |
|---|---|
| `include/ascify/` 和 `include/acl_cub/` | 生成的 Ascend 源码；编译时使用 `-I<安装目录>/include` |
| `libexec/ascify/clang/<major>/include/` | 转换器自用的 Clang 解析资源 |

本手册命令使用已测试的 Clang 主版本 `23`，其他工具链请替换为实际 resource 版本。
resource-directory 选项应指向 `libexec/ascify/clang/<major>`，即其 `include/` 的父目录。
未显式覆盖时，安装后的程序会自动发现这个位置。

不要只把 `ascify-clang` 复制到另一台机器。程序还需要匹配的 resource headers。它也可能需要 LLVM 共享库。

## 6. 完成第一次转换

下面的命令会转换 FP32 示例 `examples/vector_add.cu`。它使用纯 SIMT 模式，并生成一个 JSON 回执。

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

如果 CUDA Toolkit 不在 `/usr/local/cuda`，请改为实际路径。`run.sh` 会读取这些环境变量并调用 `ASCIFY_BINARY`。安装后的程序可以自动发现解析资源；`CLANG_RESOURCE_DIRECTORY` 用于显式覆盖该位置。

命令末尾的 `--` 用来分开 Ascify 选项和 Clang 选项。这里的 `-std=c++17` 会传给 Clang。

### 6.1 检查结果

如果转换命令没有报错，请检查输出文件：

```bash
test -s .work/examples/vector_add.cu.dpp
test -s .work/examples/vector_add.receipt.json
```

再检查主要改写和回执状态：

```bash
grep -E 'ascify_cuda_compat|acl/acl.h|aclrtMalloc' \
  .work/examples/vector_add.cu.dpp

grep -Eq '"status"[[:space:]]*:[[:space:]]*"succeeded"' \
  .work/examples/vector_add.receipt.json
```

回执中的 `succeeded` 只表示源码转换完成。它不表示目标编译、链接、设备运行、数值检查或性能测试已经通过。

### 6.2 不安装时如何转换

如果你只编译了 Ascify，没有运行安装命令，请使用 LLVM 的 resource directory：

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

安装目录更适合长期使用。它也更容易记录和检查。

## 7. 转换自己的源码

### 7.1 转换一个文件

```bash
./run.sh /path/to/project/kernel.cu \
  --target-policy=dav-c310-vec \
  --simt-math=precise \
  --target-recipe=none \
  -o "$PWD/.work/examples/kernel.cu.dpp" \
  -- -std=c++17 -I/path/to/project/include -DMY_FEATURE=1
```

请把原项目需要的 include 路径、宏、语言标准和 sysroot 传给 Clang。把这些选项放在 `--` 后面。

原源码可以由 `nvcc` 编译，不代表 Ascify 一定能直接解析。如果缺少 `-I`、`-D` 或生成的头文件，Ascify 仍会失败。

### 7.2 一起转换本地头文件

Ascify 默认只转换输入文件。你可以让它一起转换双引号引用的本地头文件：

- `--local-headers`：只处理输入文件直接引用的本地头文件。
- `--local-headers-recursive`：继续处理这些头文件引用的本地头文件。

```bash
./run.sh /path/to/project/kernel.cu \
  --local-headers-recursive \
  --target-policy=dav-c310-vec \
  --simt-math=precise \
  -o "$PWD/.work/examples/kernel.cu.dpp" \
  -- -std=c++17 -I/path/to/project/include
```

转换后的头文件会放在 `.work/examples/kernel.cu.dpp.headers/`。这个选项不会处理所有系统头文件，也不会增加 Ascify 支持的 CUDA API。详细规则请看 [本地头文件说明](local-header-closure.md)。

### 7.3 直接运行 `ascify-clang`

你也可以不用 `run.sh`：

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

## 8. 选择转换模式

第一次转换时，建议使用 `dav-c310-vec + precise + target-recipe=none`。先检查基本流程。基本流程通过后，再试其他模式。

| 目标 | 选项 | 说明 |
|---|---|---|
| 保守模式 | `--target-policy=portable --simt-math=precise` | 这是默认设置 |
| dav-c310 纯 SIMT | `--target-policy=dav-c310-vec --target-recipe=none` | 不启用 Hybrid recipe |
| fast-SIMT | 再加 `--simt-math=fast` | 需要做目标测试 |
| 950PR row-wise SIMD+SIMT Hybrid | 再加 `--target-recipe=dav-3510-rowwise-simd-v1` | 只处理通过内置检查的 Softmax、RMSNorm 和 LayerNorm 源码 |

Hybrid 模式需要下面三个选项：

```bash
--target-policy=dav-c310-vec
--simt-math=fast
--target-recipe=dav-3510-rowwise-simd-v1
```

如果缺少任何一项，Ascify 会拒绝这个设置。Hybrid 输出还需要单独的目标支持库。构建方法请看 [SIMD+SIMT 转换手册](rowwise-simd-conversion.md)。

`--frontend-compat=ascify-admitted-v1` 只支持一小组已定义的前端兼容规则。它不是通用 CUDA 兼容开关。普通输入请先使用默认值 `--frontend-compat=none`。

## 9. 常用选项

| 选项 | 用途 |
|---|---|
| `-o <file>` | 输出一个文件 |
| `-o-dir <dir>` | 把多个输入写到一个目录 |
| `-inplace` | 改写原文件；默认会备份 |
| `-no-backup` | 不创建备份 |
| `-examine` | 只检查并打印统计，不写转换结果 |
| `-print-stats` | 打印转换统计 |
| `-print-stats-csv` | 写出 CSV 统计 |
| `-o-stats <file>` | 指定统计文件 |
| `-cuda-gpu-arch=sm_XX` | 指定 CUDA 架构 |
| `--local-headers` | 转换直接引用的本地头文件 |
| `--local-headers-recursive` | 转换所有相连的本地头文件 |
| `--migration-receipt=<file>` | 写出 JSON 源码转换回执 |
| `--target-policy=portable\|dav-c310-vec` | 选择目标策略 |
| `--simt-math=precise\|fast` | 选择 SIMT 浮点模式 |
| `--target-recipe=none\|dav-3510-rowwise-simd-v1` | 选择 Hybrid recipe |
| `--frontend-compat=none\|ascify-admitted-v1` | 选择前端兼容设置 |

查看所有选项：

```bash
"$ASCIFY_BINARY" --help
```

## 10. 运行测试

### 10.1 运行 host 测试

```bash
python3 tools/check_repository.py --root .
ASCIFY_BINARY= sh tests/run_release_checks.sh
```

第一条命令检查仓库契约和文档链接。第二条命令显式清空先前设置的 `ASCIFY_BINARY`，执行 host 测试；这组测试检查转换规则和 Python 工具，不调用真实转换器。

### 10.2 使用真实程序运行测试

```bash
ASCIFY_BINARY="$PWD/build/ascify-clang" \
ASCIFY_CUDA_PATH="$CUDA_PATH" \
ASCIFY_CLANG_RESOURCE_DIRECTORY="$PWD/ascify_install/libexec/ascify/clang/23" \
sh tests/run_release_checks.sh
```

这组测试还会转换真实 CUDA fixture，所以需要更多时间。

## 11. 编译和运行生成的代码

Ascify 不会为每个 CUDA 项目自动创建完整的 CANN 工程。源码转换后，你还要完成这些工作：

1. 在目标机器安装并初始化 CANN Toolkit。
2. 把 `ascify_install/include` 加到 include 路径。
3. 加入原项目需要的 host、runtime 和 CANN 库。
4. 使用目标芯片的编译选项构建程序。
5. 在昇腾设备上检查数值、错误处理、稳定性和性能。

如果你使用 `dav-3510-rowwise-simd-v1`，还要构建 `runtime/dav_3510/rowwise/` 中的四个共享库。链接名文件和 SONAME 文件都要保留。详细步骤请看 [SIMD+SIMT 转换手册](rowwise-simd-conversion.md)。

下面这些结果都不能单独证明设备运行成功：

- Ascify 返回 0。
- JSON 回执显示 `succeeded`。
- 生成文件不是空文件。
- 编译或链接通过。
- 程序只运行一次，而且没有单独的数值检查。

## 12. 常见问题

### 没有指定 LLVM 开发文件

设置 LLVM 开发安装前缀或源码构建目录：

```bash
export LLVM_BUILD_DIR=/absolute/path/to/llvm
```

### 提示 `Host compiler is not executable`

检查显式设置的 `ASCIFY_CC`、`ASCIFY_CXX`、`CC` 和 `CXX`。它们需要指向可执行的主机编译器，或是在 `PATH` 中可以找到的命令。

### CMake 找不到 LLVM 或 Clang

错误信息通常会提到 `LLVMConfig.cmake` 或 `ClangConfig.cmake`。运行：

```bash
find "$LLVM_BUILD_DIR" -path '*/cmake/llvm/LLVMConfig.cmake' -print
find "$LLVM_BUILD_DIR" -path '*/cmake/clang/ClangConfig.cmake' -print
```

如果这两个文件不存在，请安装开发包，或使用完整的 LLVM 构建目录。

### CMake 的编译器检查失败

检查 LLVM 是否支持当前主机：

```bash
"$LLVM_BUILD_DIR/bin/clang" --version
"$LLVM_BUILD_DIR/bin/llvm-config" --host-target
```

你也可以设置 `ASCIFY_CC` 和 `ASCIFY_CXX`，然后使用系统编译器。

### 找不到 Ninja

请安装 `ninja-build`。你也可以使用 Makefiles：

```bash
export CMAKE_GENERATOR="Unix Makefiles"
./build.sh
```

### 提示 `LLVM/resource config failed`

`CLANG_RESOURCE_DIRECTORY` 要指向 `include/` 的上一级目录。检查下面的文件：

```bash
test -f "$CLANG_RESOURCE_DIRECTORY/include/__clang_cuda_runtime_wrapper.h"
```

本手册 Clang 23 构建的默认私有资源路径是：

```bash
export CLANG_RESOURCE_DIRECTORY="$PWD/ascify_install/libexec/ascify/clang/23"
```

### 找不到 CUDA 头文件或 `libdevice`

检查 `CUDA_PATH`：

```bash
test -f "$CUDA_PATH/include/cuda.h"
test -f "$CUDA_PATH/include/cuda_runtime.h"
test -d "$CUDA_PATH/nvvm"
```

如果机器上有多个 CUDA 版本，请直接设置一个完整路径。

### 找不到项目头文件

把项目的 include 路径、宏和语言标准放在 `--` 后面：

```bash
./run.sh input.cu -o output.cu.dpp -- \
  -std=c++17 -I/path/to/include -DMY_MACRO=1
```

### 没有生成输出文件

先看 Ascify 的退出码和错误输出。常见原因有语法错误、缺少头文件、输出路径无效，或回执路径与输入/输出路径相同。

也要确认目录中没有旧文件。旧文件可能让你误以为本次转换成功。

### Hybrid recipe 被拒绝

检查下面三个值：`dav-c310-vec`、`fast` 和 `dav-3510-rowwise-simd-v1`。三个值都要设置。

设置正确后，源码仍要通过 Ascify 的结构检查。不支持的源码会留在 SIMT 路径。不要手动改生成文件来跳过检查。

## 13. 保存问题信息

报告问题时，请运行并保存这些命令的输出：

```bash
git rev-parse HEAD
cmake --version | head -n 1
ninja --version
"$LLVM_BUILD_DIR/bin/clang" --version
"$LLVM_BUILD_DIR/bin/llvm-config" --host-target
readlink -f "$CUDA_PATH"
"$ASCIFY_BINARY" --help > ascify-help.txt
```

也请保存原命令、标准输出、错误输出、输入文件哈希、输出文件哈希和 JSON 回执。这些信息可以帮助你找到失败的步骤。
