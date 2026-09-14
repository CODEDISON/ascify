# Ascify

**将 CUDA C/C++ 源码转换到昇腾（Ascend）。**

[English](README.md) · 简体中文 · [使用手册](docs/user-guide.zh-CN.md) · [文档索引](docs/README.md)

Ascify 使用 Clang 分析 CUDA 源码，将支持的 API 和核函数结构改写为 Ascify
兼容层与 Ascend ACL 接口。`ascify-clang` 输出源码，再由目标 CANN 工具链完成
编译、链接和设备验证。

```text
CUDA 源码  →  ascify-clang  →  Ascend 兼容源码  →  CANN 编译与设备测试
```

## 能力与边界

- 转换支持范围内的 CUDA Runtime 和设备端构造，提供统计与可选的 JSON 转换回执。
- 按需递归转换本地头文件，检查依赖来源，并将输出作为一个事务发布。
- 提供 SIMT 兼容路径，以及需要显式启用的 Softmax、RMSNorm、LayerNorm SIMD+SIMT 混合路径。
- 对未支持的模式保留原源码或明确报错，具体行为由对应的转换契约定义。

源码生成只是迁移的一步。目标编译、链接、数值正确性和性能需要分别验证。
完整 CUDA Samples 应用的覆盖仍在推进；[验证矩阵](docs/validation-matrix.md)
列出已测样本范围、提交版本、性能结果和待补验证。

## 快速开始

需要 CMake 3.16.8+、C++17 编译器、Ninja、匹配的 LLVM/Clang 开发文件，以及用于
解析的 CUDA Toolkit。源码转换本身不需要 GPU 或 NPU；仓库检查和测试需要 Python 3.9+。

```bash
git clone https://github.com/Edconeone/ascify.git
cd ascify

export LLVM_BUILD_DIR=/path/to/llvm
./build.sh
cmake --install build

mkdir -p .work/examples
ascify_install/bin/ascify-clang examples/vector_add.cu \
  --cuda-path=/path/to/cuda \
  --target-policy=dav-c310-vec \
  --migration-receipt="$PWD/.work/examples/vector_add.receipt.json" \
  -o "$PWD/.work/examples/vector_add.cu.dpp" \
  -- -std=c++17
```

`LLVM_BUILD_DIR` 可以是 LLVM 开发安装目录或构建目录，必须包含 LLVM 与 Clang 的
CMake 配置。安装后的程序会从 `libexec/ascify/clang/<major>/` 查找配套的私有解析资源。自定义布局可显式设置
`--clang-resource-directory`。完整环境准备、结果检查和常见问题请看
[中文使用手册](docs/user-guide.zh-CN.md)。

## 选择转换模式

| 模式 | 选项 | 用途 |
|---|---|---|
| 保守默认模式 | `--target-policy=portable --simt-math=precise` | 默认源码转换策略 |
| 目标 SIMT | `--target-policy=dav-c310-vec` | 目标兼容层及受约束的改写 |
| SIMD+SIMT 混合模式 | 在目标 SIMT 上增加 `--simt-math=fast --target-recipe=dav-3510-rowwise-simd-v1` | 对证明满足要求的逐行算子使用混合实现；选择器未命中时使用整个算子的 SIMT 路径 |

混合模式需要另行构建 `runtime/dav_3510/rowwise/` 中的目标运行库。
支持的 shape、ABI 和验证方法见 [逐行算子转换说明](docs/rowwise-simd-conversion.md)。
源码中的 `CUDA2DPP` 名称与 `.dpp` 输出后缀是保留的 Ascend 兼容命名。

## 参与开发

```bash
python3 tools/check_repository.py --root .
ASCIFY_BINARY= sh tests/run_release_checks.sh
```

[贡献指南](CONTRIBUTING.md) 说明构建、CLI、安装、真实转换和目标设备验证的分工；
[转换参考](docs/conversion-reference.md) 记录命令行与语义边界；
[发布流程](docs/release-process.md) 说明候选版本和证据要求。

| 目录 | 使用方与职责 |
|---|---|
| `src/` | 转换器本身：Clang 分析、改写规则、命令行与统计 |
| `include/ascify/`、`acl_cub/` | 生成的 Ascend 源码使用的公开兼容头与目标契约 |
| `frontend_compat/` | 转换器解析 CUDA 输入时使用的版本化前端配置 |
| `runtime/` | 生成的混合算子链接的设备实现 |

例如，转换后的 `cudaMalloc` 调用使用 `ascify::cudaMalloc`，由
`<ascify/ascify_cuda_compat.hpp>` 提供。`ascify/` 是公开头文件的路径前缀；
编译生成代码时设置 `-I<安装目录>/include`。转换器自用的 Clang 解析头则安装在
`libexec/ascify/clang/<major>/include`，与公开兼容头分开。

## 许可证与来源

请阅读 [LICENSE](LICENSE) 和 [第三方声明](THIRD_PARTY_NOTICES.md)。带独立许可证
头的文件继续遵循其声明，包括继承的 MIT 许可转换器代码和第三方 CUDA 测试输入。
仓库与安装产物保留原始版权及来源信息。
