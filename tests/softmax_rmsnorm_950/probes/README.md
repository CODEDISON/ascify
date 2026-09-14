# SIMT 硬件探针

本目录提供 `dav-c310-vec` 目标的五类独立硬件测量：

1. 单 block 与覆盖可用 AIV 时的 SIMT kernel 启动延迟；
2. 2 / 4 / 8 / 16 字节 load-store 宽度分别能到多少 GM 带宽；
3. 固定 grid 和 16 字节访存时，32 到 1024 threads/block 的最优点在哪里，以及
   每 AIV 2048 resident threads 的上限形态能否继续提升；
4. `asc_reduce_add` 硬件 warp 归约是否优于 5 级 `asc_shfl_xor`；
5. `expf` 与 `rsqrtf` 的整卡吞吐上限是多少。

探针结果描述当前设备、软件版本和运行参数下的启动、访存、归约和数学函数
性能。它们与 Softmax / RMSNorm 算子 benchmark 分开，也不包含 A800 对比。

与本门禁直接相关的文件：

- `simt_hw_probes.cce`：host+device 硬件探针；
- `rowwise_recipe_traits_compile.cce`：public recipe header 的 CCEC compile-only 契约；
- `rowwise_recipe_contract_probe.cce`：不启动设备 kernel 的 host runtime fallback 契约；
- `parse_results.py`：只依赖 Python 标准库的 CSV 校验与汇总器；
- `README.md`：构建、运行和字段口径。

## 构建和运行

从仓库根目录运行。脚本需要 Bash、Python 3.9 或更新版本、`sha256sum`、
`flock` 和 `npu-smi`。设置与你的设备和目标匹配的 CANN 安装路径及可写输出目录：

```bash
export CANN_ROOT=/path/to/cann
export WORK_ROOT="$(pwd -P)/.work/softmax_rmsnorm_950"
```

构建入口使用 `CANN_ROOT` 下的环境和 CCEC 编译器。当前脚本为这个混合
host/device 源文件选择 `-x dpp --cce-aicore-arch=dav-c310-vec`；具体 include、
link 参数由 [build.sh](../scripts/build.sh) 维护。仅编译探针和契约检查：

```bash
tests/softmax_rmsnorm_950/scripts/build.sh probe
```

该目标不依赖转换后的 Softmax header。二进制写入 `${WORK_ROOT}/bin/`，
编译契约对象写入 `${WORK_ROOT}/probes/`。

完整构建、运行和解析使用：

```bash
tests/softmax_rmsnorm_950/scripts/run_probes.sh
```

`run_probes.sh` 调用 `select_device.sh` 选择健康空闲设备，在持有设备锁的期间
构建，先执行 `rowwise_recipe_contract_probe`，再执行硬件探针并检查 17 行结果。
如需使用指定设备，可设置 `DEVICE`；脚本仍会执行健康和空闲检查。已构建的二进制
可通过 `SKIP_BUILD=1` 复用。产物默认写入
`${WORK_ROOT}/results/<run_id>.{build.log,run.log,csv,summary.md}`；`RUN_ID` 和
`RESULT_DIR` 可覆盖默认名称和结果目录。

首次构建检查或排错可使用较小工作量：

```bash
tests/softmax_rmsnorm_950/scripts/run_probes.sh --quick
```

探针选项会原样传给二进制；实际检测到的 AIV、warp 和 thread 配置保存在运行日志。
已有 CSV 可单独校验和汇总，不启动设备：

```bash
python3 tests/softmax_rmsnorm_950/probes/parse_results.py \
  --strict /path/to/results.csv
```

完整默认值为 256 MiB copy、5 个 timing sample、每个 sample 10 次 throughput
kernel、每 lane 256 次归约/数学调用，以及 `32 × AIV 数` 个 throughput blocks。
每行报告各 sample 的中位数与最小值。可用以下选项覆盖：

```text
--bytes-mib N
--warmup N
--iterations N
--samples N
--launch-iterations N
--inner-iterations N
--threads N
--blocks-per-aiv N
```

线程曲线始终覆盖 `32,64,128,256,512,1024,2048`。其中前六点是实际
threads/block；beta3 A5 SIMT 的合法 block 上限为 1024，ACL
`MAX_THREAD_PER_VECTOR_CORE=2048` 表示每 AIV 最大并发线程数，而不是合法的
2048-thread block。因此最后一点使用两组 1024-thread blocks，variant 明确记录为
`copy_16b_2x1024`，并将 grid 扩为两倍，让调度器暴露每 AIV 2048 resident-thread
形态。`--threads` 只控制访存宽度对比、warp reduce 和 math probe 的统一
threads/block，必须不大于 1024。

## CSV 口径

受支持目标上的完整运行输出 17 行结果，schema version 为 1：

| probe | 行数 | 主要指标 | 工作量定义 |
|---|---:|---|---|
| `launch_floor` | 2 | `ns_per_launch` | 一个可观察的 32-thread SIMT kernel |
| `copy_bandwidth` | 4 | `gbps` | source read + destination write，即 `2 × bytes` |
| `thread_scaling` | 7 | `gbps` | 32–1024 改 threads/block；2048 用 2×1024 合法 blocks |
| `warp_reduce` | 2 | `gops` | logical warp reductions/s，不是标量 add 数 |
| `math` | 2 | `gops` | 每 lane 的 `expf` 或 `rsqrtf` calls/s |

公共字段：

- `elapsed_ms_median` / `elapsed_ms_min`：一个 batch 的 ACL event 时间；
- `timed_launches`：该 batch 中的 kernel 次数；
- `traffic_bytes`：中位数 batch 对应的读写总字节数；
- `operations`：中位数 batch 中的 logical reduction 或 math call 数；
- `checksum` / `status`：计时后的小规模正确性防护。

带宽用十进制 GB/s。`parse_results.py` 会验证表头、schema、数值有限性，列出完整
thread 曲线并标记最佳点；`--format csv` 可让上层脚本继续机器处理。`--strict`
会在任意行 `status != ok` 时返回非零。

## Target recipe 编译契约

`rowwise_recipe_traits_compile.cce` 是 CCEC compile-only 门禁，`build.sh all`
和 `build.sh probe` 都会将它编译到
`${WORK_ROOT}/probes/`。其 `static_assert` 覆盖：

- FP16 storage / FP32 compute 的 direct load/store；
- compile-time affine `true/false` 与 weight accessor；
- exact-owner 对派生类继承 marker 的拒绝；
- volatile input/output pointer 的拒绝；
- double inverse-RMS 的可编译 `NotHandled` fallback；
- dispatcher ComputeType 必须精确为 float 且与 load/store marker alias 相同；
- double/custom dispatcher ComputeType 的可编译 `NotHandled` fallback；
- mismatch 分支在编译期不得实例化 load/store accessor；
- RMSNorm rows/columns 仅在原 CUDA global kernel 的 `int` 参数可无损表示时 handled；
- caller 预定义 Softmax/RMSNorm 配置宏在 public header 前后值不变；
- `ACLCUB_WARP_SIZE` 在 caller 未定义时不泄漏、预定义时按原值恢复。

`rowwise_recipe_contract_probe.cce` 在 host 侧实际检查精确 float adapter 的 null-stream
precondition：Softmax/RMS 均须返回 `handled=false, status=ACL_SUCCESS`，且不得初始化
设备或启动 kernel；同时用纯 shape contract 检查 RMSNorm 的首个 `INT_MAX+1`
row 必须回退。double/custom dispatcher mismatch 只由上面的 trait
`static_assert` 和 well-formed 调用实例化门禁隔离验证；把它与 null stream 放在同一个
runtime case 会掩盖 compute-type guard 是否真的生效，因此不把该 smoke test 表述为
mismatch runtime 证据。

## 结果解释

保留构建日志、设备信息、运行参数和原始 CSV，使不同测量可以按相同条件比较。
`--quick` 用于检查运行流程；完整测量使用上面的默认工作量或明确记录的覆盖值。
硬件探针的吞吐与算子端到端吞吐采用不同工作量定义，不能直接当作算子正确性、
完整迁移成功率或相对 A800 性能的证明。
