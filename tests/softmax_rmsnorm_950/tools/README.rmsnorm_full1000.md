# RMSNorm full1000 inputs and bounded host correctness

The exporter accepts 1000 RMSNorm shape entries and 2000 A800 measurements
(plain and affine). It retains the full population, including the 33 entries
(66 measurement records) whose element counts exceed the earlier 32-bit product
limit. The store adapter in `inputs/rmsnorm_affine_store.cuh` uses 64-bit
addressing, including its multiplication operands, for converted fallback
execution.

Address width does not add device FP64 support or change the RMSNorm computation.
The source dimensions must still each fit the original kernel's `int` parameters.

## Export the complete existing corpus

Use the original, unfiltered files. The exporter checks all 2000 identities,
variants, 100-iteration protocol, finite positive metrics, and logical byte counts
before creating a new output directory. It preserves raw A800 bytes and original
idx, shape, scenario and duplicate geometry; it does not inspect performance to
select cases.

```sh
python3 -B tests/softmax_rmsnorm_950/tools/prepare_rmsnorm_full1000.py \
  --shapes /path/to/original/rmsnorm_shapes.csv \
  --a800-results /path/to/original/results_rmsnorm_fp16.csv \
  --out-dir /path/to/new/full1000_inputs
```

`shapes_rmsnorm_full1000_pairs2000.csv` enables correctness and performance for
both variants. `shapes_rmsnorm_previous_exclusions.csv` retains the original 33
entries, paired into 66 rows, for focused restoration checks. `manifest.json`
records SHA-256 identities, row counts and required correctness arguments.
Neither file is evidence that a device test has run.

## Check the full device shape with bounded host buffers

The checker uploads whole rows in blocks, executes the full original device
shape once, then reads every input, output and inverse-RMS element back in blocks.
It regenerates the deterministic host input from global linear
indices and uses an FP32 oracle with FP16 roundings and a fixed comparison order
within each row. Input/weight immutability, all output/inverse canaries, allocation
guards, nonfinite values and error maxima remain checked.

The default `--host-chunk-elements 1048576` bounds tensor host storage at roughly
10 MiB plus inverse and per-column weight arrays for this corpus. One whole row
is the minimum block size. Device allocation size and launch dimensions are
unchanged; the largest original pair requires approximately 29.813 GiB HBM.
`full oracle: ... elements=... rows=...` is printed only after complete readback
and comparison. It records coverage separately from the numeric pass/fail result.

The `--max-elements` limit is explicit. To include the actual full
corpus, use the maximum recorded by the exporter (8,000,000,000 for the frozen
original data), not the smoke limit:

```sh
/path/to/rmsnorm_check_production_fast \
  --shapes /path/to/new/full1000_inputs/shapes_rmsnorm_full1000_pairs2000.csv \
  --tier full --device 0 --run-id full1000 \
  --out /path/to/new/results/full_accuracy.csv \
  --max-elements 8000000000 --host-chunk-elements 1048576 \
  --abs-tol 5e-3 --scaled-rel-tol 2e-3 \
  --inv-abs-tol 2e-3 --inv-scaled-rel-tol 2e-3
```

Use [select_device.sh](../scripts/select_device.sh) to select and lock the
device before this invocation. A complete corpus check requires 2000 selected and passed records, zero failures/skips,
and complete per-shape coverage logs. Check the selected, passed, failed, and skipped counts: exit code zero alone
does not prove no cases were skipped.
Record the source commit, generated-input hashes, CANN version, device model,
and run parameters with the results. Rebuild the checker, regenerate the adapter
with the recipe, and confirm its adapter traits before device execution.
The matcher permits integral widths of at least 32 bits; actual converter and
CCEC validation remain separate from the host tests below.

## Host regression tests

```sh
python3 -B tests/softmax_rmsnorm_950/tools/test_rmsnorm_full1000_host.py -v
python3 -B tests/softmax_rmsnorm_950/tools/test_prepare_rmsnorm_full1000.py -v
```

The first command compiles the actual store adapter and checker source with
UBSan. Sparse virtual memory allows real adapter stores around 2^31 and ~8e9
element offsets without allocating a full physical host tensor. It covers
pack1/pack2 and plain/affine. An ACL/CUDA host test double validates complete
multi-block iteration, the final short block, and failures in output, input,
inverse, weight, guard, nonfinite/canary and partial readback paths. These tests
exercise checker logic and host address arithmetic, not CCEC code generation,
NPU memory behavior or device correctness.
