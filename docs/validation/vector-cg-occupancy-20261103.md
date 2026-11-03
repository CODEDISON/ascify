# Original vector samples: cooperative groups and occupancy

The complete original `simpleCooperativeGroups` and `simpleOccupancy` targets
from CUDA Samples v13.0 passed conversion, CANN O2 compilation, linking and NPU
correctness checks on the same Ascify product. Algorithms and workloads were
preserved. The tested source is `44896adad2bfefce3a9a3a46fa1be847f01ff6a8`.

| Original sample | Observed correctness |
| --- | --- |
| simpleCooperativeGroups | 64 threads and 256 shared bytes; block sum 2016 and four tile16 sums of 120 |
| simpleOccupancy | Original Test PASSED with manual32/automatic1024 threads; both one-million-element device copies checked independently |

Occupancy retains the original signed `i*i` overflow above46340. Its result is
fixed-toolchain low32-bit compatibility. Performance was not measured.

A fresh native build/install passed CTest4/4. The same product passed307 logical
CPU regression cases and a separate legacy32-lane host model. The complete
original source lists were built through the reviewed target adapter; upstream
NVCC CMake was not reproduced in this run.

The historical vector backlog has15 accepted names out of38 across Ascify
products. Two names have this product's NPU revalidation. `reduction`,
`threadFenceReduction`, `shfl_scan` and the dynamic cooperative-group examples
remain pending complete acceptance in this record.

[Structured results and exact argument templates](vector-cg-occupancy-20261103.json)
record the original source commit, source/program fingerprints, toolchain,
regression breakdown and numeric gates. `${ASCIFY_INSTALL}`, `${SAMPLES_ROOT}`,
`${CUDA_ROOT}`, `${CANN_ROOT}` and `${OUTPUT_ROOT}` stand for the corresponding
local dependency and output roots. Create each target's `units/000` directory
before running its conversion, compile and link arguments in order. Execute the
linked programs with their recorded empty argument lists and check the numeric
predicates above. Generated-text fingerprints identify the recorded execution.

See [the build and validation process](../release-process.md) for native
build/install commands and the distinction between host checks, conversion,
target builds and device correctness. This publication adds this summary to the
tested source; it does not describe a new device run under the documentation
commit.

The later [reduction record](reduction-20261103.md) adds one accepted name on a
separate product. The15/38 count above remains this earlier record's snapshot.
