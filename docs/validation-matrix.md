# Validation matrix

Evidence reviewed on 2026-09-13 without rerunning conversion or device tests.
The [machine-readable baselines](validation-baselines.json) contain full source
commits, evidence identifiers, hashes, and counts. Raw logs remain in the
project evidence workspace; they are not bundled with the source repository.

## Softmax and RMSNorm phase-one acceptance

The accepted source is `32b33a3c8b33a379cf862f20afd056120015baad`.
Execution used the SIMD+SIMT Hybrid path on 950PR with CANN 9.1.0-beta.3.
For each shape, the ratio is `A800 median latency / 950PR median latency`,
equivalently the logical-bandwidth ratio when the byte count is identical.
Each group independently requires an **arithmetic mean strictly above 0.80**.

| Group | Performance records | Mean relative to A800 | Gate |
|---|---:|---:|---|
| Softmax | 1000 | 88.288626% | Passed |
| RMSNorm plain | 1000 | 82.673452% | Passed |
| RMSNorm affine | 1000 | 81.683271% | Passed |

All 3000 timing records completed with zero failures or skips, using
5 warmups, 20 samples, and 1 inner iteration. This is an aggregate criterion;
it does not assert that every shape exceeds 80%, or that pure SIMT achieves
these Hybrid results.

The denominator is the original 1000 indexed cases in each frozen manifest.
It is not a deduplicated geometry count: there are 627 distinct `(rows, cols)`
pairs for Softmax and 562 for each RMSNorm group. Preserve the original case
population and weighting when reproducing this acceptance result.

Correctness for the same accepted candidate includes 4000 ordinary cases
across Hybrid and SIMT (each mode: Softmax 1000, RMSNorm plain 500, affine 500),
plus 66 large-shape cases per mode (33 shapes in each RMSNorm variant), all
with full-element checking. The 33 shapes are included in the full1000
performance population; this result is not the older 967-shape result.

The later local candidate `721ca1fa45568337b4865ce62f331ec45eeaae60` retains
the measured row-wise recipe, target runtime, and operator harness sources,
but changes shared translation and compatibility code. The table is historical
acceptance for `32b33a3`, not a hardware validation of the later candidate.

## CUDA Samples coverage

All populations below use CUDA Samples v13.3 source
`b7c5481c556c3fe98db060207ecaa41a4b9a9abc`. Counts are **source files**,
not independent operators or fully migrated Sample projects.

| Population / Ascify source | Source generation | Whole-file O2 object compilation | Full Sample link/run |
|---|---:|---:|---|
| All 189 CUDA files / `ac3dced` | 100/189 (52.91%) | 950PR: 25/189 (13.23%) | Not measured |
| Previously generated 100 CUDA files / `ac3dced` | DT: 100/100 | DT: 23/100 | Not measured in this batch |
| Fixed phase-one 20 files / `32b33a3` | 13/20 (65%) | DT and 950PR: 7/20 (35%) | Not measured |
| Fixed phase-one 20 files / `6e9da55` | 18/20 (90%) | DT: 10/20 (50%) | Not measured |
| Current local candidate / `721ca1f` | No new full-population measurement | DT validation pending | Not measured |

The 100-file DT replay is a selected successful subset, not a full-repository
100% conversion result. Do not combine different commits or platforms into a
new success set. Repository organization and CI alone establish no new
conversion rate for either population.

Separate earlier DT device probes for SimpleAtomic and VectorAdd linked but
failed during execution. They do not establish successful complete-Sample
migration and are not part of the selected100 object-compilation batch.

Generation requires a real successful exit, nonempty output, and successful
global/per-input migration receipts. Object compilation requires a successful
target compiler invocation and a valid nonempty relocatable ELF object.
Neither establishes successful linking, numerical correctness, or performance.

The full inventory has 213 C++/CUDA Sample projects; 175 contain CUDA sources.
The same generation run produced every CUDA file in 87 projects and every
tested C/C++/CUDA source in 58 projects. These remain source-generation counts.
The 200 host C/C++ files are a separate population: 74 generated, of which 43
compiled on 950PR. The 213-project inventory includes two LLVM-IR-only
projects with no tested C/C++ inputs (211 projects have tested source files);
33 Python examples are excluded.

## Remaining generalization work

Against the historical full inventory, 89/189 CUDA files still failed source
generation and 164/189 lacked a passing 950PR object result. The first-error
diagnostic categories for the 89 failures were 58 missing headers/dependencies,
18 declarations/API/include context, 7 compatibility-profile rejections,
5 overload/type mismatches, and 1 other frontend error. These are diagnostic
categories, not proof that supplying headers makes the files work.

For the more recent fixed20 DT run, 2 files failed conversion (Reduction and
TileVectorAdd), 8 generated but failed object compilation, and 10 compiled.
The unvalidated `721ca1f` helper, private-copy, and scratch-reduction work does
not change those counts. Device FP64 remains outside phase-one scope.

Phase two must freeze its source and dependency manifests and record separate
generation, object, link, run, and correctness outcomes. Start with dependency
and include-context reproducibility, then shared APIs and semantic adapters.
Library-backed samples, CUDA Tile, and platform/runtime integration need their
own support decisions and validation. No complete-Sample conversion percentage
or defensible schedule for universal support has yet been measured.
