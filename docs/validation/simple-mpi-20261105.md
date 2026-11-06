# Original simpleMPI validation

The complete CUDA Samples v13.0 `simpleMPI` target passed conversion, O2
compilation, linking and execution on Ascend950DT with CANN 9.1.0. Both original
translation units and their header were preserved. Each of two MPI ranks used
the original grid of 10000 blocks and 256 threads, for 5120000 total float values.

The existing public Ascify production code needed no changes. The CUDA TU was
translated with precise math and compiled with CCEC. The unchanged C++ host TU
was compiled with the real OpenMPI 4.1.2 package headers and linked to its
same-version C and C++ libraries. Missing headers and launch tools were extracted
into the DT task directory; no global package installation or repair was made.
This was a complete two-TU build adapter, rather than an upstream CMake build.

One two-rank MPI invocation returned zero and printed:

```text
Running on 2 nodes
Average of square roots is: 0.667279
PASSED
```

Because the original `PASSED` has no numerical threshold, a supplemental gate
was fixed before execution. A forwarding observer captured each rank's actual
10240000-byte H2D and D2H payloads without changing the sample or substituting
results. All 5120000 outputs matched rounded float32 square roots within one ULP;
maximum absolute error was 5.960464477539063e-08. Replaying the original serial
float sums yielded 1708138.75 and 1708328.875. Their MPI float sum was 3416467.5,
and the float average 0.6672788262367249 matched the printed six significant digits.

Actual dependency views, both translation units, the complete ELF, 749 bound
inputs, capture framing and providers were checked. The two-pointer device
argument region is 16 bytes; the host launch copies 32 bytes including the packed
launch dimensions. The device was healthy and idle before and after execution.

Validated production source: `e26769084285e293179228a237eebcdf55e9bec8`.
Translator SHA256: `f3c1ef0fe1929b144b8a25b73ccc2420939e7a23a7fcf9d186b7339be86b9b3b`.
Target ELF SHA256: `a1f2221b3920870984b41be1692c50d852d288bf848fe8468294dfb15313b345`.
See the [compact acceptance record](simple-mpi-20261105.json) for source hashes,
actual PIDs and DT evidence identities. Full payloads, objects and binaries stay on DT.

The validated scope is one host with two MPI processes using the same implicit
default logical device. It supplies no multi-host network or performance result.
Strict v13.0 historical coverage is now 18/38 vector projects and 33/75 core
projects. Other historical projects retain their actual product-specific evidence.
