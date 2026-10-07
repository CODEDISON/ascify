# NVIDIA CUDA Samples v13.4 helper fixture

`Common/helper_cuda.h` is vendored without modification from NVIDIA CUDA Samples
commit `5443602d89ed99aede2e4b7bf329daddeadb320e` (v13.4):
[pinned source](https://github.com/NVIDIA/cuda-samples/blob/5443602d89ed99aede2e4b7bf329daddeadb320e/Common/helper_cuda.h).
The original NVIDIA copyright, redistribution conditions and disclaimer remain
in the header.

The file is 28219 bytes; its SHA-256 is
`0494246768a4e3dbfcd688d79f4abad1fa39fc9f2d9dc3264a894f1c96d3785d`.
Compared with the existing `b7c5481c556c3fe98db060207ecaa41a4b9a9abc` fixture,
it adds the `0xa7` entries to the cores-per-SM and architecture-name tables.

The v13.4 `helper_string.h`, `helper_functions.h`, `helper_image.h` and
`helper_timer.h` files are byte-identical to the existing reviewed fixtures.
`check_sample_helper_profiles.py` copies that existing Common directory into
temporary test inputs and replaces only `helper_cuda.h` with this file.
