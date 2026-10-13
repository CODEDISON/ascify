# NVIDIA CUDA Samples v13.0 helper dependency

`Common/helper_string.h` is copied unchanged from NVIDIA cuda-samples tag
`v13.0`, commit `3f1c50965017932fc81e6d94a3fc9e04c105b312`:
<https://github.com/NVIDIA/cuda-samples/blob/3f1c50965017932fc81e6d94a3fc9e04c105b312/Common/helper_string.h>.
It retains its original NVIDIA license. Size: 14,875 bytes; SHA-256:
`616ac8b2588a6e73aed207f85faf0da5e617404390df1790f6637b1042dbcc94`.

The profile test composes this file with the byte-identical v13.3
`helper_cuda.h`, `helper_functions.h`, `helper_image.h`, and `helper_timer.h`
from `../nvidia_samples/Common`. It changes only copies in its evidence
directory for negative identity cases. The actual source fixture is immutable.

The difference from the newer helper_string.h consists of `sdkFindFilePath`
search-path strings. Retaining the original header is essential: conversion
must not replace these old `Samples/` paths with the newer `cpp/` paths.
