#!/usr/bin/env python3
"""dim3 default-member-initializer fix + full re-anchor cycle."""
import pathlib, hashlib, subprocess

ROOT = pathlib.Path(__file__).resolve().parent
HEADER = ROOT / "include/ascify/ascify_cuda_compat.hpp"

t = HEADER.read_text()
old = """#if defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)
struct cudaKernelNodeParams {
  void *func;  // converter bridge: typed <<<>>> thunk address
  ::dim3 gridDim;
  ::dim3 blockDim;
  unsigned int sharedMemBytes;
  void **kernelParams;
  void **extra;
};
#else
struct cudaGraphDim3Fallback {
  unsigned int x, y, z;
};
struct cudaKernelNodeParams {
  void *func;
  cudaGraphDim3Fallback gridDim;
  cudaGraphDim3Fallback blockDim;
  unsigned int sharedMemBytes;
  void **kernelParams;
  void **extra;
};
#endif"""
new = """#if defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)
struct cudaKernelNodeParams {
  void *func;  // converter bridge: typed <<<>>> thunk address
  // Default member initializers keep the CUDA `= {0}` aggregate spelling
  // valid: dpp's dim3 has no default constructor.
  ::dim3 gridDim{1, 1, 1};
  ::dim3 blockDim{1, 1, 1};
  unsigned int sharedMemBytes = 0;
  void **kernelParams = nullptr;
  void **extra = nullptr;
};
#else
struct cudaGraphDim3Fallback {
  unsigned int x = 1, y = 1, z = 1;
};
struct cudaKernelNodeParams {
  void *func = nullptr;
  cudaGraphDim3Fallback gridDim;
  cudaGraphDim3Fallback blockDim;
  unsigned int sharedMemBytes = 0;
  void **kernelParams = nullptr;
  void **extra = nullptr;
};
#endif"""
if t.count(old) == 1:
    t = t.replace(old, new)
    HEADER.write_text(t)
    print("DMI fix applied")
else:
    print("DMI fix already applied or drifted:", t.count(old))

subprocess.run(["python3", "/tmp/regen_ids.py"], cwd=ROOT, check=True)
NEW_SIZE = str(HEADER.stat().st_size)
NEW_SHA = hashlib.sha256(HEADER.read_bytes()).hexdigest()
OLD_SIZE = "231160"
OLD_SHA = "4686af916a387363134b3473bfa6113f633e9bb2d1306b0d3f7f7cd8bf86707b"

def reanchor(rel):
    p = ROOT / rel
    s = p.read_text()
    n_size = s.count(OLD_SIZE)
    n_sha = s.count(OLD_SHA)
    assert n_size >= 1, f"{rel}: old size missing"
    if n_size:
        s = s.replace(OLD_SIZE, NEW_SIZE)
    if n_sha:
        s = s.replace(OLD_SHA, NEW_SHA)
    p.write_text(s)
    print(f"{rel}: size x{n_size}, sha x{n_sha} -> {NEW_SIZE}/{NEW_SHA[:8]}")

reanchor("src/AscifyAction.cpp")
reanchor("tests/rewrite/check_sample_helper_compat.sh")
reanchor("tests/frontend_compat/check_frontend_host_math.py")
