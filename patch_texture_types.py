#!/usr/bin/env python3
"""Append the 3-D memcpy graph-node admission to texture_types.hpp (where
cudaMemcpy3DParms and the make_cuda* helpers live). The compat header is
included at the top of this file, so the graph engine's types and the 1-D
node builder are visible."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
APPENDIX = '''
// ---- CUDA Graph emulation: memcpy node from 3-D parameters ---------------
// Admitted only with the graph engine from ascify_cuda_compat.hpp (included
// above): linear pitched copies; array sources/destinations have no graph
// representation on this platform.
inline aclError cudaGraphAddMemcpyNode(cudaGraphNode_t *node,
                                       cudaGraph_t graph,
                                       const cudaGraphNode_t *dependencies,
                                       size_t dependency_count,
                                       const cudaMemcpy3DParms *params) {
  if (node == nullptr || params == nullptr)
    return ACL_ERROR_RT_PARAM_INVALID;
  *node = {};
  if (params->srcArray != nullptr || params->dstArray != nullptr)
    return ACL_ERROR_FEATURE_UNSUPPORTED;
  if (!detail::isSupportedMemcpyKind(params->kind))
    return ACL_ERROR_RT_PARAM_INVALID;
  const size_t bytes = params->extent.width * params->extent.height *
                       params->extent.depth;
  return cudaGraphAddMemcpyNode1D(node, graph, dependencies, dependency_count,
                                  params->dstPtr.ptr, params->srcPtr.ptr,
                                  bytes, params->kind);
}
'''
target = ROOT / "include/ascify/texture_types.hpp"
text = target.read_text()
assert "cudaGraphAddMemcpyNode" not in text, "already appended"
close = "} // namespace ascify"
assert text.count(close) == 1, "unexpected namespace close count"
text = text.replace(close, APPENDIX + close)
target.write_text(text)
print("texture_types.hpp appended (inside namespace ascify)")
