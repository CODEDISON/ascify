#!/usr/bin/env python3
"""Apply the CUDA Graph host-emulation converter-side changes.

Text-anchored edits with exact-match assertions; aborts cleanly if the
tree drifted (another agent mid-edit). Run from source_vector_next root.
"""
import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
FILES = {}

def load(rel):
    p = ROOT / rel
    FILES[rel] = p.read_text()
    return FILES[rel]

def patch(rel, old, new, count=1):
    text = FILES[rel]
    found = text.count(old)
    assert found == count, f"{rel}: anchor found {found}x (want {count}):\n{old[:200]}"
    FILES[rel] = text.replace(old, new)
    print(f"  patched {rel}: {old.splitlines()[0][:70]}...")

def save(rel):
    (ROOT / rel).write_text(FILES[rel])
    print(f"saved {rel}")

GRAPH_FUNCS_ANCHOR = '''  m["cudaLaunchHostFunc"] = {"ascify::cudaLaunchHostFunc", CONV_EXECUTION, API_RUNTIME, 0};'''
GRAPH_FUNCS_NEW = GRAPH_FUNCS_ANCHOR + '''
  // CUDA Graph host-emulation line (simpleCudaGraphs / graphMemoryNodes):
  // the compat layer records and replays graphs entirely on the host.
  m["cudaGraphCreate"] = {"ascify::cudaGraphCreate", CONV_EXECUTION, API_RUNTIME, 0};
  m["cudaGraphDestroy"] = {"ascify::cudaGraphDestroy", CONV_EXECUTION, API_RUNTIME, 0};
  m["cudaGraphAddKernelNode"] = {"ascify::cudaGraphAddKernelNode", CONV_EXECUTION, API_RUNTIME, 0};
  m["cudaGraphAddMemcpyNode"] = {"ascify::cudaGraphAddMemcpyNode", CONV_EXECUTION, API_RUNTIME, 0};
  m["cudaGraphAddMemcpyNode1D"] = {"ascify::cudaGraphAddMemcpyNode1D", CONV_EXECUTION, API_RUNTIME, 0};
  m["cudaGraphAddMemsetNode"] = {"ascify::cudaGraphAddMemsetNode", CONV_EXECUTION, API_RUNTIME, 0};
  m["cudaGraphAddHostNode"] = {"ascify::cudaGraphAddHostNode", CONV_EXECUTION, API_RUNTIME, 0};
  m["cudaGraphAddMemAllocNode"] = {"ascify::cudaGraphAddMemAllocNode", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaGraphAddMemFreeNode"] = {"ascify::cudaGraphAddMemFreeNode", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaGraphGetNodes"] = {"ascify::cudaGraphGetNodes", CONV_EXECUTION, API_RUNTIME, 0};
  m["cudaGraphInstantiate"] = {"ascify::cudaGraphInstantiate", CONV_EXECUTION, API_RUNTIME, 0};
  m["cudaGraphLaunch"] = {"ascify::cudaGraphLaunch", CONV_EXECUTION, API_RUNTIME, 0};
  m["cudaGraphClone"] = {"ascify::cudaGraphClone", CONV_EXECUTION, API_RUNTIME, 0};
  m["cudaGraphExecDestroy"] = {"ascify::cudaGraphExecDestroy", CONV_EXECUTION, API_RUNTIME, 0};
  m["cudaStreamBeginCapture"] = {"ascify::cudaStreamBeginCapture", CONV_EXECUTION, API_RUNTIME, 0};
  m["cudaStreamEndCapture"] = {"ascify::cudaStreamEndCapture", CONV_EXECUTION, API_RUNTIME, 0};
  m["cudaMallocManaged"] = {"ascify::cudaMallocManaged", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaDriverGetVersion"] = {"ascify::cudaDriverGetVersion", CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};'''

GRAPH_TYPES_ANCHOR = '''  m["cudaMemcpyDefault"] = {"ascify::cudaMemcpyDefault", CONV_TYPE, API_RUNTIME, SEC::MEMORY};'''
GRAPH_TYPES_NEW = GRAPH_TYPES_ANCHOR + '''
  // CUDA Graph host-emulation handles and node-parameter records: host-side
  // descriptors owned by the compat layer (no kernel-observed layout).
  m["cudaGraph_t"] = {"ascify::cudaGraph_t", CONV_TYPE, API_RUNTIME, SEC::STREAM};
  m["cudaGraphExec_t"] = {"ascify::cudaGraphExec_t", CONV_TYPE, API_RUNTIME, SEC::STREAM};
  m["cudaGraphNode_t"] = {"ascify::cudaGraphNode_t", CONV_TYPE, API_RUNTIME, SEC::STREAM};
  m["cudaKernelNodeParams"] = {"ascify::cudaKernelNodeParams", CONV_TYPE, API_RUNTIME, SEC::STREAM};
  m["cudaMemsetParams"] = {"ascify::cudaMemsetParams", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaHostNodeParams"] = {"ascify::cudaHostNodeParams", CONV_TYPE, API_RUNTIME, SEC::STREAM};
  m["cudaMemAllocNodeParams"] = {"ascify::cudaMemAllocNodeParams", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaMemPoolProps"] = {"ascify::cudaMemPoolProps", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaMemAllocationTypePinned"] = {"ascify::cudaMemAllocationTypePinned", CONV_TYPE, API_RUNTIME, SEC::MEMORY};
  m["cudaStreamCaptureModeGlobal"] = {"ascify::cudaStreamCaptureModeGlobal", CONV_TYPE, API_RUNTIME, SEC::STREAM};
  m["cudaStreamCaptureModeThreadLocal"] = {"ascify::cudaStreamCaptureModeThreadLocal", CONV_TYPE, API_RUNTIME, SEC::STREAM};
  m["cudaStreamCaptureModeRelaxed"] = {"ascify::cudaStreamCaptureModeRelaxed", CONV_TYPE, API_RUNTIME, SEC::STREAM};
  m["cudaMemAttachGlobal"] = {"ascify::cudaMemAttachGlobal", CONV_TYPE, API_RUNTIME, SEC::MEMORY};'''

ADMIT_ANCHOR = '''      "cudaProfilerStart", "cudaProfilerStop",
      "cudaLaunchHostFunc", "cudaStreamAddCallback",'''
ADMIT_NEW = '''      "cudaProfilerStart", "cudaProfilerStop",
      "cudaLaunchHostFunc", "cudaStreamAddCallback",
      "cudaGraphCreate", "cudaGraphDestroy", "cudaGraphAddKernelNode",
      "cudaGraphAddMemcpyNode", "cudaGraphAddMemcpyNode1D",
      "cudaGraphAddMemsetNode", "cudaGraphAddHostNode",
      "cudaGraphAddMemAllocNode", "cudaGraphAddMemFreeNode",
      "cudaGraphGetNodes", "cudaGraphInstantiate", "cudaGraphLaunch",
      "cudaGraphClone", "cudaGraphExecDestroy",
      "cudaStreamBeginCapture", "cudaStreamEndCapture",
      "cudaMallocManaged", "cudaDriverGetVersion",'''

REQHDR_ANCHOR = '''         name == "cudaErrorAssert" ||'''
REQHDR_NEW = '''         isCudaGraphEmulationName(name) ||
         name == "cudaErrorAssert" ||'''

HELPER_ANCHOR = '''static bool requiresCudaCompatHeader(llvm::StringRef name) {'''
HELPER_NEW = '''// Names of the CUDA Graph host-emulation surface: every one of them is
// implemented by ascify_cuda_compat.hpp, and the memcpy-node parameter
// records additionally need the pitched-copy descriptors that
// texture_types.hpp owns.
static bool isCudaGraphEmulationName(llvm::StringRef name) {
  return name == "cudaGraphCreate" || name == "cudaGraphDestroy" ||
         name == "cudaGraphAddKernelNode" ||
         name == "cudaGraphAddMemcpyNode" ||
         name == "cudaGraphAddMemcpyNode1D" ||
         name == "cudaGraphAddMemsetNode" ||
         name == "cudaGraphAddHostNode" ||
         name == "cudaGraphAddMemAllocNode" ||
         name == "cudaGraphAddMemFreeNode" || name == "cudaGraphGetNodes" ||
         name == "cudaGraphInstantiate" || name == "cudaGraphLaunch" ||
         name == "cudaGraphClone" || name == "cudaGraphExecDestroy" ||
         name == "cudaStreamBeginCapture" ||
         name == "cudaStreamEndCapture" || name == "cudaGraph_t" ||
         name == "cudaGraphExec_t" || name == "cudaGraphNode_t" ||
         name == "cudaKernelNodeParams" || name == "cudaMemsetParams" ||
         name == "cudaHostNodeParams" ||
         name == "cudaMemAllocNodeParams" || name == "cudaMemPoolProps" ||
         name == "cudaMallocManaged" || name == "cudaDriverGetVersion" ||
         name == "make_cudaPitchedPtr" || name == "make_cudaPos" ||
         name == "make_cudaExtent";
}

static bool requiresCudaCompatHeader(llvm::StringRef name) {'''

# Token-pass flag set: where requiresCudaCompatHeader is consumed in
# RewriteToken.
TOKEN_ANCHOR = '''  if (requiresCudaCompatHeader(name))
    needsCudaCompatHeader = true;'''
TOKEN_NEW = '''  if (requiresCudaCompatHeader(name))
    needsCudaCompatHeader = true;
  if (isCudaGraphEmulationName(name))
    needsGraphTextureTypesHeader = true;'''

INCLUDE_ANCHOR = '''  if (needsCudaTextureHeader)
    includes += "#include <ascify/texture_compat.hpp>\\n";'''
INCLUDE_NEW = INCLUDE_ANCHOR + '''
  // Graph memcpy/memset node parameter records reuse the pitched-copy
  // descriptor types owned by texture_types.hpp (cudaMemcpy3DParms and
  // the make_cuda* helpers); graph-using samples get exactly that leaf
  // header, not the whole texture adapter.
  if (needsGraphTextureTypesHeader)
    includes += "#include <ascify/texture_types.hpp>\\n";'''

# RecordAudit exemptions, following the CUmem* pattern.
AUDIT1_ANCHOR = '''      // VMM allocation descriptors (CUmem* types from the compat header)
      // are host-side request descriptors with no kernel-observed layout.
      // Their anonymous nested unions/structs (CUmemLocation's id union,
      // CUmemAllocationProp's allocFlags struct) are part of the same
      // compat-owned descriptor layout.
      if (const auto *vmmRecord = type->getAsTagDecl()) {
        if (llvm::StringRef(vmmRecord->getNameAsString()).starts_with("CUmem"))
          return true;
        if (vmmRecord->getNameAsString().empty())
          if (const auto *owner = vmmRecord->getDeclContext())
            if (const auto *ownerRecord =
                    llvm::dyn_cast_or_null<clang::RecordDecl>(owner))
              if (llvm::StringRef(ownerRecord->getNameAsString())
                      .starts_with("CUmem"))
                return true;
      }'''
AUDIT1_NEW = AUDIT1_ANCHOR + '''
      // CUDA Graph node-parameter records (cudaKernelNodeParams,
      // cudaMemsetParams, cudaHostNodeParams, cudaMemAllocNodeParams,
      // cudaMemPoolProps and the graph handle typedefs' pointee tags) are
      // host-side descriptors of the graph emulation in the compat header;
      // they carry no kernel-observed layout.
      if (const auto *graphRecord = type->getAsTagDecl()) {
        const llvm::StringRef graphName = graphRecord->getNameAsString();
        if (graphName.starts_with("cudaKernelNode") ||
            graphName.starts_with("cudaMemsetNode") ||
            graphName.starts_with("cudaMemsetParams") ||
            graphName.starts_with("cudaHostNode") ||
            graphName.starts_with("cudaMemAllocNode") ||
            graphName.starts_with("cudaMemPool") ||
            graphName.starts_with("cudaMemFreeNode") ||
            graphName.starts_with("cudaGraphNode") ||
            graphName.starts_with("cudaMemcpyNode"))
          return true;
      }'''

AUDIT2_ANCHOR = '''        // VMM allocation descriptors are host-side data structures used to
        // describe physical memory requests; they carry no kernel-observed
        // layout and the VMM compat owns their definition.
        if (llvm::StringRef(type->getAsTagDecl() ?
                type->getAsTagDecl()->getNameAsString() : "").contains("CUmem"))
          return true;'''
AUDIT2_NEW = AUDIT2_ANCHOR + '''
        // Graph node-parameter records are host-side graph-emulation
        // descriptors owned by the compat header (same policy as CUmem*).
        if (llvm::StringRef(type->getAsTagDecl() ?
                type->getAsTagDecl()->getNameAsString() : "")
                .starts_with("cuda"))
          if (llvm::StringRef(type->getAsTagDecl()
                  ->getNameAsString())
                  .contains("NodeParams") ||
              llvm::StringRef(type->getAsTagDecl()->getNameAsString())
                  .starts_with("cudaMemPoolProps"))
            return true;'''

def main():
    print("== CUDA2DPP_Runtime_API_functions.cpp")
    load("src/CUDA2DPP_Runtime_API_functions.cpp")
    patch("src/CUDA2DPP_Runtime_API_functions.cpp", GRAPH_FUNCS_ANCHOR, GRAPH_FUNCS_NEW)
    save("src/CUDA2DPP_Runtime_API_functions.cpp")

    print("== CUDA2DPP_Runtime_API_types.cpp")
    load("src/CUDA2DPP_Runtime_API_types.cpp")
    patch("src/CUDA2DPP_Runtime_API_types.cpp", GRAPH_TYPES_ANCHOR, GRAPH_TYPES_NEW)
    save("src/CUDA2DPP_Runtime_API_types.cpp")

    print("== AscifyAction.cpp (maps, admission, includes, audit)")
    load("src/AscifyAction.cpp")
    patch("src/AscifyAction.cpp", ADMIT_ANCHOR, ADMIT_NEW)
    patch("src/AscifyAction.cpp", HELPER_ANCHOR, HELPER_NEW)
    patch("src/AscifyAction.cpp", REQHDR_ANCHOR, REQHDR_NEW)
    patch("src/AscifyAction.cpp", TOKEN_ANCHOR, TOKEN_NEW)
    patch("src/AscifyAction.cpp", INCLUDE_ANCHOR, INCLUDE_NEW)
    patch("src/AscifyAction.cpp", AUDIT1_ANCHOR, AUDIT1_NEW)
    patch("src/AscifyAction.cpp", AUDIT2_ANCHOR, AUDIT2_NEW)
    save("src/AscifyAction.cpp")
    print("ALL PATCHES APPLIED")

if __name__ == "__main__":
    main()
