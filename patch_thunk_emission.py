#!/usr/bin/env python3
"""Rework the graph thunk emission to the includes-block path (CDP pattern)."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent

p = ROOT / "src/AscifyAction.cpp"
t = p.read_text()

# 1) Remove debug traces.
t = t.replace("""  llvm::errs() << "ASCIFY DBG graphhandle cast kind="
               << cast->getStmtClassName() << "\\n";
""", "")
t = t.replace("""  if (kernelRef == nullptr || kernelRef->hasQualifier()) {
    llvm::errs() << "ASCIFY DBG graphhandle: no kernel ref\\n";
    return false;
  }""", """  if (kernelRef == nullptr || kernelRef->hasQualifier()) return false;""")
t = t.replace("""  llvm::errs() << "ASCIFY DBG graphhandle ref="
               << kernelRef->getNameInfo().getAsString();
  if (kernel == nullptr) { llvm::errs() << " NOTFN\\n"; return false; }
  llvm::errs() << " global=" << (kernel->hasAttr<clang::CUDAGlobalAttr>() ? 1 : 0)
               << " body=" << (kernel->getBody() != nullptr ? 1 : 0) << "\\n";
""", "")

# 2) Replace the after-body glue insertion with includes-block accumulation.
old = """  const std::string kernelName = kernel->getNameAsString();
  const std::string thunkName = "ascify_graphglue_" + kernelName;
  const auto &options = getCompilerInstance().getLangOpts();
  if (graphGlueKernels.insert(kernelName).second) {
    const clang::SourceLocation afterBody =
        clang::Lexer::getLocForEndOfToken(bodyEnd, 0, source, options);
    if (afterBody.isInvalid()) return false;
    clang::PrintingPolicy policy(context.getLangOpts());
    std::string glue;
    llvm::raw_string_ostream stream(glue);
    stream << "\\n\\n// Ascify graph kernel-node bridge: typed <<<>>> thunk for "
           << kernelName << " (CUDA graph emulation).\\n"
           << "namespace {\\n"
           << "static void " << thunkName
           << "(unsigned gx, unsigned gy, unsigned gz, unsigned bx, "
              "unsigned by, unsigned bz, unsigned shm, ::aclrtStream s, "
              "void **kp) {\\n"
           << "  " << kernelName
           << "<<<::dim3(gx, gy, gz), ::dim3(bx, by, bz), shm, s>>>(";
    bool first = true;
    for (const auto *parameter : kernel->parameters()) {
      if (!first) stream << ", ";
      first = false;
      const std::string spelling = parameter->getType().getAsString(policy);
      stream << "*static_cast<" << spelling << " *>(kp["
             << parameter->getFunctionScopeIndex() << "])";
    }
    stream << ");\\n}\\n}  // namespace\\n";
    stream.flush();
    const ct::Replacement glueEdit(source, afterBody, 0, glue);
    const bool glueInserted = insertSemanticReplacement(
        glueEdit, clang::FullSourceLoc(afterBody, source), afterBody,
        afterBody);
    llvm::errs() << "ASCIFY DBG graphhandle glue " << kernelName
                 << " inserted=" << (glueInserted ? 1 : 0) << "\\n";
    if (!glueInserted) {
      graphGlueKernels.erase(kernelName);
      return false;
    }
  }"""
new = """  const std::string kernelName = kernel->getNameAsString();
  const std::string thunkName = "ascify_graphglue_" + kernelName;
  if (graphGlueKernels.insert(kernelName).second) {
    // The thunk forward-declares the kernel and is emitted with the
    // injected includes block (the same materialization path the CDP
    // replay glue uses); the use-site rewrite below points the graph
    // node's launcher at the thunk.
    clang::PrintingPolicy policy(context.getLangOpts());
    std::string glue;
    llvm::raw_string_ostream stream(glue);
    stream << "// Ascify graph kernel-node bridge: typed <<<>>> thunk for "
           << kernelName << " (CUDA graph emulation).\\n";
    stream << "ASCIFY_GLOBAL void " << kernelName << "(";
    bool first = true;
    for (const auto *parameter : kernel->parameters()) {
      if (!first) stream << ", ";
      first = false;
      stream << parameter->getType().getAsString(policy);
    }
    stream << ");\\n";
    stream << "namespace {\\n"
           << "static void " << thunkName
           << "(unsigned gx, unsigned gy, unsigned gz, unsigned bx, "
              "unsigned by, unsigned bz, unsigned shm, ::aclrtStream s, "
              "void **kp) {\\n"
           << "  " << kernelName
           << "<<<dim3(gx, gy, gz), dim3(bx, by, bz), shm, s>>>(";
    first = true;
    for (const auto *parameter : kernel->parameters()) {
      if (!first) stream << ", ";
      first = false;
      const std::string spelling = parameter->getType().getAsString(policy);
      stream << "*static_cast<" << spelling << " *>(kp["
             << parameter->getFunctionScopeIndex() << "])";
    }
    stream << ");\\n}\\n}  // namespace\\n";
    stream.flush();
    graphGlueText += glue;
    needsGraphTextureTypesHeader = true;
  }"""
assert t.count(old) == 1, f"glue block anchor {t.count(old)}"
t = t.replace(old, new)

# 3) Drop the site-insert debug wrapper.
old2 = """  const bool siteInserted = insertSemanticReplacement(
      ct::Replacement(source, begin, length, replacement),
      clang::FullSourceLoc(begin, source), begin, end);
  llvm::errs() << "ASCIFY DBG graphhandle site " << kernelName
               << " off=" << offset << " inserted=" << (siteInserted ? 1 : 0)
               << "\\n";
  if (!siteInserted) {
    graphKernelHandleOffsets.erase(offset);
    return false;
  }"""
new2 = """  if (!insertSemanticReplacement(
          ct::Replacement(source, begin, length, replacement),
          clang::FullSourceLoc(begin, source), begin, end)) {
    graphKernelHandleOffsets.erase(offset);
    return false;
  }"""
assert t.count(old2) == 1, f"site anchor {t.count(old2)}"
t = t.replace(old2, new2)

# 4) Materialize the accumulated glue with the includes block.
old3 = """  if (needsGraphTextureTypesHeader)
    includes += "#include <ascify/texture_types.hpp>\\n";"""
new3 = """  if (needsGraphTextureTypesHeader)
    includes += "#include <ascify/texture_types.hpp>\\n";
  if (!graphGlueText.empty())
    includes += graphGlueText;"""
assert t.count(old3) == 1
t = t.replace(old3, new3)
p.write_text(t)
print("handler emission reworked")

h = ROOT / "src/AscifyAction.h"
th = h.read_text()
oldh = """  std::set<std::string> graphGlueKernels;
  std::set<unsigned> graphKernelHandleOffsets;"""
newh = """  std::set<std::string> graphGlueKernels;
  std::set<unsigned> graphKernelHandleOffsets;
  std::string graphGlueText;"""
assert th.count(oldh) == 1
th = th.replace(oldh, newh)
h.write_text(th)
print("member added")
