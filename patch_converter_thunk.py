#!/usr/bin/env python3
"""Converter-side changes for the graph kernel-node thunk bridge."""
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
    assert found == count, f"{rel}: anchor found {found}x (want {count}):\n{old[:300]}"
    FILES[rel] = text.replace(old, new)
    print(f"  patched {rel}: {old.splitlines()[0][:70]}...")

def save(rel):
    (ROOT / rel).write_text(FILES[rel])
    print(f"saved {rel}")

H_DECL_ANCHOR = '''  bool rewriteCudaLaunchCooperativeKernel(const clang::ast_matchers::MatchFinder::MatchResult &Result);
  bool rewriteDeviceWideningFloatCast(const clang::ast_matchers::MatchFinder::MatchResult &Result);'''
H_DECL_NEW = '''  bool rewriteCudaLaunchCooperativeKernel(const clang::ast_matchers::MatchFinder::MatchResult &Result);
  bool rewriteCudaGraphKernelHandle(const clang::ast_matchers::MatchFinder::MatchResult &Result);
  bool rewriteDeviceWideningFloatCast(const clang::ast_matchers::MatchFinder::MatchResult &Result);'''

H_MEMBER_ANCHOR = '''  std::set<unsigned> rewrittenGlobalAtomicOffsets;'''
H_MEMBER_NEW = '''  std::set<unsigned> rewrittenGlobalAtomicOffsets;
  // CUDA Graph kernel-node bridge: kernels whose address was taken (one
  // emitted typed <<<>>> thunk each) and the already-rewritten cast sites.
  std::set<std::string> graphGlueKernels;
  std::set<unsigned> graphKernelHandleOffsets;'''

DISPATCH_ANCHOR = '''  if (Result.Nodes.getNodeAs<clang::CastExpr>(sWideningFloatCast) != nullptr) {
    (void)rewriteDeviceWideningFloatCast(Result);
    return;
  }'''
DISPATCH_NEW = '''  if (Result.Nodes.getNodeAs<clang::CastExpr>(sWideningFloatCast) != nullptr) {
    if (!rewriteCudaGraphKernelHandle(Result))
      (void)rewriteDeviceWideningFloatCast(Result);
    return;
  }'''

IMPL_ANCHOR = '''bool AscifyAction::rewriteCudaLaunchCooperativeKernel(
    const mat::MatchFinder::MatchResult &Result) {'''
IMPL_NEW = '''// CUDA Graph manual kernel-node bridge. A host-side `(void *)kernel`
// (or `&kernel`) expression is rewritten to the address of a typed <<<>>>
// thunk emitted directly after the kernel definition. The compat layer's
// cudaGraphAddKernelNode stores that address as the node's launcher and
// replays the node by handing the caller's CUDA kernelParams array to the
// thunk, which performs a real launch; the ccec argument packing is never
// reconstructed. Same proof shape as the cooperative-launch projection:
// unqualified main-file non-template __global__ kernels only.
bool AscifyAction::rewriteCudaGraphKernelHandle(
    const mat::MatchFinder::MatchResult &Result) {
  const auto *cast = Result.Nodes.getNodeAs<clang::CastExpr>(sWideningFloatCast);
  if (cast == nullptr || Result.Context == nullptr ||
      Result.SourceManager == nullptr || cast->isTypeDependent() ||
      cast->isValueDependent())
    return false;
  auto &context = *Result.Context;
  auto &source = *Result.SourceManager;
  const bool explicitCast =
      llvm::isa_and_nonnull<clang::CStyleCastExpr>(cast) ||
      llvm::isa_and_nonnull<clang::CXXStaticCastExpr>(cast) ||
      llvm::isa_and_nonnull<clang::CXXReinterpretCastExpr>(cast) ||
      llvm::isa_and_nonnull<clang::CXXConstCastExpr>(cast);
  if (!explicitCast) return false;
  const clang::Expr *sub = cast->getSubExpr()->IgnoreParenImpCasts();
  const clang::DeclRefExpr *kernelRef = nullptr;
  if (const auto *address = llvm::dyn_cast<clang::UnaryOperator>(sub)) {
    if (address->getOpcode() != clang::UO_AddrOf) return false;
    kernelRef = llvm::dyn_cast<clang::DeclRefExpr>(
        address->getSubExpr()->IgnoreParenImpCasts());
  } else {
    kernelRef = llvm::dyn_cast<clang::DeclRefExpr>(sub);
  }
  if (kernelRef == nullptr || kernelRef->hasQualifier()) return false;
  const auto *kernel =
      llvm::dyn_cast<clang::FunctionDecl>(kernelRef->getDecl());
  if (kernel == nullptr || !kernel->hasAttr<clang::CUDAGlobalAttr>() ||
      kernel->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate ||
      kernel->getBody() == nullptr)
    return false;
  // Host context only: a device or kernel function cannot host the thunk
  // address (device-side kernel references belong to the CDP path).
  {
    // Bounded walk up the semantic parents to the nearest function.
    const clang::FunctionDecl *owner = nullptr;
    std::vector<clang::ast_type_traits::DynTypedNode> frontier =
        context.getParents(*cast);
    int hops = 0;
    while (!frontier.empty() && hops < 96) {
      std::vector<clang::ast_type_traits::DynTypedNode> next;
      for (const auto &node : frontier) {
        if (const auto *fn = node.get<clang::FunctionDecl>()) {
          owner = fn;
          break;
        }
        for (const auto &parent : context.getParents(node))
          next.push_back(parent);
      }
      if (owner != nullptr) break;
      frontier = std::move(next);
      ++hops;
    }
    if (owner != nullptr && (owner->hasAttr<clang::CUDAGlobalAttr>() ||
                             owner->hasAttr<clang::CUDADeviceAttr>()))
      return false;
  }
  const clang::SourceLocation definition =
      source.getExpansionLoc(kernel->getLocation());
  const clang::SourceLocation bodyEnd =
      source.getExpansionLoc(kernel->getBody()->getEndLoc());
  if (definition.isInvalid() || bodyEnd.isInvalid() ||
      !source.isWrittenInMainFile(definition) ||
      !source.isWrittenInMainFile(bodyEnd))
    return false;
  const std::string kernelName = kernel->getNameAsString();
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
    if (!insertSemanticReplacement(
            glueEdit, clang::FullSourceLoc(afterBody, source), afterBody,
            afterBody)) {
      graphGlueKernels.erase(kernelName);
      return false;
    }
  }
  const auto begin = source.getFileLoc(cast->getBeginLoc());
  const auto last = source.getFileLoc(cast->getEndLoc());
  if (begin.isInvalid() || last.isInvalid() ||
      !source.isWrittenInMainFile(begin) ||
      !source.isWrittenInMainFile(last))
    return true;  // thunk stays; a macro-spelled site keeps its own text
  const auto end = clang::Lexer::getLocForEndOfToken(last, 0, source, options);
  if (end.isInvalid()) return false;
  const auto offset = source.getFileOffset(begin);
  if (!graphKernelHandleOffsets.insert(offset).second) return true;
  const auto length = source.getFileOffset(end) - offset;
  const std::string replacement = "(void *)&" + thunkName;
  if (!insertSemanticReplacement(
          ct::Replacement(source, begin, length, replacement),
          clang::FullSourceLoc(begin, source), begin, end)) {
    graphKernelHandleOffsets.erase(offset);
    return false;
  }
  static const dppCounter counter = {
      "CUDA graph kernel bridge", CONV_EXECUTION, API_RUNTIME, 0, FULL};
  Statistics::current().incrementCounter(counter,
                                         kernelName + " graph kernel thunk");
  return true;
}

bool AscifyAction::rewriteCudaLaunchCooperativeKernel(
    const mat::MatchFinder::MatchResult &Result) {'''

def main():
    print("== AscifyAction.h")
    load("src/AscifyAction.h")
    patch("src/AscifyAction.h", H_DECL_ANCHOR, H_DECL_NEW)
    patch("src/AscifyAction.h", H_MEMBER_ANCHOR, H_MEMBER_NEW)
    save("src/AscifyAction.h")

    print("== AscifyAction.cpp (thunk bridge)")
    load("src/AscifyAction.cpp")
    patch("src/AscifyAction.cpp", DISPATCH_ANCHOR, DISPATCH_NEW)
    patch("src/AscifyAction.cpp", IMPL_ANCHOR, IMPL_NEW)
    save("src/AscifyAction.cpp")
    print("THUNK PATCHES APPLIED")

if __name__ == "__main__":
    main()
