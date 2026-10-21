#pragma once

#include "clang/Tooling/CompilationDatabase.h"
#include <string>

namespace ascify {

// One original TU command, never inferred from the temporary parse filename.
class FrontendLanguageDatabase final
    : public clang::tooling::CompilationDatabase {
 public:
  bool initialize(const clang::tooling::CompilationDatabase &source,
                  const std::string &root, std::string &error);
  bool hostCxx() const { return hostCxx_; }
  std::vector<clang::tooling::CompileCommand>
  getCompileCommands(llvm::StringRef file) const override;

 private:
  clang::tooling::CompileCommand command_;
  bool hostCxx_ = false;
};

} // namespace ascify
