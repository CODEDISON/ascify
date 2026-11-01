#pragma once

#include <string>
#include <vector>
#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/Core/Replacement.h"

namespace clang { class CompilerInstance; }
namespace ascify {
class HostCxxRootInputWitness;
// This proof preserves literal source; it does not synthesize packed layouts.
// All events and final replacements must be checked before publication.
class PackedRecordBoundary {
 public:
  bool pragma(clang::CompilerInstance &, clang::SourceLocation,
              bool open, bool reset, bool selected, bool governedVirtualMain,
              const HostCxxRootInputWitness *unmodifiedHostInput = nullptr);
  void includeOrFileTransition();
  bool prove(clang::CompilerInstance &, bool unrelatedPragmaFailure,
             std::vector<clang::tooling::Replacement> &ownedEdits);
  bool auditFinalEdits(const clang::tooling::Replacements &,
                       const std::vector<clang::tooling::Replacement> &ownedEdits);
  bool used() const { return !regions.empty(); }
  const std::string &error() const { return failure; }
  // All raw pack directives that this complete-region proof actually owns.
  std::vector<std::pair<clang::FileID, unsigned>> ownedPragmaLines() const;
 private:
  struct Region {
    clang::FileID file;
    std::string path, bytes;
    unsigned begin = 0, open = 0, bodyBegin = 0, bodyEnd = 0, reset = 0, end = 0;
    std::vector<unsigned> pragmaLines;
    std::vector<std::string> undefinedNames;
    bool closed = false;
  };
  std::vector<Region> regions;
  bool active = false;
  std::string failure;
  bool fail(const std::string &);
};
}
