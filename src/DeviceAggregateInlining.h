#pragma once
#include <functional>
#include <string>
#include <vector>
#include <utility>
#include "clang/Basic/SourceLocation.h"
namespace clang { class CompilerInstance; class Decl; class Token; class MacroInfo; class MacroDefinition; class MacroArgs; }
namespace ascify {
struct DeviceNativeStringArgument {
  clang::SourceLocation hash,parameter,argument;
  std::string spelling;
};
struct DeviceNativeMacroUse {
  const clang::MacroInfo* definition=nullptr;
  std::string name;
  clang::SourceLocation nameLocation;
  clang::SourceRange invocation;
  bool unambiguous=false;
  std::vector<DeviceNativeStringArgument> stringArguments;
};
struct DeviceNativeMacroEvidence {
  std::vector<DeviceNativeMacroUse> uses;
  std::vector<std::pair<std::string,clang::SourceLocation>> controls;
  std::vector<clang::SourceLocation> pragmas;
};
// Records actual PP callbacks. No caller manifest creates these witnesses.
void ObserveDeviceNativeMacro(clang::CompilerInstance&,const clang::Token&,
    const clang::MacroDefinition&,clang::SourceRange,const clang::MacroArgs*,
    DeviceNativeMacroEvidence&);
struct DeviceAggregateInlineFact {
  std::string projectStamp, sourcePath, sourceSha256, recordName, layout, bodyGraph, signature;
  std::vector<std::string> referenceSites;
  unsigned functionOffset=0, insertOffset=0, guardOffset=0;
  unsigned recordBytes=0, recordAlign=0, sharedBytes=0, callerOffset=0, calls=0;
  bool operator==(const DeviceAggregateInlineFact& o) const;
};
struct DeviceRegisterTile32Fact {
  std::string projectStamp,sourcePath,sourceSha256;
  unsigned beginOffset=0,endOffset=0;
  bool operator==(const DeviceRegisterTile32Fact& other) const;
};
struct DeviceAggregateInlineAnalysis {
  std::string projectStamp;
  std::vector<DeviceAggregateInlineFact> facts;
  std::vector<DeviceRegisterTile32Fact> registerFactories;
};
bool ValidateDeviceInlineProjectSnapshot(const std::string& root,
    const std::string& expectedStamp, std::string& error);

// Explicit owned standalone target capability plus its complete real AST.
// No supplied manifest or local/parent CMake assertion can grant authority.
bool AnalyzeDeviceAggregateInlining(clang::CompilerInstance&, const std::string& originalRoot, const std::string& currentSource,
    const std::function<bool(clang::SourceLocation)>& owned,
    const std::function<bool(const clang::Decl*)>& sdk,
    const std::function<bool(clang::SourceLocation)>& native,
    const DeviceNativeMacroEvidence&,
    const std::function<bool(const DeviceNativeMacroUse&)>& helperDiagnostic,
    const std::function<bool(clang::FileID)>& completeGuard,
    const std::vector<DeviceAggregateInlineFact>* rootWitnesses,
    const std::vector<DeviceRegisterTile32Fact>* rootRegisterWitnesses,
    DeviceAggregateInlineAnalysis&, std::string& error);
} // namespace ascify
