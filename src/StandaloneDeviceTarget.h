#pragma once
#include <functional>
#include <string>
#include <vector>
#include <utility>
#include "clang/Basic/SourceLocation.h"
namespace clang { class CompilerInstance; }
namespace ascify {
// Created only by the product's recursive standalone executable entry. No
// caller-authored manifest can construct this in-process build capability.
struct StandaloneDeviceIncludeEdge {
  std::string parent, child, spelling, search; unsigned offset=0; bool angled=false;
};
bool RecordStandaloneCallerEdges(const std::vector<StandaloneDeviceIncludeEdge>&, std::string& error);
bool RecordStandaloneParserArguments(const std::string& source,
    const std::vector<std::string>& arguments,std::string& error);
bool PrepareStandaloneDeviceTarget(const std::string& root, const std::string& output,
    const std::string& bundle, const char* executable, std::string& error);
bool BindStandaloneDeviceInputs(clang::CompilerInstance&, const std::string& current,
    const std::string& root, bool rootParse,
    const std::function<bool(clang::SourceLocation)>& native, std::string& error);
bool StandaloneDeviceSnapshot(const std::string& root, std::string& stamp, std::string& error);
bool BuildStandaloneDeviceTarget(const std::string& stagedRoot,
    const std::string& stagedBundle,
    const std::vector<std::pair<std::string,std::string>>& translatedHeaders, std::string& error);
bool GetStandaloneDeviceArtifacts(const std::string& bundle,
    std::vector<std::pair<std::string,std::string>>& artifacts, std::string& error);
} // namespace ascify
