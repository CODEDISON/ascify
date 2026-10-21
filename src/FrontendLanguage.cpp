#include "FrontendLanguage.h"
#include "ArgParse.h"
#if __has_include("clang/Options/Options.h")
#include "clang/Options/Options.h"
namespace options = clang::options;
using clang::getDriverOptTable;
#else
#include "clang/Driver/Options.h"
namespace options = clang::driver::options;
using clang::driver::getDriverOptTable;
#endif
#include "llvm/Option/ArgList.h"
#include <filesystem>
#include <set>

namespace ascify {
namespace {
std::string normalized(const std::string &file, const std::string &directory) {
  std::filesystem::path path(file);
  if (path.is_relative())
    path = std::filesystem::path(directory) / path;
  std::error_code ec;
  const auto absolute = std::filesystem::weakly_canonical(path, ec);
  return ec ? path.lexically_normal().string() : absolute.string();
}

bool cudaPredefine(llvm::StringRef definition) {
  const auto name = definition.split('=').first;
  return name == "__CUDACC__" || name == "__CUDA__" ||
         name == "__CUDA_ARCH__" || name == "__CUDACC_RDC__";
}

bool deviceArgument(llvm::StringRef spelling) {
  // Do not silently discard device flags, nor let forwarded cc1 arguments
  // override the selected language after driver argument normalization.
  const std::string value = spelling.str();
  for (const char *prefix : {"--cuda-", "-fcuda-", "-fno-cuda-", "-fgpu-",
                             "-fno-gpu-", "--offload-", "-Xcuda-",
                             "-Xarch_", "-Xclang", "-Xopenmp-target",
                             "-Xpreprocessor", "-Wp,"}) {
    if (value.find(prefix) == 0)
      return true;
  }
  return value == "-nocudainc" || value == "-nocudalib" ||
         value == "-nogpuinc" || value == "-nogpulib" ||
         value == "-fopenmp-is-target-device";
}
} // namespace

bool FrontendLanguageDatabase::initialize(
    const clang::tooling::CompilationDatabase &source,
    const std::string &root, std::string &error) {
  if (InputLanguage != "auto" && InputLanguage != "cuda" &&
      InputLanguage != "c++") {
    error = "invalid --input-language; expected auto, cuda or c++";
    return false;
  }
  const auto commands = source.getCompileCommands(root);
  if (commands.size() != 1 || commands.front().CommandLine.empty()) {
    error = "input language requires exactly one original TU compile command: " + root;
    return false;
  }
  command_ = commands.front();
  const auto &argv = command_.CommandLine;
  std::vector<const char *> raw;
  for (std::size_t index = 1; index < argv.size(); ++index) {
    if (!argv[index].empty() && argv[index][0] == '@') {
      error = "unexpanded response file cannot select an input language";
      return false;
    }
    raw.push_back(argv[index].c_str());
  }
  unsigned missingIndex = 0, missingCount = 0;
  const auto parsed = getDriverOptTable().ParseArgs(raw, missingIndex, missingCount);
  if (missingCount != 0) {
    error = "missing compiler argument while selecting input language";
    return false;
  }
  std::string language;
  bool sawInput = false;
  std::set<std::size_t> removed;
  for (const auto *arg : parsed) {
    if (arg->getOption().matches(options::OPT_x)) {
      if (sawInput) {
        error = "place compiler -x before the original TU input";
        return false;
      }
      language = arg->getValue();
      const std::size_t index = arg->getIndex() + 1;
      removed.insert(index);
      if (argv[index] == "-x" || argv[index] == "--language")
        removed.insert(index + 1);
    } else if (arg->getOption().matches(options::OPT__DASH_DASH)) {
      if (sawInput || arg->getNumValues() != 1 ||
          normalized(arg->getValue(), command_.Directory) !=
              normalized(root, command_.Directory)) {
        error = "input language requires the single original TU after --: " + root;
        return false;
      }
      sawInput = true;
      removed.insert(arg->getIndex() + 1);
      removed.insert(arg->getIndex() + 2);
    } else if (arg->getOption().matches(options::OPT_INPUT)) {
      if (sawInput || normalized(arg->getValue(), command_.Directory) !=
                          normalized(root, command_.Directory)) {
        error = "input language requires the single original TU input: " + root;
        return false;
      }
      sawInput = true;
      removed.insert(arg->getIndex() + 1);
    }
  }
  if (!sawInput) {
    error = "original TU input missing from compile command: " + root;
    return false;
  }
  if (!language.empty() && language != "cuda" && language != "c++") {
    error = "unsupported compiler -x language; expected cuda or c++";
    return false;
  }
  if (InputLanguage != "auto" && !language.empty() &&
      language != std::string(InputLanguage)) {
    error = "--input-language conflicts with compiler -x";
    return false;
  }
  if (language.empty())
    language = InputLanguage == "auto" ? "cuda" : std::string(InputLanguage);
  hostCxx_ = language == "c++";
  if (hostCxx_) {
    if (std::filesystem::path(root).extension() == ".cu") {
      error = "host C++ input language refuses a .cu root; use CUDA mode";
      return false;
    }
    if (!CudaGpuArch.empty() || AscifyAMAP) {
      error = "host C++ input language conflicts with --cuda-gpu-arch or --amap";
      return false;
    }
    for (const auto *arg : parsed) {
      if (deviceArgument(arg->getSpelling()) ||
          (arg->getOption().matches(options::OPT_D) &&
           cudaPredefine(arg->getValue()))) {
        error = "host C++ input language conflicts with compiler argument: " +
                arg->getSpelling().str();
        return false;
      }
    }
    for (const auto &macro : MacroNames) {
      if (cudaPredefine(macro)) {
        error = "host C++ input language conflicts with CUDA predefine: " + macro;
        return false;
      }
    }
  }
  std::vector<std::string> normalizedArgs;
  for (std::size_t index = 0; index < argv.size(); ++index) {
    if (removed.count(index) == 0)
      normalizedArgs.push_back(argv[index]);
  }
  normalizedArgs.push_back("-x");
  normalizedArgs.push_back(language);
  command_.CommandLine = std::move(normalizedArgs);
  return true;
}

std::vector<clang::tooling::CompileCommand>
FrontendLanguageDatabase::getCompileCommands(llvm::StringRef file) const {
  auto command = command_;
  command.Filename = file.str();
  command.CommandLine.push_back("--");
  command.CommandLine.push_back(file.str());
  return {std::move(command)};
}
} // namespace ascify
