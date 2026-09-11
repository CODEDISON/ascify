/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <utility>
#include "LLVMCompat.h"
#include "AscifyAction.h"
#include "ArgParse.h"
#include "llvm/Support/Debug.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Compilation.h"
#if LLVM_VERSION_MAJOR >= 22
#include "clang/Driver/CudaInstallationDetector.h"
#endif
#include "clang/Driver/Tool.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#if LLVM_VERSION_MAJOR >= 16
#include "llvm/TargetParser/Host.h"
#else
#include "llvm/Support/Host.h"
#endif

#include "LocalHeader.h"
#include "ImplicitCudaHeaders.h"
#include "FrontendCompatibility.h"
#include "MigrationReceipt.h"
#include "MigrationReceiptIO.h"

#if LLVM_VERSION_MAJOR < 8
#include "llvm/Support/Path.h"
#endif

#define STRINGIFY(x) #x
#define STRINGIFY_EXPANDED(x) STRINGIFY(x)

constexpr auto DEBUG_TYPE = "cuda2hip";

namespace ct = clang::tooling;

void cleanupAscifyOptions(std::vector<const char*> &args) {
  for (const auto &a : ascifyOptions) {
    args.erase(std::remove(args.begin(), args.end(), "--" + a), args.end());
    args.erase(std::remove(args.begin(), args.end(), "-" + a), args.end());
  }
  for (const auto &a : ascifyOptionsWithTwoArgs) {
    const std::string longOption = "--" + a;
    const std::string shortOption = "-" + a;
    for (auto it = args.begin(); it != args.end();) {
      const std::string current(*it);
      if (current == longOption || current == shortOption) {
        it = args.erase(it);
        if (it != args.end())
          it = args.erase(it);
      } else if (current.find(longOption + "=") == 0 ||
                 current.find(shortOption + "=") == 0) {
        it = args.erase(it);
      } else {
        ++it;
      }
    }
  }
}

void DetectCUDA(const std::unique_ptr<clang::driver::Compilation> &C) {
#if LLVM_VERSION_MAJOR >= 22
  const clang::driver::Driver &driver = C->getDriver();
  clang::driver::CudaInstallationDetector CudaInstallation(driver, llvm::Triple(driver.getTargetTriple()), C->getArgs());
  auto& FS = driver.getVFS();
  if (auto cuda_h_file = FS.getBufferForFile(CudaInstallation.getInstallPath() + "/include/cuda.h"))
    Statistics::setCudaVersion((*cuda_h_file)->getBuffer());
  llvm::errs() << "\n" << sAscify << "CUDA Installation Path: " << CudaInstallation.getInstallPath();
  llvm::errs() << "\n" << sAscify << "CUDA_VERSION: " << Statistics::getCudaVersion() << "\n";
#endif
}

void Init(int argc, const char **argv, std::vector<std::string> &files) {
#if LLVM_VERSION_MAJOR >= 21
  clang::DiagnosticOptions diagOpts;
  clang::TextDiagnosticPrinter diagClient(llvm::errs(), diagOpts);
  clang::DiagnosticsEngine Diagnostics(IntrusiveRefCntPtr<clang::DiagnosticIDs>(new clang::DiagnosticIDs()), diagOpts, &diagClient, false);
#else
  IntrusiveRefCntPtr<clang::DiagnosticOptions> diagOpts(new clang::DiagnosticOptions());
  clang::TextDiagnosticPrinter diagClient(llvm::errs(), &*diagOpts);
  clang::DiagnosticsEngine Diagnostics(IntrusiveRefCntPtr<clang::DiagnosticIDs>(new clang::DiagnosticIDs()), &*diagOpts, &diagClient, false);
#endif
  std::unique_ptr<clang::driver::Driver> driver(new clang::driver::Driver("", llvm::sys::getDefaultTargetTriple(), Diagnostics));
  std::vector<const char*> Args(argv, argv + argc);
  cleanupAscifyOptions(Args);
  std::unique_ptr<clang::driver::Compilation> C(driver->BuildCompilation(Args));
  DetectCUDA(C);
  if (files.size() < 2) return;
  std::vector<std::string> sortedFiles;
  for (const auto &J : C->getJobs()) {
    if (std::string(J.getCreator().getName()) != "clang") continue;
    const auto &JA = J.getArguments();
    for (size_t i = 0; i < JA.size(); ++i) {
      const auto &A = std::string(JA[i]);
      if (std::find(files.begin(), files.end(), A) != files.end() &&
        i > 0 && std::string(JA[i - 1]) == "-main-file-name") {
        sortedFiles.push_back(A);
      }
    }
  }
  if (sortedFiles.empty()) return;
  std::reverse(sortedFiles.begin(), sortedFiles.end());
  files.assign(sortedFiles.begin(), sortedFiles.end());
}

bool checkLLVM(std::string &path_to_check) {
  const std::string file_name_to_check = "__clang_cuda_runtime_wrapper.h";
  const std::string file_name_to_check_2 = "algorithm";
  const std::string cuda_wrappers_dir = "cuda_wrappers";
  std::string fileToCheck = path_to_check + "/" + file_name_to_check;
  bool bExist = llvm::sys::fs::exists(llvm::Twine(fileToCheck.c_str()));
  if (bExist) {
    fileToCheck = path_to_check + "/" + cuda_wrappers_dir + "/" + file_name_to_check_2;
    bExist = llvm::sys::fs::exists(llvm::Twine(fileToCheck.c_str()));
  }
  return bExist;
}

bool setLLVM(ct::RefactoringTool &Tool, const char *ascify_exe) {
  static int Dummy;
  std::string ascify = llvm::sys::fs::getMainExecutable(ascify_exe, (void*)&Dummy);
  std::string ascify_parent_path = std::string(llvm::sys::path::parent_path(ascify));
  std::string clang_ver = STRINGIFY_EXPANDED(LIB_CLANG_RES);
  std::string clang_res_path, clang_inc_path, fileToCheck;
  const std::string include_dir = "include";
  bool bExist = false;
  // 1. --clang-resource-dir is specified
  if (!ClangResourceDir.empty()) {
    clang_res_path = ClangResourceDir;
    clang_inc_path = clang_res_path + "/" + include_dir;
    bExist = checkLLVM(clang_inc_path);
  }
  // 2. Check for ROCm LLVM
  if (!bExist) {
#if defined(_WIN32)
    // HIP SDK for Windows
    clang_res_path = ascify_parent_path + "/../lib/clang/" + clang_ver;
#else
    // ROCm Linux
    clang_res_path = ascify_parent_path + "/../lib/llvm/lib/clang/" + clang_ver;
#endif
    clang_inc_path = clang_res_path + "/" + include_dir;
    bExist = checkLLVM(clang_inc_path);
  }
#ifndef _WIN32
  if (!bExist) {
    // 2.1. ROCm Linux: ascify-clang standalone package
    clang_res_path = ascify_parent_path + "/../" + include_dir + "/ascify";
    clang_inc_path = clang_res_path + "/" + include_dir;
    bExist = checkLLVM(clang_inc_path);
  }
#endif
  // 3. Check for clang include copied by cmake install
  if (!bExist) {
    clang_res_path = ascify_parent_path;
    clang_inc_path = clang_res_path + "/" + include_dir;
    bExist = checkLLVM(clang_inc_path);
  }
  if (bExist) {
    std::string sRes = "-resource-dir=" + clang_res_path;
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster(sRes.c_str(), ct::ArgumentInsertPosition::BEGIN));
  }
  return bExist;
}

bool appendArgumentsAdjusters(
    ct::RefactoringTool &Tool,
    const std::string &sSourceAbsPath,
    const char *ascify_exe,
    ascify::FrontendCompatibilityConfig &frontendCompatibility) {
  if (!setLLVM(Tool, ascify_exe)) {
    llvm::errs() << "\n" << sAscify << sError << "LLVM to work with not found. Ascification is impossible. Exiting. To provide ascify-clang with LLVM to work with, please specify the `--clang-resource-directory` option." << "\n";
    return false;
  }
  if (!IncludeDirs.empty()) {
    for (std::string s : IncludeDirs) {
      Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster(s.c_str(), ct::ArgumentInsertPosition::BEGIN));
      Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-I", ct::ArgumentInsertPosition::BEGIN));
    }
  }
  if (!MacroNames.empty()) {
    for (std::string s : MacroNames) {
      Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster(s.c_str(), ct::ArgumentInsertPosition::BEGIN));
      Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-D", ct::ArgumentInsertPosition::BEGIN));
    }
  }
  // Standard c++ to use in hipification by default
  llcompat::setStdCPP(Tool);
  std::string sInclude = "-I" + sys::path::parent_path(sSourceAbsPath).str();
  Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster(sInclude.c_str(), ct::ArgumentInsertPosition::BEGIN));
  Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-fno-delayed-template-parsing", ct::ArgumentInsertPosition::BEGIN));
  if (llcompat::pragma_once_outside_header()) {
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-Wno-pragma-once-outside-header", ct::ArgumentInsertPosition::BEGIN));
  }
  Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("--cuda-host-only", ct::ArgumentInsertPosition::BEGIN));
  if (!CudaGpuArch.empty()) {
    std::string sCudaGpuArch = "--cuda-gpu-arch=" + CudaGpuArch;
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster(sCudaGpuArch.c_str(), ct::ArgumentInsertPosition::BEGIN));
  }
  if (!CudaPath.empty()) {
    std::string sCudaPath = "--cuda-path=" + CudaPath;
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster(sCudaPath.c_str(), ct::ArgumentInsertPosition::BEGIN));
  }

  // Implicit CUDA headers to mimic nvcc behavior
  ascify::addImplicitCudaHeaders(Tool);
  llcompat::addTargetIfNeeded(Tool);
  Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("cuda", ct::ArgumentInsertPosition::BEGIN));
  Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-x", ct::ArgumentInsertPosition::BEGIN));
  std::string frontendCompatibilityError;
  if (!ascify::ConfigureFrontendCompatibility(
          Tool, FrontendCompat, ascify_exe, frontendCompatibility,
          frontendCompatibilityError)) {
    llvm::errs() << "\n" << sAscify << sError
                 << frontendCompatibilityError << "\n";
    return false;
  }
  if (Verbose) {
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-v", ct::ArgumentInsertPosition::END));
  }
  Tool.appendArgumentsAdjuster(ct::getClangSyntaxOnlyAdjuster());
  return true;
}

bool ascifySingleSource(const std::string &srcPath,
                               const std::string &dstPath,
                               const ct::CompilationDatabase *compDB,
                               ct::CommonOptionsParser *OptionsParserPtr,
                               const char *ascify_exe_path,
                               const std::string &mainContextPath,
                               bool preserveTemp,
                               LocalHeaderRewriteContext *localHeaderContext) {
  if (localHeaderContext != nullptr &&
      localHeaderContext->virtualInput != nullptr) {
    // Reparse a verified contextual input at the original root identity.
    // No physical source is overwritten: only apply edits to the memory
    // buffer, then write the existing unpublished staging destination.
    ct::RefactoringTool Tool(
        compDB ? *compDB : OptionsParserPtr->getCompilations(), srcPath);
    Tool.mapVirtualFile(srcPath, *localHeaderContext->virtualInput);
    ct::Replacements &edits = llcompat::getReplacements(Tool, srcPath);
    ascify::FrontendCompatibilityConfig frontendCompatibility;
    if (!appendArgumentsAdjusters(Tool, mainContextPath, ascify_exe_path,
                                  frontendCompatibility))
      return false;
    ReplacementsFrontendActionFactory<AscifyAction> factory(
        &edits, localHeaderContext, frontendCompatibility);
    if (Tool.run(&factory))
      return false;
    auto rewritten = ct::applyAllReplacements(
        *localHeaderContext->virtualInput, edits);
    if (!rewritten) {
      llvm::errs() << sAscify << sError << llvm::toString(rewritten.takeError())
                   << "\n";
      return false;
    }
    std::error_code writeError;
    llvm::raw_fd_ostream out(dstPath, writeError);
    if (writeError)
      return false;
    out << *rewritten;
    out.close();
    return !out.has_error();
  }
  std::error_code EC;
  SmallString<128> tmpFile;
  StringRef srcFileName = sys::path::filename(srcPath);

  EC = sys::fs::createTemporaryFile(srcFileName, "hip", tmpFile);
  if (EC) {
    llvm::errs() << "\n" << sAscify << sError << "Failed to create temporary file: " << EC.message() << "\n";
    return false;
  }

  // Copy source to temp
  EC = sys::fs::copy_file(srcPath, tmpFile);
  if (EC) {
    llvm::errs() << "\n" << sAscify << sError << EC.message() 
                 << ": while copying " << srcPath << " to " << tmpFile << "\n";
    if (!SaveTemps && !preserveTemp) sys::fs::remove(tmpFile);
    return false;
  }

  // RefactoringTool operates on the file in-place. Giving it the output path is no good,
  // because that'll break relative includes, and we don't want to overwrite the input file.
  // So what we do is operate on a copy, which we then move to the output.
  ct::RefactoringTool Tool((compDB ? *compDB : OptionsParserPtr->getCompilations()),
                           std::string(tmpFile.c_str()));
  ct::Replacements &replacementsToUse = llcompat::getReplacements(Tool, tmpFile.c_str());
  ascify::FrontendCompatibilityConfig frontendCompatibility;

  if (!appendArgumentsAdjusters(
          Tool, mainContextPath, ascify_exe_path, frontendCompatibility)) {
    llvm::errs() << "\n" << sAscify << sError 
                 << "LLVM/resource config failed for: " << srcPath << "\n";
    if (!SaveTemps && !preserveTemp) sys::fs::remove(tmpFile);
    return false;
  }
  ReplacementsFrontendActionFactory<AscifyAction> actionFactory(
      &replacementsToUse, localHeaderContext, frontendCompatibility);

  // Ascify _all_ the things!
  if (Tool.runAndSave(&actionFactory)) {
    llvm::errs() << "\n" << sAscify << sError 
                 << "Ascifying failed: " << srcPath << "\n";
    if (!SaveTemps && !preserveTemp) sys::fs::remove(tmpFile);
    return false;
  }

  // Copy the tmpfile to the output
  if (!dstPath.empty()) {
    EC = sys::fs::copy_file(tmpFile, dstPath);
    if (EC) {
      llvm::errs() << "\n" << sAscify << sError << EC.message() 
                   << ": while copying " << tmpFile << " to " << dstPath << "\n";
      if (!SaveTemps && !preserveTemp) sys::fs::remove(tmpFile);
      return false;
    }
  }

  // Remove the tmp file without error chec
  if (!SaveTemps && !preserveTemp) {
    sys::fs::remove(tmpFile);
  }

  return true;
}

bool generatePython() {
    return true;
}

void printVersions() {
  llvm::errs() << "\n" << sAscify << "Supports DPP compatibility versions from " << Statistics::getDppVersion(dppVersions::DPP_1050) << " up to " << Statistics::getDppVersion(dppVersions::DPP_LATEST);
  llvm::errs() << "\n" << sAscify << "Supports CUDA Toolkit from " << Statistics::getCudaVersion(cudaVersions::CUDA_70) << " up to " << Statistics::getCudaVersion(cudaVersions::CUDA_LATEST);
  llvm::errs() << "\n" << sAscify << "Supports cuDNN from " << Statistics::getCudaVersion(cudaVersions::CUDNN_705) << " up to " << Statistics::getCudaVersion(cudaVersions::CUDNN_LATEST) << " \n";
}

std::string migrationReceiptLocalHeaderMode() {
  if (OptLocalHeadersRecursive)
    return "recursive";
  if (OptLocalHeaders)
    return "direct";
  return "disabled";
}

std::string migrationReceiptOutputMode() {
  if (NoOutput)
    return "no_output";
  if (Inplace)
    return "inplace";
  if (!OutputDir.empty())
    return "output_directory";
  if (!OutputFilename.empty())
    return "output_file";
  return "side_by_side";
}

std::string normalizedPathForReceiptComparison(const std::string &path) {
  if (path.empty())
    return "";
  std::error_code canonicalError;
  const std::filesystem::path canonical =
      std::filesystem::weakly_canonical(std::filesystem::path(path),
                                        canonicalError);
  if (!canonicalError)
    return canonical.string();
  SmallString<256> normalized(path);
  std::error_code ec = sys::fs::make_absolute(normalized);
  if (ec)
    return path;
  sys::path::remove_dots(normalized, true);
  return normalized.str().str();
}

bool migrationReceiptPathConflicts(const std::string &receiptPath,
                                   const std::string &artifactPath) {
  return !receiptPath.empty() && !artifactPath.empty() &&
         normalizedPathForReceiptComparison(receiptPath) ==
             normalizedPathForReceiptComparison(artifactPath);
}

int main(int argc, const char **argv) {
  std::vector<const char*> new_argv(argv, argv + argc);
  std::string sCompilationDatabaseDir;
  auto it = std::find(new_argv.begin(), new_argv.end(), std::string("-p"));
  bool bCompilationDatabase = it == new_argv.end() ? false : true;
  bool bNoCompilationDatabaseDir = false;
  if (bCompilationDatabase) {
    if (it+1 != new_argv.end()) sCompilationDatabaseDir = *(it+1);
    else bNoCompilationDatabaseDir = true;
  } else {
    for (auto &s : new_argv) {
      std::string str = std::string(s);
      if (str.find("-p=") != std::string::npos) {
        bCompilationDatabase = true;
        sCompilationDatabaseDir = str.substr(3, str.size()-3);
        if (sCompilationDatabaseDir.empty()) {
          bNoCompilationDatabaseDir = true;
        }
        break;
      }
    }
  }
  if (bCompilationDatabase && bNoCompilationDatabaseDir) {
    llvm::errs() << "\n" << sAscify << sError << "Must specify compilation database directory" << "\n";
    return 1;
  }
  if (!bCompilationDatabase && std::find(new_argv.begin(), new_argv.end(), std::string("--")) == new_argv.end()) {
    new_argv.push_back("--");
    new_argv.push_back(nullptr);
    argv = new_argv.data();
    argc++;
  }
  llcompat::PrintStackTraceOnErrorSignal();
#if LLVM_VERSION_MAJOR > 12
  auto cop = ct::CommonOptionsParser::create(argc, argv, ToolTemplateCategory, llvm::cl::ZeroOrMore);
  if (!cop) {
    llvm::errs() << "\n" << sAscify << sError << cop.takeError() << "\n";
    return 1;
  }
  ct::CommonOptionsParser &OptionsParser = cop.get();
#else
  ct::CommonOptionsParser OptionsParser(argc, argv, ToolTemplateCategory, llvm::cl::ZeroOrMore);
#endif
  if (!llcompat::CheckCompatibility()) {
    return 1;
  }
  std::unique_ptr<ct::CompilationDatabase> compilationDatabase;
  std::vector<std::string> fileSources;
  if (bCompilationDatabase) {
    std::string serr;
    compilationDatabase = ct::CompilationDatabase::loadFromDirectory(sCompilationDatabaseDir, serr);
    if (nullptr == compilationDatabase.get()) {
      llvm::errs() << "\n" << sAscify << sError << "loading Compilation Database from \"" << sCompilationDatabaseDir << "compile_commands.json\" failed\n";
      return 1;
    }
    fileSources = compilationDatabase->getAllFiles();
  } else {
    fileSources = OptionsParser.getSourcePathList();
  }
  if (fileSources.empty() && !GeneratePerl && !GeneratePython && !GenerateMarkdown && !GenerateCSV && !Versions) {
    llvm::errs() << "\n" << sAscify << sError << "Must specify at least 1 positional argument for source file" << "\n";
    return 1;
  }
  if (Versions) printVersions();
  if (fileSources.empty()) {
    return 0;
  }
  std::string dst = OutputFilename, dstDir = OutputDir;
  std::error_code EC;
  std::string sOutputDirAbsPath = getAbsoluteDirectoryPath(OutputDir, EC, "output");
  if (EC) {
    return 1;
  }
  if (!dst.empty()) {
    if (fileSources.size() > 1) {
      llvm::errs() << sAscify << sConflict << "-o and multiple source files are specified\n";
      return 1;
    }
    if (Inplace) {
      llvm::errs() << sAscify << sConflict << "both -o and -inplace options are specified\n";
      return 1;
    }
    if (NoOutput) {
      llvm::errs() << sAscify << sConflict << "both -no-output and -o options are specified\n";
      return 1;
    }
    if (!dstDir.empty()) {
      dst = sOutputDirAbsPath + "/" + dst;
    }
  }
  if (NoOutput && Inplace) {
    llvm::errs() << sAscify << sConflict << "both -no-output and -inplace options are specified\n";
    return 1;
  }
  if (!dstDir.empty() && Inplace) {
    llvm::errs() << sAscify << sConflict << "both -o-dir and -inplace options are specified\n";
    return 1;
  }
  if (!dstDir.empty() && fileSources.size() > 1) {
    std::set<std::string> outputNames;
    for (const std::string &source : fileSources) {
      const std::string outputName =
          sys::path::filename(source).str() + ".dpp";
      if (!outputNames.insert(outputName).second) {
        llvm::errs() << sAscify << sConflict
                     << "multiple inputs map to the same -o-dir artifact: "
                     << outputName << "\n";
        return 1;
      }
    }
  }
  if (Examine) {
    NoOutput = PrintStats = true;
  }
  int Result = 0;
  SmallString<128> tmpFile;
  StringRef sourceFileName, ext = "dpp", csv_ext = "csv";
  std::string sTmpFileName, sSourceAbsPath;
  std::string sTmpDirAbsParh = getAbsoluteDirectoryPath(TemporaryDir, EC);
  if (EC) {
    return 1;
  }
  // Arguments for the Statistics print routines.
  std::unique_ptr<std::ostream> csv = nullptr;
  llvm::raw_ostream *statPrint = nullptr;
  bool create_csv = false;
  if (!OutputStatsFilename.empty()) {
    PrintStatsCSV = true;
    create_csv = true;
  } else {
    if (PrintStatsCSV && fileSources.size() > 1) {
      OutputStatsFilename = "sum_stat.csv";
      create_csv = true;
    }
  }
  if (create_csv) {
    if (!OutputDir.empty()) {
      OutputStatsFilename = sOutputDirAbsPath + "/" + OutputStatsFilename;
    }
  }

  if (!MigrationReceiptPath.empty()) {
    for (const std::string &source : fileSources) {
      if (migrationReceiptPathConflicts(MigrationReceiptPath, source)) {
        llvm::errs() << sAscify << sConflict
                     << "migration receipt path and source path are the same: "
                     << source << "\n";
        return 1;
      }

      std::string prospectiveOutput;
      if (!NoOutput) {
        if (Inplace) {
          prospectiveOutput = source;
        } else if (!OutputFilename.empty()) {
          prospectiveOutput = dst;
        } else if (!OutputDir.empty()) {
          prospectiveOutput =
              sOutputDirAbsPath + "/" +
              sys::path::filename(source).str() + ".dpp";
        } else {
          prospectiveOutput = source + ".dpp";
        }
      }
      if (migrationReceiptPathConflicts(MigrationReceiptPath,
                                        prospectiveOutput)) {
        llvm::errs() << sAscify << sConflict
                     << "migration receipt path and translated output path "
                        "are the same: "
                     << prospectiveOutput << "\n";
        return 1;
      }
      if (!TemporaryDir.empty()) {
        const std::string prospectiveTemporary =
            sTmpDirAbsParh + "/" +
            sys::path::filename(source).str() + ".dpp";
        if (migrationReceiptPathConflicts(MigrationReceiptPath,
                                          prospectiveTemporary)) {
          llvm::errs() << sAscify << sConflict
                       << "migration receipt path and conversion temporary "
                          "path are the same: "
                       << prospectiveTemporary << "\n";
          return 1;
        }
      }
      if (PrintStatsCSV && !create_csv) {
        std::string prospectiveStatistics =
            sys::path::filename(source).str() + ".csv";
        if (!OutputDir.empty()) {
          prospectiveStatistics =
              sOutputDirAbsPath + "/" + prospectiveStatistics;
        }
        if (migrationReceiptPathConflicts(MigrationReceiptPath,
                                          prospectiveStatistics)) {
          llvm::errs() << sAscify << sConflict
                       << "migration receipt path and statistics output path "
                          "are the same: "
                       << prospectiveStatistics << "\n";
          return 1;
        }
      }
    }
    if (create_csv &&
        migrationReceiptPathConflicts(MigrationReceiptPath,
                                      OutputStatsFilename)) {
      llvm::errs() << sAscify << sConflict
                   << "migration receipt path and statistics output path are "
                      "the same: "
                   << OutputStatsFilename << "\n";
      return 1;
    }
  }

  std::unique_ptr<ascify::MigrationReceipt> migrationReceipt;
  if (!MigrationReceiptPath.empty()) {
    ascify::MigrationReceiptConfig receiptConfig{
        TargetPolicy,
        SimtMathMode,
        TargetRecipe,
        FrontendCompat,
        migrationReceiptLocalHeaderMode(),
        migrationReceiptOutputMode(),
    };
    migrationReceipt = std::make_unique<ascify::MigrationReceipt>(
        std::move(receiptConfig));
    std::string receiptError;
    if (!ascify::publishMigrationReceiptAtomically(
            MigrationReceiptPath, *migrationReceipt, receiptError)) {
      llvm::errs() << sAscify << sError << receiptError << "\n";
      return 1;
    }
  }

  auto finishWithMigrationReceipt =
      [&](int result, ascify::MigrationReceiptStatus status,
          ascify::MigrationReceiptStage stage,
          const std::string &diagnosticSummary) {
        if (!migrationReceipt)
          return result;
        migrationReceipt->setInvocationResult(status, stage,
                                              diagnosticSummary);
        std::string receiptError;
        if (!ascify::publishMigrationReceiptAtomically(
                MigrationReceiptPath, *migrationReceipt, receiptError)) {
          llvm::errs() << sAscify << sError << receiptError << "\n";
          return 1;
        }
        return result;
      };

  if (create_csv) {
    csv = std::unique_ptr<std::ostream>(new std::ofstream(OutputStatsFilename, std::ios_base::trunc));
  }
  if (PrintStats) {
    statPrint = &llvm::errs();
  }
  Init(argc, argv, fileSources);
  ascify::MigrationReceiptStage lastReceiptStage =
      ascify::MigrationReceiptStage::Initialization;
  for (const auto &src : fileSources) {
    ascify::MigrationReceiptInput receiptInput;
    receiptInput.input = src;
    lastReceiptStage = ascify::MigrationReceiptStage::InputResolution;
    // Create a copy of the file to work on. When we're done, we'll move this onto the
    // output (which may mean overwriting the input, if we're in-place).
    // Should we fail for some reason, we'll just leak this file and not corrupt the input.
    sSourceAbsPath = getAbsoluteFilePath(src, EC);
    if (EC) {
      receiptInput.status = ascify::MigrationReceiptStatus::Failed;
      receiptInput.stage = ascify::MigrationReceiptStage::InputResolution;
      receiptInput.diagnosticSummary = "input path could not be resolved";
      if (migrationReceipt)
        migrationReceipt->addInputResult(std::move(receiptInput));
      Result = 1;
      continue;
    }
    receiptInput.resolvedInput = sSourceAbsPath;
    sourceFileName = sys::path::filename(sSourceAbsPath);
    if (dst.empty()) {
      if (Inplace) {
        dst = src;
      } else {
        dst = src + "." + ext.str();
        if (!dstDir.empty()) {
          dst = sOutputDirAbsPath + "/" + sourceFileName.str() + "." + ext.str();
        }
      }
    }
    receiptInput.output = NoOutput ? "" : dst;
    lastReceiptStage = ascify::MigrationReceiptStage::InputPreparation;
    if (TemporaryDir.empty()) {
      EC = sys::fs::createTemporaryFile(sourceFileName, ext, tmpFile);
      if (EC) {
        llvm::errs() << "\n" << sAscify << sError << EC.message() << ": " << tmpFile << "\n";
        receiptInput.status = ascify::MigrationReceiptStatus::Failed;
        receiptInput.stage = ascify::MigrationReceiptStage::InputPreparation;
        receiptInput.diagnosticSummary =
            "conversion temporary file could not be created";
        if (migrationReceipt)
          migrationReceipt->addInputResult(std::move(receiptInput));
        Result = 1;
        continue;
      }
    } else {
      sTmpFileName = sTmpDirAbsParh + "/" + sourceFileName.str() + "." + ext.str();
      tmpFile = sTmpFileName;
    }
    EC = sys::fs::copy_file(src, tmpFile);
    if (EC) {
      llvm::errs() << "\n" << sAscify << sError << EC.message() << ": while copying " << src << " to " << tmpFile << "\n";
      receiptInput.status = ascify::MigrationReceiptStatus::Failed;
      receiptInput.stage = ascify::MigrationReceiptStage::InputPreparation;
      receiptInput.diagnosticSummary =
          "input could not be copied to the conversion temporary file";
      if (migrationReceipt)
        migrationReceipt->addInputResult(std::move(receiptInput));
      Result = 1;
      continue;
    }
    if (PrintStatsCSV) {
      if (OutputStatsFilename.empty()) {
        OutputStatsFilename = sourceFileName.str() + "." + csv_ext.str();
        if (!OutputDir.empty()) {
          OutputStatsFilename = sOutputDirAbsPath + "/" + OutputStatsFilename;
        }
      }
      if (!csv) {
        csv = std::unique_ptr<std::ostream>(new std::ofstream(OutputStatsFilename, std::ios_base::trunc));
      }
    }
    // Initialise the statistics counters for this file.
    Statistics::setActive(src);
    std::string outputPath = NoOutput ? "" : dst;
    lastReceiptStage = ascify::MigrationReceiptStage::SourceConversion;
    const bool conversionOk =
        (OptLocalHeaders || OptLocalHeadersRecursive)
            ? ascifySourceWithLocalHeaderClosure(
                  sSourceAbsPath, outputPath, compilationDatabase.get(),
                  &OptionsParser, argv[0], OptLocalHeadersRecursive, Inplace,
                  NoOutput)
            : ascifySingleSource(
                  src, outputPath, compilationDatabase.get(), &OptionsParser,
                  argv[0], sSourceAbsPath, false, nullptr);
    if (!conversionOk) {
      Statistics::current().hasErrors = true;
      Result = 1;
      LLVM_DEBUG(llvm::dbgs() << "Hipification failed for: " << src << "\n");
    }

    receiptInput.status = conversionOk
                              ? ascify::MigrationReceiptStatus::Succeeded
                              : ascify::MigrationReceiptStatus::Failed;
    receiptInput.stage = ascify::MigrationReceiptStage::SourceConversion;
    receiptInput.diagnosticSummary =
        conversionOk
            ? "source conversion completed; later pipeline stages are outside this receipt"
            : "source conversion failed; detailed diagnostics were emitted on stderr";
    if (migrationReceipt)
      migrationReceipt->addInputResult(std::move(receiptInput));

    Statistics::current().markCompletion();
    Statistics::current().print(csv.get(), statPrint);
    dst.clear();
  }
  if (fileSources.size() > 1) {
    Statistics::printAggregate(csv.get(), statPrint);
  }
  if (Result == 0) {
    return finishWithMigrationReceipt(
        Result, ascify::MigrationReceiptStatus::Succeeded,
        lastReceiptStage,
        "all requested source conversions completed; target compilation, linking, and device validation are outside this receipt");
  }
  return finishWithMigrationReceipt(
      Result, ascify::MigrationReceiptStatus::Failed, lastReceiptStage,
      "one or more inputs failed; see per-input results and stderr diagnostics");
}
