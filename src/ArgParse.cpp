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

#include "ArgParse.h"

cl::OptionCategory ToolTemplateCategory("CUDA to Ascend source translator options");

cl::opt<std::string> OutputFilename("o",
  cl::desc("Output filename"),
  cl::value_desc("filename"),
  cl::cat(ToolTemplateCategory));

cl::opt<std::string> OutputDir("o-dir",
  cl::desc("Output directory"),
  cl::value_desc("directory"),
  cl::cat(ToolTemplateCategory));

cl::opt<std::string> TemporaryDir("temp-dir",
  cl::desc("Temporary directory"),
  cl::value_desc("directory"),
  cl::cat(ToolTemplateCategory));

cl::opt<std::string> CudaPath("cuda-path",
  cl::desc("CUDA installation path"),
  cl::value_desc("directory"),
  cl::cat(ToolTemplateCategory));

cl::opt<bool> SaveTemps("save-temps",
  cl::desc("Save temporary files"),
  cl::value_desc("save-temps"),
  cl::cat(ToolTemplateCategory));

cl::opt<bool> Verbose("v",
  cl::desc("Show commands to run and use verbose output"),
  cl::value_desc("v"),
  cl::cat(ToolTemplateCategory));

cl::opt<bool> Inplace("inplace",
  cl::desc("Modify input file in-place"),
  cl::value_desc("inplace"),
  cl::cat(ToolTemplateCategory));

cl::opt<bool> NoBackup("no-backup",
  cl::desc("Don't create a backup file for the translated source"),
  cl::value_desc("no-backup"),
  cl::cat(ToolTemplateCategory));

cl::opt<bool> NoOutput("no-output",
  cl::desc("Don't write any translated output to stdout"),
  cl::value_desc("no-output"),
  cl::cat(ToolTemplateCategory));

cl::opt<bool> PrintStats("print-stats",
  cl::desc("Print translation statistics"),
  cl::value_desc("print-stats"),
  cl::cat(ToolTemplateCategory));

cl::opt<bool> PrintStatsCSV("print-stats-csv",
  cl::desc("Print translation statistics in a CSV file"),
  cl::value_desc("print-stats-csv"),
  cl::cat(ToolTemplateCategory));

cl::opt<std::string> OutputStatsFilename("o-stats",
  cl::desc("Output filename for statistics"),
  cl::value_desc("filename"),
  cl::cat(ToolTemplateCategory));

cl::opt<bool> Examine("examine",
  cl::desc("Combine the '-no-output' and '-print-stats' options"),
  cl::value_desc("examine"),
  cl::cat(ToolTemplateCategory));

cl::opt<bool> DashDash("  ",
  cl::desc("Separator between ascify-clang and clang options; don't specify if there are no clang options"),
  cl::ValueDisallowed,
  cl::cat(ToolTemplateCategory));

cl::list<std::string> IncludeDirs("I",
  cl::desc("Add directory to include search path"),
  cl::value_desc("directory"),
  cl::ZeroOrMore,
  cl::Prefix,
  cl::cat(ToolTemplateCategory));

cl::list<std::string> MacroNames("D",
  cl::desc("Define <macro> to <value> or 1 if <value> omitted"),
  cl::value_desc("macro>=<value"),
  cl::ZeroOrMore,
  cl::Prefix,
  cl::cat(ToolTemplateCategory));

cl::opt<bool> SkipExcludedPPConditionalBlocks("skip-excluded-preprocessor-conditional-blocks",
  cl::desc("Enable default preprocessor behaviour by skipping undefined conditional blocks"),
  cl::value_desc("skip-excluded-preprocessor-conditional-blocks"),
  cl::cat(ToolTemplateCategory));

cl::opt<bool> DefaultPreprocessor("default-preprocessor",
  cl::desc("Enable default preprocessor behaviour (synonymous with '--skip-excluded-preprocessor-conditional-blocks')"),
  cl::value_desc("default-preprocessor"),
  cl::cat(ToolTemplateCategory));

cl::opt<bool> AscifyAMAP("amap",
  cl::desc("Try to ascify as much as possible; ignores 'default-preprocessor'"),
  cl::value_desc("amap"),
  cl::cat(ToolTemplateCategory));

cl::opt<std::string> CudaGpuArch("cuda-gpu-arch",
  cl::desc("CUDA GPU architecture (e.g. sm_35); may be specified more than once"),
  cl::value_desc("value"),
  cl::ZeroOrMore,
  cl::Prefix,
  cl::cat(ToolTemplateCategory));

cl::opt<bool> NoLowerDeviceDoubleParams("no-lower-device-double-params",
  cl::desc("Keep by-value scalar double parameters on CUDA __global__ functions "
           "(default: lower them to float for Ascend SIMT)"),
  cl::init(false),
  cl::cat(ToolTemplateCategory));

cl::opt<std::string> TargetPolicy("target-policy",
  cl::desc("SIMT target policy: portable (default) or dav-c310-vec"),
  cl::value_desc("policy"),
  cl::init("portable"),
  cl::cat(ToolTemplateCategory));

cl::opt<std::string> SimtMathMode("simt-math",
  cl::desc("SIMT floating-point transformation mode: precise (default) or fast"),
  cl::value_desc("mode"),
  cl::init("precise"),
  cl::cat(ToolTemplateCategory));

cl::opt<std::string> TargetRecipe("target-recipe",
  cl::desc("Target recipe: none (default) or dav-3510-rowwise-simd-v1"),
  cl::value_desc("recipe"),
  cl::init("none"),
  cl::cat(ToolTemplateCategory));

cl::opt<std::string> FrontendCompat("frontend-compat",
  cl::desc("Frontend compatibility profile: none (default) or "
           "ascify-admitted-v1"),
  cl::value_desc("profile"),
  cl::init("none"),
  cl::cat(ToolTemplateCategory));

cl::opt<std::string> MigrationReceiptPath("migration-receipt",
  cl::desc("Atomically write a deterministic JSON migration receipt"),
  cl::value_desc("filename"),
  cl::cat(ToolTemplateCategory));

cl::opt<bool> Versions("versions",
  cl::desc("Display LLVM build and resource versions; target validation is documented separately"),
  cl::value_desc("versions"),
  cl::cat(ToolTemplateCategory));

cl::opt<std::string> ClangResourceDir("clang-resource-directory",
  cl::desc("The clang resource path - the path to the parent folder for the 'include' folder, containing '__clang_cuda_runtime_wrapper.h' and other header files used on runtime"),
  cl::value_desc("directory"),
  cl::ZeroOrMore,
  cl::cat(ToolTemplateCategory));

cl::opt<bool> OptLocalHeaders("local-headers",
  cl::desc("Translate quoted local headers (non-recursive)"),
  cl::init(false),
  cl::cat(ToolTemplateCategory));

cl::opt<bool> OptLocalHeadersRecursive("local-headers-recursive",
  cl::desc("Translate quoted local headers recursively"),
  cl::init(false),
  cl::cat(ToolTemplateCategory));

cl::extrahelp CommonHelp(ct::CommonOptionsParser::HelpMessage);

const std::vector<std::string> ascifyOptions {
  std::string(PrintStatsCSV.ArgStr),
  std::string(PrintStats.ArgStr),
  std::string(SkipExcludedPPConditionalBlocks.ArgStr),
  std::string(DefaultPreprocessor.ArgStr),
  std::string(NoBackup.ArgStr),
  std::string(NoOutput.ArgStr),
  std::string(Inplace.ArgStr),
  std::string(Examine.ArgStr),
  std::string(SaveTemps.ArgStr),
  std::string(NoLowerDeviceDoubleParams.ArgStr),
  std::string(Versions.ArgStr),
  std::string(AscifyAMAP.ArgStr),
  std::string(OptLocalHeaders.ArgStr),
  std::string(OptLocalHeadersRecursive.ArgStr),
};

const std::vector<std::string> ascifyOptionsWithTwoArgs {
  std::string(OutputDir.ArgStr),
  std::string(OutputStatsFilename.ArgStr),
  std::string(TemporaryDir.ArgStr),
  std::string(ClangResourceDir.ArgStr),
  std::string(TargetPolicy.ArgStr),
  std::string(SimtMathMode.ArgStr),
  std::string(TargetRecipe.ArgStr),
  std::string(FrontendCompat.ArgStr),
  std::string(MigrationReceiptPath.ArgStr),
};
