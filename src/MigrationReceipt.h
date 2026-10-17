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

#pragma once

#include <string>
#include <vector>

namespace ascify {

enum class MigrationReceiptStatus {
  InProgress,
  Succeeded,
  Failed,
};

enum class MigrationReceiptStage {
  Initialization,
  InputResolution,
  InputPreparation,
  SourceConversion,
};

struct MigrationReceiptConfig {
  std::string targetPolicy;
  std::string simtMath;
  std::string targetRecipe;
  std::string frontendCompatibility;
  std::string localHeaderClosure;
  std::string outputMode;
};

struct MigrationReceiptInput {
  std::string input;
  std::string resolvedInput;
  std::string output;
  MigrationReceiptStatus status = MigrationReceiptStatus::Failed;
  MigrationReceiptStage stage = MigrationReceiptStage::InputResolution;
  std::string diagnosticSummary;
  unsigned boundedDeviceTemplateDefinitions = 0;
};

class MigrationReceipt {
public:
  explicit MigrationReceipt(MigrationReceiptConfig config);

  void setInvocationResult(MigrationReceiptStatus status,
                           MigrationReceiptStage stage,
                           std::string diagnosticSummary);
  void addInputResult(MigrationReceiptInput input);

  std::string toJson() const;

private:
  MigrationReceiptConfig config;
  MigrationReceiptStatus status = MigrationReceiptStatus::InProgress;
  MigrationReceiptStage stage = MigrationReceiptStage::Initialization;
  std::string diagnosticSummary = "conversion has not completed";
  std::vector<MigrationReceiptInput> inputs;
};

const char *migrationReceiptStatusName(MigrationReceiptStatus status);
const char *migrationReceiptStageName(MigrationReceiptStage stage);

} // namespace ascify
