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

#include "MigrationReceipt.h"

#include <algorithm>
#include <sstream>
#include <tuple>
#include <utility>

namespace ascify {
namespace {

std::string escapeJson(const std::string &value) {
  static const char hex[] = "0123456789abcdef";
  std::string escaped;
  escaped.reserve(value.size());
  for (const unsigned char ch : value) {
    switch (ch) {
    case '"':
      escaped += "\\\"";
      break;
    case '\\':
      escaped += "\\\\";
      break;
    case '\b':
      escaped += "\\b";
      break;
    case '\f':
      escaped += "\\f";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\r':
      escaped += "\\r";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      if (ch < 0x20) {
        escaped += "\\u00";
        escaped += hex[(ch >> 4) & 0x0f];
        escaped += hex[ch & 0x0f];
      } else {
        escaped += static_cast<char>(ch);
      }
      break;
    }
  }
  return escaped;
}

void writeJsonString(std::ostringstream &out, const std::string &value) {
  out << '"' << escapeJson(value) << '"';
}

} // namespace

const char *migrationReceiptStatusName(MigrationReceiptStatus status) {
  switch (status) {
  case MigrationReceiptStatus::InProgress:
    return "in_progress";
  case MigrationReceiptStatus::Succeeded:
    return "succeeded";
  case MigrationReceiptStatus::Failed:
    return "failed";
  }
  return "failed";
}

const char *migrationReceiptStageName(MigrationReceiptStage stage) {
  switch (stage) {
  case MigrationReceiptStage::Initialization:
    return "initialization";
  case MigrationReceiptStage::InputResolution:
    return "input_resolution";
  case MigrationReceiptStage::InputPreparation:
    return "input_preparation";
  case MigrationReceiptStage::SourceConversion:
    return "source_conversion";
  }
  return "initialization";
}

MigrationReceipt::MigrationReceipt(MigrationReceiptConfig config)
    : config(std::move(config)) {}

void MigrationReceipt::setInvocationResult(
    MigrationReceiptStatus newStatus, MigrationReceiptStage newStage,
    std::string newDiagnosticSummary) {
  status = newStatus;
  stage = newStage;
  diagnosticSummary = std::move(newDiagnosticSummary);
}

void MigrationReceipt::addInputResult(MigrationReceiptInput input) {
  inputs.push_back(std::move(input));
}

std::string MigrationReceipt::toJson() const {
  std::vector<MigrationReceiptInput> sortedInputs = inputs;
  std::stable_sort(
      sortedInputs.begin(), sortedInputs.end(),
      [](const MigrationReceiptInput &lhs, const MigrationReceiptInput &rhs) {
        return std::tie(lhs.input, lhs.resolvedInput, lhs.output, lhs.status,
                        lhs.stage, lhs.diagnosticSummary, lhs.boundedDeviceTemplateDefinitions,
                        lhs.cudaVec3LayoutEvidence) <
               std::tie(rhs.input, rhs.resolvedInput, rhs.output, rhs.status,
                        rhs.stage, rhs.diagnosticSummary, rhs.boundedDeviceTemplateDefinitions,
                        rhs.cudaVec3LayoutEvidence);
      });

  std::ostringstream out;
  out << "{\n";
  out << "  \"schema\": \"ascify.migration-receipt\",\n";
  out << "  \"schema_version\": 1,\n";
  out << "  \"tool\": \"ascify-clang\",\n";
  out << "  \"status\": \"" << migrationReceiptStatusName(status)
      << "\",\n";
  out << "  \"stage\": \"" << migrationReceiptStageName(stage) << "\",\n";
  out << "  \"diagnostic_summary\": ";
  writeJsonString(out, diagnosticSummary);
  out << ",\n";
  out << "  \"target_profile\": {\n";
  out << "    \"policy\": ";
  writeJsonString(out, config.targetPolicy);
  out << ",\n    \"simt_math\": ";
  writeJsonString(out, config.simtMath);
  out << ",\n    \"recipe\": ";
  writeJsonString(out, config.targetRecipe);
  out << "\n  },\n";
  out << "  \"contract_decisions\": {\n";
  out << "    \"frontend_compatibility\": ";
  writeJsonString(out, config.frontendCompatibility);
  out << ",\n    \"local_header_closure\": ";
  writeJsonString(out, config.localHeaderClosure);
  out << ",\n    \"output_mode\": ";
  writeJsonString(out, config.outputMode);
  out << "\n  },\n";
  out << "  \"inputs\": [";
  if (!sortedInputs.empty())
    out << '\n';
  for (std::size_t index = 0; index < sortedInputs.size(); ++index) {
    const MigrationReceiptInput &input = sortedInputs[index];
    out << "    {\n";
    out << "      \"input\": ";
    writeJsonString(out, input.input);
    out << ",\n      \"resolved_input\": ";
    writeJsonString(out, input.resolvedInput);
    out << ",\n      \"output\": ";
    writeJsonString(out, input.output);
    out << ",\n      \"status\": \""
        << migrationReceiptStatusName(input.status) << "\",\n";
    out << "      \"stage\": \"" << migrationReceiptStageName(input.stage)
        << "\",\n";
    out << "      \"diagnostic_summary\": ";
    writeJsonString(out, input.diagnosticSummary);
    out << ",\n      \"bounded_device_template_inlining\": {\"rule\": "
           "\"int-nttp-scalar-dag-v1\", \"definitions\": "
        << input.boundedDeviceTemplateDefinitions << '}';
    out << ",\n      \"cuda_vec3_local_layout\": {\"rule\": \"closed-scalar-local-v1\", \"evidence\": [";
    for (std::size_t item = 0; item < input.cudaVec3LayoutEvidence.size(); ++item) {
      if (item != 0) out << ',';
      writeJsonString(out, input.cudaVec3LayoutEvidence[item]);
    }
    out << "]}";
    out << "\n    }";
    if (index + 1 != sortedInputs.size())
      out << ',';
    out << '\n';
  }
  if (!sortedInputs.empty())
    out << "  ";
  out << "]\n";
  out << "}\n";
  return out.str();
}

} // namespace ascify
