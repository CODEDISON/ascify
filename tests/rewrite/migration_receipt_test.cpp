#include "MigrationReceipt.h"

#include <cassert>
#include <iostream>
#include <string>

namespace {

ascify::MigrationReceipt makeSuccessReceipt(bool reverseInputs) {
  ascify::MigrationReceipt receipt({
      "dav-c310-vec",
      "precise",
      "none",
      "ascify-admitted-v1",
      "recursive",
      "output_directory",
  });
  receipt.setInvocationResult(
      ascify::MigrationReceiptStatus::Succeeded,
      ascify::MigrationReceiptStage::SourceConversion,
      "all inputs completed");

  ascify::MigrationReceiptInput first{
      "b\\input.cu",
      "/work/b\\input.cu",
      "/out/b\ninput.cu.dpp",
      ascify::MigrationReceiptStatus::Succeeded,
      ascify::MigrationReceiptStage::SourceConversion,
      "source conversion completed",
  };
  ascify::MigrationReceiptInput second{
      "a-input.cu",
      "/work/a-input.cu",
      "/out/a-input.cu.dpp",
      ascify::MigrationReceiptStatus::Succeeded,
      ascify::MigrationReceiptStage::SourceConversion,
      "source conversion completed",
  };
  first.boundedDeviceTemplateDefinitions = 2;
  first.cudaVec3LayoutEvidence = {"site=/work/quoted\"name.cu:7;type=float3;layout=12/4"};
  first.cudaProfilerAvailabilityEvidence = {
      "source=/work/quoted\"name.cu;if_offset=7;sdk=/cuda/include/cuda_profiler_api.h"};
  if (reverseInputs) {
    receipt.addInputResult(first);
    receipt.addInputResult(second);
  } else {
    receipt.addInputResult(second);
    receipt.addInputResult(first);
  }
  return receipt;
}

} // namespace

int main() {
  const std::string forward = makeSuccessReceipt(false).toJson();
  const std::string reverse = makeSuccessReceipt(true).toJson();
  assert(forward == reverse);
  assert(forward.find("\"schema_version\": 1") != std::string::npos);
  assert(forward.find("\"status\": \"succeeded\"") != std::string::npos);
  assert(forward.find("target compilation") == std::string::npos);
  assert(forward.find("a-input.cu") < forward.find("b\\\\input.cu"));
  assert(forward.find("b\\\\input.cu") != std::string::npos);
  assert(forward.find("b\\ninput.cu.dpp") != std::string::npos);
  assert(forward.find("\"rule\": \"int-nttp-scalar-dag-v1\", \"definitions\": 2") != std::string::npos);
  assert(forward.find("\"rule\": \"closed-scalar-local-v1\"") != std::string::npos);
  assert(forward.find("quoted\\\"name.cu:7;type=float3;layout=12/4") != std::string::npos);
  assert(forward.find("\"backend_status\": \"unverified\"") != std::string::npos);
  assert(forward.find("\"target_header\": \"acl/acl_prof.h\"") != std::string::npos);
  assert(forward.find("quoted\\\"name.cu;if_offset=7;sdk=") != std::string::npos);
  assert(forward.find("\"backend_status\": \"verified\"") == std::string::npos);

  ascify::MigrationReceipt failed({
      "portable", "precise", "none", "none", "disabled", "side_by_side",
  });
  failed.setInvocationResult(
      ascify::MigrationReceiptStatus::Failed,
      ascify::MigrationReceiptStage::InputPreparation,
      "temporary file failed");
  const std::string failedJson = failed.toJson();
  assert(failedJson.find("\"status\": \"failed\"") != std::string::npos);
  assert(failedJson.find("\"stage\": \"input_preparation\"") !=
         std::string::npos);
  assert(failedJson.find("\"status\": \"succeeded\"") ==
         std::string::npos);
  assert(failedJson.find("cuda_profiler_availability") == std::string::npos);

  std::cout << forward;
  return 0;
}
