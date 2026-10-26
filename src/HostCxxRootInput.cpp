#include "HostCxxRootInput.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#if LLVM_VERSION_MAJOR >= 13
#include "llvm/Support/SHA256.h"
#endif
#include <utility>

namespace ascify {
namespace {
std::string digest(llvm::StringRef buffer) {
#if LLVM_VERSION_MAJOR >= 13
  llvm::SHA256 hash;
  hash.update(buffer);
  const auto bytes = hash.final();
  const char hex[] = "0123456789abcdef";
  std::string result;
  for (const auto byte : bytes) {
    const unsigned value = static_cast<unsigned char>(byte);
    result += hex[value >> 4];
    result += hex[value & 15];
  }
  return result;
#else
  return {};
#endif
}

bool status(const std::string &path, llvm::sys::fs::file_status &result) {
  llvm::SmallString<256> physical;
  return llvm::sys::path::is_absolute(path) &&
      !llvm::sys::fs::real_path(path, physical) && physical == path &&
      !llvm::sys::fs::status(path, result) &&
      llvm::sys::fs::is_regular_file(result);
}
} // namespace

HostCxxRootInputWitness::HostCxxRootInputWitness(
    std::string path, std::string bytes, std::string hash,
    std::uint64_t device, std::uint64_t file, std::uint64_t size)
    : path_(std::move(path)), bytes_(std::move(bytes)),
      sha256_(std::move(hash)), device_(device), file_(file), size_(size) {}

std::unique_ptr<const HostCxxRootInputWitness>
HostCxxRootInputWitness::capture(const std::string &path) {
  llvm::sys::fs::file_status before, after;
  if (!status(path, before)) return nullptr;
  const auto physical = llvm::MemoryBuffer::getFile(path);
  if (!physical || !status(path, after) ||
      before.getUniqueID() != after.getUniqueID() ||
      before.getSize() != after.getSize() ||
      after.getSize() != (*physical)->getBufferSize()) return nullptr;
  const auto hash = digest((*physical)->getBuffer());
  if (hash.empty()) return nullptr;
  std::unique_ptr<const HostCxxRootInputWitness> witness(
      new HostCxxRootInputWitness(path, (*physical)->getBuffer().str(), hash,
          after.getUniqueID().getDevice(), after.getUniqueID().getFile(),
          after.getSize()));
  if (!witness->matchesPhysical()) return nullptr;
  return witness;
}

bool HostCxxRootInputWitness::matchesPhysical() const {
  llvm::sys::fs::file_status before, after;
  if (!status(path_, before) || before.getUniqueID().getDevice() != device_ ||
      before.getUniqueID().getFile() != file_ || before.getSize() != size_)
    return false;
  const auto physical = llvm::MemoryBuffer::getFile(path_);
  return physical && status(path_, after) &&
      before.getUniqueID() == after.getUniqueID() &&
      before.getSize() == after.getSize() &&
      (*physical)->getBuffer() == bytes_ &&
      digest((*physical)->getBuffer()) == sha256_;
}

bool HostCxxRootInputWitness::matchesParsed(
    llvm::StringRef filename, llvm::StringRef buffer,
    std::uint64_t entrySize) const {
  return filename == path_ && entrySize == size_ && buffer == bytes_ &&
      digest(buffer) == sha256_ && matchesPhysical();
}
} // namespace ascify
