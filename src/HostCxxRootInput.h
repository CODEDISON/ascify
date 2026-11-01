#pragma once

#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <memory>
#include <string>

namespace ascify {

// Identity evidence for main.cpp's own unmodified Host C++ main-file mapping,
// including a selected physical header parsed as a standalone host main.
// Identity alone grants no SDK/profiler permission and does not authenticate
// caller remaps, contextual inputs or any SDK header. Those users retain their
// own narrower root-only admission contract.
class HostCxxRootInputWitness final {
 public:
  static std::unique_ptr<const HostCxxRootInputWitness>
  capture(const std::string &absolutePhysicalPath);

  bool matchesPhysical() const;
  bool matchesParsed(llvm::StringRef filename, llvm::StringRef buffer,
                     std::uint64_t entrySize) const;
  const std::string &path() const { return path_; }
  const std::string &bytes() const { return bytes_; }
  const std::string &sha256() const { return sha256_; }

 private:
  HostCxxRootInputWitness(std::string path, std::string bytes,
                          std::string digest, std::uint64_t device,
                          std::uint64_t file, std::uint64_t size);
  const std::string path_, bytes_, sha256_;
  const std::uint64_t device_, file_, size_;
};

} // namespace ascify
