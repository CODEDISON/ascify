#include "HostCxxRootInput.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
static void require(bool value, const char *message) {
  if (!value) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}
static void write(const fs::path &path, const std::string &bytes) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file << bytes;
  file.close();
  require(!file.fail(), "could not write private test input");
}
int main() {
  llvm::SmallString<256> created, canonical;
  require(!llvm::sys::fs::createUniqueDirectory(
      "ascify-host-cxx-root-witness", created), "could not create test directory");
  require(!llvm::sys::fs::real_path(created, canonical), "could not resolve test directory");
  const fs::path directory(canonical.str().str());
  struct Cleanup {
    fs::path path;
    ~Cleanup() { std::error_code error; fs::remove_all(path, error); }
  } cleanup{directory};
  const fs::path source = directory / "original.cpp";
  const std::string bytes = "int original=1;\n";
  write(source, bytes);
  auto witness = ascify::HostCxxRootInputWitness::capture(source.string());
  require(bool(witness), "real root was not captured");
  require(witness->path() == source.string() && witness->bytes() == bytes &&
          witness->sha256().size() == 64 && witness->matchesPhysical(),
          "original source was not bound");
  require(witness->matchesParsed(source.string(), bytes, bytes.size()),
          "exact mapped input refused");
  require(!witness->matchesParsed((directory / "other.cpp").string(), bytes, bytes.size()),
          "other main-file path acquired witness");
  require(!witness->matchesParsed(source.string(), bytes, bytes.size() + 1),
          "wrong virtual entry size acquired witness");
  require(!witness->matchesParsed(source.string(), "int original=2;\n", bytes.size()),
          "different parsed bytes acquired witness");
  require(!ascify::HostCxxRootInputWitness::capture("original.cpp"),
          "relative path acquired witness");
  require(!ascify::HostCxxRootInputWitness::capture(directory.string()),
          "directory acquired witness");
  write(source, "int original=2;\n");
  require(!witness->matchesPhysical(), "same-size physical content change accepted");
  auto replacementWitness = ascify::HostCxxRootInputWitness::capture(source.string());
  require(bool(replacementWitness), "new physical source was not captured");
  const fs::path replacement = directory / "replacement.cpp";
  write(replacement, replacementWitness->bytes());
  fs::rename(replacement, source);
  require(!replacementWitness->matchesPhysical(),
          "different inode with identical bytes acquired witness");
  const fs::path alias = directory / "alias.cpp";
  fs::create_symlink(source, alias);
  require(!ascify::HostCxxRootInputWitness::capture(alias.string()),
          "uncanonicalized alias acquired witness");
  auto sizeWitness = ascify::HostCxxRootInputWitness::capture(source.string());
  require(bool(sizeWitness), "replacement source was not captured");
  write(source, "int original=2222;\n");
  require(!sizeWitness->matchesPhysical(), "changed physical size acquired witness");
  // A header parsed as its own host main has the identical physical identity
  // contract; its extension confers no permission and cannot hide replacement.
  const fs::path header = directory / "selected.h";
  const std::string record = "#pragma pack(1)\nstruct P { unsigned short x; };\n#pragma pack()\n";
  write(header, record);
  auto headerWitness = ascify::HostCxxRootInputWitness::capture(header.string());
  require(bool(headerWitness) && headerWitness->matchesParsed(
      header.string(), record, record.size()), "standalone header identity refused");
  require(!headerWitness->matchesParsed(source.string(), record, record.size()),
          "different parsed path acquired header identity");
  require(!headerWitness->matchesParsed(header.string(), record + "\n", record.size()),
          "forged parsed header buffer acquired identity");
  const fs::path headerReplacement = directory / "replacement.h";
  write(headerReplacement, record);
  fs::rename(headerReplacement, header);
  require(!headerWitness->matchesPhysical() &&
          !headerWitness->matchesParsed(header.string(), record, record.size()),
          "same-content header inode replacement acquired identity");
  std::cout << "Host C++ root witness: exact input, path/size/content/UID/alias rejection PASS\n";
}
