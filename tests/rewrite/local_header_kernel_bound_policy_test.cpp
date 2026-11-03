// Direct tests of the production transaction API, not a Python state model.
// Prepared only: compile/link/run must be authorized separately.
#include "LocalHeader.h"
#include "llvm/Support/SHA256.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdlib>
#include <iterator>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
namespace fs = std::filesystem;
extern const std::string sAscify="policy-test ";
extern const std::string sError="error: ";
extern const std::string sConflict="conflict: ";
extern const std::string sWarning="warning: ";
// LocalHeader.cpp's CLI driver is not exercised by this standalone API test.
bool ascifySingleSource(const std::string&, const std::string&,
    const ct::CompilationDatabase*, ct::CommonOptionsParser*, const char*,
    const std::string&, bool, LocalHeaderRewriteContext*) { assert(false); return false; }
static void write(const fs::path& p, const std::string& text) {
  fs::create_directories(p.parent_path()); std::ofstream(p) << text;
}
static std::string hash(const std::string& text) {
  llvm::SHA256 digest; digest.update(text); const auto bytes=digest.final();
  const char hex[]="0123456789abcdef"; std::string result;
  for (const auto byte : bytes) { const unsigned v=static_cast<unsigned char>(byte); result+=hex[v>>4]; result+=hex[v&15]; }
  return result;
}
struct Fixture {
  fs::path base, root, header, output, bundle, stagedRoot, stagedBundle, stagedHeader;
  std::string original="__global__ void kernel(int*p){}\n";
  LocalHeaderClosurePlan plan;
  LocalKernelThreadBoundPolicy policy;
  static fs::path temp() { char p[]="/tmp/ascify-bound-policy.XXXXXX"; auto* r=mkdtemp(p); assert(r); return r; }
  Fixture() : base(temp()), root(base/"src/main.cu"), header(base/"src/kernel.cuh"),
      output(base/"out/main.cce"), bundle(base/"out/main.cce.headers"),
      stagedRoot(base/"stage/main.cce"), stagedBundle(base/"stage/headers"),
      plan(prepare(),output.string(),bundle.string(),LocalHeaderClosureMode::Recursive) {
    const auto include=plan.observeInclude(root.string(),output.string(),0,0,"kernel.cuh",header.string(),false,true);
    assert(include.redirect && plan.nodes().size()==1);
    const auto& node=plan.nodes().front(); stagedHeader=stagedBundle/node.artifactRelativePath;
    policy.sourcePath=fs::canonical(header).string(); policy.sourceSha256=hash(original);
    policy.sourceBytes=original.size(); policy.kernelName="kernel"; policy.declarationOffset=16;
    policy.qualifierOffset=0; policy.maximum=1024; policy.artifactPath=node.artifactPath;
    write(stagedRoot,"#include \""+include.emittedSpelling+"\"\n");
    write(stagedHeader,"ASCIFY_GLOBAL __launch_bounds__(1024) void kernel(int*p){}\n");
    write(output,"PREEXISTING ROOT\n"); write(bundle/"previous","PREEXISTING BUNDLE\n");
    plan.markTranslating(0); plan.markStaged(0);
  }
  std::string prepare() { write(root,"#include \"kernel.cuh\"\n"); write(header,original); return root.string(); }
  ~Fixture() { fs::remove_all(base); }
  LocalKernelThreadBoundPolicies applied() const { return {{{policy.sourcePath,policy.qualifierOffset},policy}}; }
  bool seal() { return plan.sealKernelThreadBoundPolicies(policy.sourcePath,policy.artifactPath,stagedHeader.string(),applied()); }
  bool validate() { return plan.validateStagedArtifacts(stagedRoot.string(),stagedBundle.string()); }
  void oldOutputs() {
    std::ifstream rootFile(output), bundleFile(bundle/"previous");
    assert(std::string(std::istreambuf_iterator<char>(rootFile), {})=="PREEXISTING ROOT\n");
    assert(std::string(std::istreambuf_iterator<char>(bundleFile), {})=="PREEXISTING BUNDLE\n");
  }
};
int main() {
  { Fixture f; assert(f.plan.requireKernelThreadBoundPolicy(f.policy)); assert(f.seal()); assert(f.validate()); f.oldOutputs(); }
  { Fixture f; assert(f.plan.requireKernelThreadBoundPolicy(f.policy)); assert(!f.validate()); f.oldOutputs(); } // Missing fulfillment blocks publication.
  { Fixture f; assert(f.plan.requireKernelThreadBoundPolicy(f.policy)); auto other=f.policy; other.maximum=512;
    assert(!f.plan.requireKernelThreadBoundPolicy(other)); assert(!f.validate()); f.oldOutputs(); } // Same path/offset conflict.
  { Fixture f; assert(f.plan.requireKernelThreadBoundPolicy(f.policy)); auto changed=f.original; changed.back()=' '; write(f.header,changed);
    assert(!f.seal()); assert(!f.validate()); f.oldOutputs(); } // Same-size source change after root proof.
  { Fixture f; assert(f.plan.requireKernelThreadBoundPolicy(f.policy)); assert(f.seal()); write(f.header,f.original+"\n");
    assert(!f.validate()); f.oldOutputs(); } // Late source change after fulfillment.
  { Fixture f; assert(f.plan.requireKernelThreadBoundPolicy(f.policy)); write(f.stagedHeader,"ASCIFY_GLOBAL void kernel(int*p){}\n");
    assert(!f.seal()); assert(!f.validate()); f.oldOutputs(); } // Claimed edit is absent from actual artifact.
  { Fixture f; assert(f.plan.requireKernelThreadBoundPolicy(f.policy)); assert(f.seal()); write(f.stagedHeader,"CHANGED ARTIFACT\n");
    assert(!f.validate()); f.oldOutputs(); } // Actual sealed artifact binding.
  { Fixture f; assert(f.plan.requireKernelThreadBoundPolicy(f.policy));
    const auto otherHeader=f.header.parent_path()/"other.cuh"; write(otherHeader,f.original);
    assert(f.plan.observeInclude(f.root.string(),f.output.string(),0,1,"other.cuh",otherHeader.string(),false,true).redirect);
    auto second=f.policy; second.sourcePath=fs::canonical(otherHeader).string();
    second.artifactPath=f.plan.nodes().at(1).artifactPath;
    assert(f.plan.requireKernelThreadBoundPolicy(second)); assert(f.seal());
    assert(!f.plan.validateKernelThreadBoundPolicies(f.stagedBundle.string())); f.oldOutputs();
  } // Identical name/hash/offset in another physical file does not inherit fulfillment.
}
