#include "StandaloneDeviceTarget.h"
#include "ArgParse.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/Allocator.h"
#include "llvm/ADT/SmallVector.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/TargetParser/Triple.h"
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <vector>
#include <memory>
#include <optional>
#include <sstream>
#include <cctype>
#include <cstdint>
#include <algorithm>
#include <tuple>
#include <cstring>
namespace ascify { namespace {
namespace fs=std::filesystem;
bool safePath(llvm::StringRef path){return !path.empty()&&std::none_of(path.begin(),path.end(),[](char c){const auto x=static_cast<unsigned char>(c);return x<32||x==127;});}
std::string physicalPath(const fs::path& p){if(!safePath(p.string()))return {};std::error_code ec;auto q=fs::canonical(p,ec);return ec||!safePath(q.string())?std::string():q.string();}
bool read(const fs::path& p,std::string& s){
 const auto buffer=llvm::MemoryBuffer::getFileAsStream(p.string());
 if(!buffer)return false;
 s=(*buffer)->getBuffer().str();
 return true;
}
std::string hash(llvm::StringRef s){llvm::SHA256 h;h.update(s);auto data=h.final();std::string out;const char hex[]="0123456789abcdef";for(auto x:data){out+=hex[(static_cast<unsigned char>(x)>>4)&15];out+=hex[static_cast<unsigned char>(x)&15];}return out;}
bool within(const std::string& path,const std::string& root){return path.size()>root.size()&&path.compare(0,root.size(),root)==0&&path[root.size()]=='/';}
struct Target {
 std::string root,output,bundle,cann,product,compiler,compilerResource,activeStagedRoot,activeBundle,privateLinkTemporary;
 std::string privateTemporaryPhysical;
 std::optional<llvm::sys::fs::UniqueID> privateTemporaryIdentity;
 std::vector<std::string> ownedLinkPlan;
 struct GeneratedPhysical {
  std::string canonical,digest; llvm::sys::fs::UniqueID uid; std::uint64_t size;
 };
 std::map<std::string,GeneratedPhysical> generatedPhysical;
 bool generatedMatch()const{
  for(const auto& item:generatedPhysical){const auto& expected=item.second;
   std::error_code ec;const auto status=fs::symlink_status(item.first,ec);
   llvm::sys::fs::UniqueID uid;std::string bytes;
   if(ec||!fs::is_regular_file(status)||fs::is_symlink(status)||physicalPath(item.first)!=expected.canonical||
      llvm::sys::fs::getUniqueID(item.first,uid)||uid!=expected.uid||!read(item.first,bytes)||
      bytes.size()!=expected.size||hash(bytes)!=expected.digest)return false;
  }return true;
 }
 bool rememberGeneratedPhysical(const fs::path& path){
  const auto spelling=path.string();std::error_code ec;const auto status=fs::symlink_status(path,ec);
  const auto canonical=physicalPath(path);llvm::sys::fs::UniqueID first,last;std::string bytes;
  if(ec||canonical.empty()||!fs::is_regular_file(status)||fs::is_symlink(status)||
     llvm::sys::fs::getUniqueID(path.string(),first)||!read(path,bytes)||
     llvm::sys::fs::getUniqueID(path.string(),last)||first!=last)return false;
  const auto old=generatedPhysical.find(spelling);const auto digest=hash(bytes);
  if(old!=generatedPhysical.end())return old->second.canonical==canonical&&old->second.uid==first&&
     old->second.size==bytes.size()&&old->second.digest==digest;
  generatedPhysical.emplace(spelling,GeneratedPhysical{canonical,digest,first,bytes.size()});return true;
 }
 std::map<std::string,std::string> sealedArtifacts,builtInputs;
 std::set<std::string> callers;
 std::vector<StandaloneDeviceIncludeEdge> edges;bool edgesBound=false;
 std::map<std::string,std::vector<std::string>> parserArguments;
 std::map<std::string,std::string> parserDriverMacros,parserArchitectures,parserInputSpellings;std::string argumentFailure;
 std::map<std::string,std::string> inputs,bindings;
 std::map<std::string,llvm::sys::fs::UniqueID> identities,directoryIdentities;
 std::map<std::string,std::string> directoryBindings;
 std::set<std::string> absentDirectories;
 std::vector<std::pair<std::string,std::string>> searches;
 bool bound=false,consumed=false;
 bool matchesPrivateTemporary()const{
  if(privateLinkTemporary.empty())return privateTemporaryPhysical.empty()&&!privateTemporaryIdentity;
  llvm::sys::fs::file_status status;llvm::sys::fs::UniqueID uid;
  return privateTemporaryIdentity&&physicalPath(privateLinkTemporary)==privateTemporaryPhysical&&
   !llvm::sys::fs::status(privateLinkTemporary,status,false)&&llvm::sys::fs::is_directory(status)&&
   status.permissions()==llvm::sys::fs::owner_all&&!llvm::sys::fs::getUniqueID(privateLinkTemporary,uid)&&uid==*privateTemporaryIdentity;
 }
 bool remember(const fs::path& p){const auto path=physicalPath(p);std::string bytes;llvm::sys::fs::UniqueID before,after;
  if(path.empty()||llvm::sys::fs::getUniqueID(path,before)||!read(path,bytes)||llvm::sys::fs::getUniqueID(path,after)||before!=after)return false;
  std::error_code ec;const auto spelling=fs::absolute(p,ec).string();if(ec)return false;auto binding=bindings.find(spelling);if(binding!=bindings.end()&&binding->second!=path)return false;bindings[spelling]=path;
  const auto digest=hash(bytes);auto old=inputs.find(path);if(old!=inputs.end())return old->second==digest&&identities.at(path)==before;inputs[path]=digest;identities.emplace(path,before);return true;
 }
 bool rememberDirectory(const fs::path& p){std::error_code ec;const auto spelling=fs::absolute(p,ec).string();
  if(ec||!safePath(spelling))return false;llvm::sys::fs::file_status status;const auto state=llvm::sys::fs::status(spelling,status);
  if(state==std::errc::no_such_file_or_directory){if(directoryBindings.count(spelling))return false;absentDirectories.insert(spelling);return true;}
  if(state||!llvm::sys::fs::is_directory(status)||absentDirectories.count(spelling))return false;
  const auto physical=physicalPath(p);llvm::sys::fs::UniqueID uid;
  if(physical.empty()||llvm::sys::fs::getUniqueID(spelling,uid))return false;
  const auto old=directoryBindings.find(spelling);if(old!=directoryBindings.end())return old->second==physical&&directoryIdentities.at(spelling)==uid;
  directoryBindings[spelling]=physical;directoryIdentities.emplace(spelling,uid);return true;
 }
 bool matches()const{if(!matchesPrivateTemporary()||!generatedMatch())return false;for(const auto& p:absentDirectories){llvm::sys::fs::file_status status;if(llvm::sys::fs::status(p,status)!=std::errc::no_such_file_or_directory)return false;}
  for(const auto& p:directoryBindings){llvm::sys::fs::UniqueID uid;if(physicalPath(p.first)!=p.second||llvm::sys::fs::getUniqueID(p.first,uid)||uid!=directoryIdentities.at(p.first))return false;}
  for(const auto& p:bindings)if(physicalPath(p.first)!=p.second)return false;
  for(const auto& p:inputs){std::string bytes;llvm::sys::fs::UniqueID uid;if(llvm::sys::fs::getUniqueID(p.first,uid)||uid!=identities.at(p.first)||!read(p.first,bytes)||hash(bytes)!=p.second)return false;}return true;
 }
 std::string stamp()const{std::string text;for(const auto& p:inputs)text+=p.first+"\n"+p.second+"\n"+std::to_string(identities.at(p.first).getDevice())+":"+std::to_string(identities.at(p.first).getFile())+"\n";return hash(text);}
};
std::unique_ptr<Target> target;
const std::vector<std::string> libraries={"ascendcl","runtime","register","error_manager","profapi","ascendalog","mmpa","ascend_dump","c_sec","msprofiler"};
bool execute(Target& t,const fs::path& directory,const std::string& label,const std::vector<std::string>& storage,std::string& error){
 if(!t.matches()){error="owned source/provider/generated/temporary identity changed before subprocess";return false;}
 std::vector<llvm::StringRef> argv;for(const auto& s:storage)argv.push_back(s);
 std::vector<std::string> environment={"PATH="+t.cann+"/tools/bisheng_compiler/bin:/usr/bin:/bin","LD_LIBRARY_PATH="+t.cann+"/lib64:"+t.cann+"/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common","ASCEND_HOME_PATH="+t.cann,"ASCEND_OPP_PATH="+t.cann+"/opp"};
 if(!t.privateLinkTemporary.empty())environment.push_back("TMPDIR="+t.privateLinkTemporary);
 std::vector<llvm::StringRef> env;for(const auto& s:environment)env.push_back(s);
 llvm::json::Array args;for(const auto& s:storage)args.push_back(s);{std::ofstream out(directory/(label+".argv.json"));out<<llvm::formatv("{0:2}",llvm::json::Value(std::move(args))).str()<<"\n";}
 const std::string stdoutPath=(directory/(label+".stdout.log")).string(),stderrPath=(directory/(label+".stderr.log")).string();
 const std::optional<llvm::StringRef> redirects[]={std::nullopt,llvm::StringRef(stdoutPath),llvm::StringRef(stderrPath)};
 bool failed=false;std::string why;const int rc=llvm::sys::ExecuteAndWait(storage.front(),argv,env,redirects,300,0,&why,&failed);
 {std::ofstream out(directory/(label+".process.json"));out<<"{\"returncode\":"<<rc<<",\"execution_failed\":"<<(failed?"true":"false")<<"}\n";}
 std::string diagnostics;read(stderrPath,diagnostics);
 if(rc||failed||diagnostics.find(": error:")!=std::string::npos||diagnostics.find("\nerror:")!=std::string::npos){error="owned standalone "+label+" failed; logs: "+directory.string()+" "+why;return false;}
 if(!t.matches()){error="owned source/provider/generated/temporary identity changed during subprocess";return false;}return true;
}
bool rememberTree(Target& t,const fs::path& root,std::string& error){
 std::error_code ec;
 for(fs::recursive_directory_iterator it(root,ec),end;!ec&&it!=end;it.increment(ec)){
  if(it->is_regular_file(ec)&&!t.remember(it->path())){error="changed/unreadable owned toolchain file: "+it->path().string();return false;}
 }
 if(ec){error="cannot enumerate owned toolchain inputs: "+root.string();return false;}return true;
}
bool dependencyRules(const fs::path& path,std::vector<std::set<std::string>>& rules,std::string& error){
 std::string data;if(!read(path,data)){error="missing actual compiler dependency output";return false;}
 const std::string prefix="ascify-owned-target:";std::size_t offset=0;
 while(offset<data.size()){
  while(offset<data.size()&&std::isspace(static_cast<unsigned char>(data[offset])))++offset;
  if(offset==data.size())break;
  if(data.compare(offset,prefix.size(),prefix)!=0){error="unexpected compiler stdout/dependency target";return false;}offset+=prefix.size();
  std::set<std::string> paths;std::string item;bool escape=false;
  const auto add=[&](){if(item.empty())return true;const fs::path p(item);if(!p.is_absolute()||physicalPath(p).empty())return false;paths.insert(item);item.clear();return true;};
  for(;offset<data.size();){const char c=data[offset++];
   if(escape){escape=false;if(c!='\n')item+=c;continue;}
   if(c=='\\'){escape=true;continue;}
   if(c=='$'){if(offset>=data.size()||data[offset++]!='$'){error="unsupported make dependency substitution";return false;}item+='$';continue;}
   if(c=='\n'){if(!add()){error="nonphysical compiler dependency";return false;}break;}
   if(std::isspace(static_cast<unsigned char>(c))){if(!add()){error="nonphysical compiler dependency";return false;}}
   else if(c==':'||c=='#'){error="unsupported make dependency spelling";return false;}else item+=c;
  }
  if(escape||!add()||paths.empty()){error="incomplete compiler dependency rule";return false;}rules.push_back(std::move(paths));
 }
 if(rules.empty()){error="missing compiler dependency rules";return false;}return true;
}
bool dependencyList(const fs::path& path,std::set<std::string>& paths,std::string& error){
 std::vector<std::set<std::string>> rules;if(!dependencyRules(path,rules,error))return false;
 if(rules.size()!=1){error="compiler dependency query must have exactly one rule";return false;}paths=std::move(rules.front());return true;
}
// A compiler query is still run through the same fixed environment and raw
// log collector. No JSON supplied by a caller grants an executable capability.
bool rememberDynamicProviders(Target& t,const fs::path& directory,const std::string& binary,unsigned& index,std::string& error){
 const std::string ldd="/usr/bin/ldd";if(!t.remember(ldd)){error="missing physical provider-query tool";return false;}
 const auto label="providers-"+std::to_string(index++);
 if(!execute(t,directory,label,{ldd,binary},error))return false;
 std::string output;if(!read(directory/(label+".stdout.log"),output))return false;
 std::istringstream lines(output);std::string line;
 while(std::getline(lines,line)){
  const auto arrow=line.find("=>");const auto start=arrow==std::string::npos?line.find_first_not_of(" \t"):line.find_first_not_of(" \t",arrow+2);
  if(start==std::string::npos)continue;
  if(line.compare(start,9,"not found")==0){error="unresolved actual dynamic provider";return false;}
  if(line[start]!='/')continue;
  const auto end=line.find_first_of(" \t",start);const auto path=line.substr(start,end-start);
  if(!t.remember(path)){error="changed physical dynamic provider: "+path;return false;}
 }
 return true;
}
// These are the two actual GNU AArch64 scripts selected by the fixed native
// toolchain, not a general interpreter for caller-provided linker scripts.
bool rememberNativeScriptInputs(Target& t,llvm::StringRef bytes,const std::vector<fs::path>& searches,std::string& error){
 const char libc[]="/* GNU ld script\n   Use the shared library, but some functions are only in\n   the static library, so try that secondarily.  */\nOUTPUT_FORMAT(elf64-littleaarch64)\nGROUP ( /lib/aarch64-linux-gnu/libc.so.6 /usr/lib/aarch64-linux-gnu/libc_nonshared.a  AS_NEEDED ( /lib/ld-linux-aarch64.so.1 ) )\n";
 const char gcc[]="/* GNU ld script\n   Use the shared library, but some functions are only in\n   the static library.  */\nGROUP ( libgcc_s.so.1 -lgcc )\n";
 if(bytes==libc){for(const auto path:{"/lib/aarch64-linux-gnu/libc.so.6","/usr/lib/aarch64-linux-gnu/libc_nonshared.a","/lib/ld-linux-aarch64.so.1"})if(!t.remember(path)){error="changed fixed GNU libc script provider";return false;}return true;}
 if(bytes==gcc){
  for(const auto name:{"libgcc_s.so.1","libgcc.a"}){bool found=false;for(const auto& dir:searches){std::error_code ec;const auto path=dir/name;if(!fs::is_regular_file(path,ec))continue;if(!t.remember(path)){error="changed fixed GNU gcc script provider";return false;}found=true;break;}if(!found){error="missing fixed GNU gcc script provider";return false;}}
  return true;
 }
 error="unmapped native linker script";return false;
}
bool rememberDriverPlan(Target& t,const fs::path& directory,const std::string& label,const std::vector<std::string>& command,unsigned& providers,std::string& error,bool stdoutDependencies=false){
 auto query=command;query.push_back("-###");if(!execute(t,directory,label,query,error))return false;
 std::string output;if(!read(directory/(label+".stderr.log"),output))return false;
 std::istringstream lines(output);std::string line;unsigned commands=0;
 while(std::getline(lines,line)){
  const auto first=line.find_first_not_of(" \t");if(first==std::string::npos||line[first]!='"')continue;
  llvm::BumpPtrAllocator alloc;llvm::StringSaver saver(alloc);llvm::SmallVector<const char*,64> args;
  llvm::cl::TokenizeGNUCommandLine(line,saver,args);
  if(args.empty()||!fs::path(args.front()).is_absolute()||!t.remember(args.front())){error="driver selected a nonphysical subprocess";return false;}
  const auto tool=physicalPath(args.front());
  const bool compilerRole=tool==physicalPath(t.compiler)&&args.size()>1&&llvm::StringRef(args[1])=="-cc1";
  const bool systemLinkerRole=tool=="/usr/bin/aarch64-linux-gnu-ld.bfd"||tool=="/usr/bin/aarch64-linux-gnu-ld"||tool=="/usr/bin/ld.bfd"||tool=="/usr/bin/ld";
  const auto cceLinker=t.cann+"/tools/bisheng_compiler/bin/cce-ld",lld=t.cann+"/tools/bisheng_compiler/bin/ld.lld";
  const bool ownedLinkerRole=llvm::StringRef(args.front())==cceLinker&&tool==physicalPath(cceLinker)&&args.size()>1&&llvm::StringRef(args[1])==lld&&!physicalPath(lld).empty();
  const bool linkerRole=systemLinkerRole||ownedLinkerRole;
  if(!compilerRole&&!linkerRole){error="unmapped driver subprocess role: "+tool;return false;}
  if(compilerRole){
   std::string resource;unsigned count=0;
   for(std::size_t i=1;i<args.size();++i)if(llvm::StringRef(args[i])=="-resource-dir"){
    if(++count!=1||i+1==args.size()){error="ambiguous owned compiler resource plan";return false;}resource=args[++i];
   }
   const auto physicalResource=physicalPath(resource),physicalCompiler=physicalPath(t.cann+"/tools/bisheng_compiler");
   if(count!=1||!fs::path(resource).is_absolute()||physicalResource.empty()||physicalCompiler.empty()||!within(physicalResource,physicalCompiler)||!t.rememberDirectory(resource)||(!t.compilerResource.empty()&&t.compilerResource!=resource)){
    error="owned compiler resource changed or left its sealed toolchain";return false;
   }t.compilerResource=resource;
  }
  if(ownedLinkerRole){
   if(t.compilerResource.empty()||!t.remember(args[1])||!rememberDynamicProviders(t,directory,args[1],providers,error)){if(error.empty())error="owned linker backend is changed or unbound";return false;}
   const std::set<std::string> switches={"-x","-cce-aicore-arch=dav-c310-vec","-pie","-EL","--hash-style=gnu","--eh-frame-hdr","--as-needed","--no-as-needed","-t"};
   const auto compilerRoot=physicalPath(t.cann+"/tools/bisheng_compiler");
   std::set<std::string> names(libraries.begin(),libraries.end());names.insert({"stdc++","m","gcc","gcc_s","pthread","c"});
   unsigned reproductions=0;
   for(std::size_t i=2;i<args.size();++i){const llvm::StringRef arg(args[i]);
    const auto value=[&]()->std::string{return i+1<args.size()?args[++i]:std::string();};
    if(switches.count(arg.str()))continue;
    if(arg=="-z"){if(value()=="relro")continue;error="unmapped owned linker -z mode";return false;}
    if(arg=="-m"){if(value()=="aarch64linux")continue;error="unmapped owned linker emulation";return false;}
    if(arg=="-o"){if(value()==(directory/"target").string())continue;error="owned linker output is outside its target";return false;}
    if(arg=="-dynamic-linker"){if(value()=="/lib/ld-linux-aarch64.so.1")continue;error="unmapped owned program loader";return false;}
    if(arg=="-rpath"){const auto path=value();if((path==t.cann+"/lib64"||path==t.cann+"/aarch64-linux/lib64")&&t.rememberDirectory(path))continue;error="unmapped owned linker runtime search";return false;}
    if(arg=="-cce-stub-dir"||arg=="-cce-install-dir"){const auto path=value(),expected=arg=="-cce-stub-dir"?t.compilerResource+"/include/cce_stub":t.cann+"/tools/bisheng_compiler/bin";
     if(path==expected&&t.rememberDirectory(path))continue;error="unmapped owned linker resource directory";return false;}
    if(arg=="--reproduce"){if(++reproductions==1&&value()==(directory/"link-inputs.tar").string())continue;error="unmapped owned linker reproduction artifact";return false;}
    if(arg=="-L"||arg.starts_with("-L")){const auto path=arg=="-L"?value():arg.drop_front(2).str();if(fs::path(path).is_absolute()&&t.rememberDirectory(path))continue;error="unbound owned linker library search";return false;}
    if(arg.starts_with("-l")&&arg.size()>2){if(names.count(arg.drop_front(2).str()))continue;error="unmapped owned linker library name";return false;}
    if(fs::path(arg.str()).is_absolute()&&safePath(arg)){const auto path=physicalPath(arg.str());
     if(arg==(directory/"target.o").string()||within(path,compilerRoot)||within(path,t.cann+"/lib64")||within(path,t.cann+"/aarch64-linux/lib64")||within(path,"/usr/lib")||within(path,"/lib"))continue;
    }
    error="unmapped owned cce-ld/ld.lld argument: "+arg.str();return false;
   }
   if(reproductions!=1){error="owned linker did not seal its actual internal inputs";return false;}
   std::vector<std::string> plan;for(const auto* arg:args)plan.emplace_back(arg);
   if(!t.ownedLinkPlan.empty()&&t.ownedLinkPlan!=plan){error="owned linker plan changed";return false;}t.ownedLinkPlan=std::move(plan);
  }
  if(stdoutDependencies){
   const char* triples[]={"hiipu64-hisilicon-cce","aarch64-unknown-linux-gnu"};
   std::map<std::string,std::string> options;
   for(std::size_t i=1;i<args.size();++i){const std::string option(args[i]);if(option!="-triple"&&option!="-dependency-file"&&option!="-MT")continue;
    if(i+1==args.size()||!options.emplace(option,args[++i]).second){error="ambiguous actual dependency driver options";return false;}
   }
   if(!compilerRole||commands>=2||options["-triple"]!=triples[commands]||options["-dependency-file"]!="-"||options["-MT"]!="ascify-owned-target"){
    error="actual compiler plan does not preserve ordered device/host stdout dependency rules";return false;
   }
  }
  if(!rememberDynamicProviders(t,directory,args.front(),providers,error))return false;
  ++commands;
  // Only a real linker command gives -L/-l their library-search meaning.
  // In particular, cc1's -mllvm -licm-safe-hoist=false is not a library.
  if(linkerRole){
  std::vector<fs::path> libraryPaths;std::vector<std::string> names;
  for(std::size_t i=1;i<args.size();++i){const llvm::StringRef arg(args[i]);
   if(arg=="-L"&&i+1<args.size())libraryPaths.emplace_back(args[++i]);
   else if(arg.starts_with("-L")&&arg.size()>2)libraryPaths.emplace_back(arg.drop_front(2).str());
   else if(arg.starts_with("-l")&&arg.size()>2)names.push_back(arg.drop_front(2).str());
  }
  for(const auto& name:names){bool found=false;
   for(const auto& dir:libraryPaths){for(const auto& leaf:{"lib"+name+".so","lib"+name+".a"}){
    const auto path=dir/leaf;std::error_code ec;if(!fs::is_regular_file(path,ec))continue;
    if(!t.remember(path)){error="changed driver library provider";return false;}
    std::string bytes;if(!read(path,bytes))return false;
    if(bytes.size()>4&&bytes.compare(0,4,std::string("\177ELF",4))==0&&!rememberDynamicProviders(t,directory,path.string(),providers,error))return false;
    if(bytes.compare(0,4,std::string("\177ELF",4))!=0&&bytes.compare(0,8,"!<arch>\n")!=0&&!rememberNativeScriptInputs(t,bytes,libraryPaths,error))return false;
    found=true;break;
   }if(found)break;}
   if(!found){error="driver library selection is outside explicit search paths: "+name;return false;}
  }
  }
  for(std::size_t i=1;i<args.size();++i){const fs::path p(args[i]);std::error_code ec;
   if(llvm::StringRef(args[i]).starts_with("@")){error="response files are outside the owned driver plan";return false;}
   if(p.is_absolute()&&p.string()!=t.activeStagedRoot&&!within(p.string(),t.activeBundle)&&fs::is_regular_file(p,ec)&&!t.remember(p)){error="changed physical driver input";return false;}
  }
 }
 if(!commands||(stdoutDependencies&&commands!=2)){error="actual driver did not provide the required subprocess plan";return false;}return true;
}
// Read the observed LLD USTAR dialect in memory. Never extract a member using
// its pathname, or treat an archive member as source/native ownership evidence.
bool readOwnedUstar(const fs::path& path,std::map<std::string,std::string>& members,std::string& error){
 std::string data;if(!read(path,data)||data.size()>128u*1024u*1024u||data.size()%512){error="invalid/oversized owned linker reproduction";return false;}
 const auto number=[](llvm::StringRef field,std::size_t& out){out=0;bool digit=false,tail=false;for(const char c:field){if(c=='\0'||c==' '){tail=true;continue;}if(tail||c<'0'||c>'7'||out>(static_cast<std::size_t>(-1)-7)/8)return false;digit=true;out=out*8+(c-'0');}return digit;};
 const auto text=[](llvm::StringRef field,std::string& out){const auto end=field.find('\0');out=field.take_front(end).str();return end==llvm::StringRef::npos||std::all_of(field.begin()+end,field.end(),[](char c){return c=='\0';});};
 std::size_t offset=0;
 while(offset+512<=data.size()){
  const llvm::StringRef header(data.data()+offset,512);
  if(std::all_of(header.begin(),header.end(),[](char c){return c=='\0';})){
   if(data.size()-offset<1024||members.empty()||!std::all_of(data.begin()+offset,data.end(),[](char c){return c=='\0';})){error="invalid owned USTAR terminator";return false;}return true;
  }
  std::size_t size=0,checksum=0,sum=0;std::string name,prefix,link;
  for(std::size_t i=0;i<512;++i)sum+=i>=148&&i<156?' ':static_cast<unsigned char>(header[i]);
  if(header.substr(257,8)!=llvm::StringRef("ustar\00000",8)||(header[156]!='\0'&&header[156]!='0')||!text(header.take_front(100),name)||!text(header.substr(345,155),prefix)||!text(header.substr(157,100),link)||!link.empty()||!number(header.substr(124,12),size)||!number(header.substr(148,8),checksum)||checksum!=sum){error="unmapped/nonregular owned USTAR member";return false;}
  if(!prefix.empty())name=prefix+"/"+name;
  if(!safePath(name)||fs::path(name).is_absolute()||name.find("//")!=std::string::npos||name.back()=='/'||members.size()>=1024){error="unsafe owned USTAR name";return false;}
  for(const auto& part:fs::path(name))if(part=="."||part==".."){error="unmapped owned USTAR path component";return false;}
  offset+=512;if(size>data.size()-offset){error="truncated owned USTAR body";return false;}
  const auto padding=(512-size%512)%512;
  if(padding>data.size()-offset-size||!std::all_of(data.begin()+offset+size,data.begin()+offset+size+padding,[](char c){return c=='\0';})||!members.emplace(name,data.substr(offset,size)).second){error="duplicate/unpadded owned USTAR body";return false;}
  offset+=size+padding;
 }
 error="missing owned USTAR terminator";return false;
}
bool ownedRegistrationObject(llvm::StringRef bytes,std::string& error){
 auto parsed=llvm::object::ObjectFile::createObjectFile(llvm::MemoryBufferRef(bytes,"owned linker registration output"));
 if(!parsed){llvm::consumeError(parsed.takeError());error="malformed owned registration object";return false;}
 auto& object=**parsed;
 if(object.getFileFormatName()!="elf64-littleaarch64"||object.getArch()!=llvm::Triple::aarch64||!object.isRelocatableObject()){error="unmapped owned registration object ABI";return false;}
 unsigned binary=0,file=0;
 for(const auto& section:object.sections()){auto name=section.getName();if(!name){llvm::consumeError(name.takeError());error="malformed owned registration section";return false;}if(*name==".aicore_binary"){if(!section.getSize()||++binary!=1){error="ambiguous owned registration device payload";return false;}}}
 for(const auto& symbol:object.symbols()){auto name=symbol.getName();auto type=symbol.getType();if(!name||!type){if(!name)llvm::consumeError(name.takeError());if(!type)llvm::consumeError(type.takeError());error="malformed owned registration symbol";return false;}if(*type==llvm::object::SymbolRef::ST_File){if(*name!="__clang_cce_link.stub"||++file!=1){error="unmapped owned registration source role";return false;}}}
 if(binary!=1||file!=1){error="missing owned registration stub/payload role";return false;}return true;
}
bool verifyOwnedLinkReproduction(Target& t,const fs::path& directory,const std::string& object,std::map<std::string,std::string>& generated,std::string& error){
 std::map<std::string,std::string> members;if(!readOwnedUstar(directory/"link-inputs.tar",members,error))return false;
 const std::string prefix="link-inputs/";std::string intermediate,intermediateBytes,response;bool sawObject=false,sawVersion=false;
 for(const auto& member:members){
  if(member.first.compare(0,prefix.size(),prefix)!=0){error="owned USTAR member left its fixed archive root";return false;}
  const auto name=member.first.substr(prefix.size());
  if(name=="response.txt"){response=member.second;continue;}
  if(name=="version.txt"){if(member.second!="LLD 15.0.5\n"){error="unmapped actual owned linker version";return false;}sawVersion=true;continue;}
  const auto requested="/"+name;
  if(requested==object){if(hash(member.second)!=generated.at(object)){error="reproducer selected another owned object";return false;}sawObject=true;continue;}
  if(fs::path(requested).parent_path()==fs::path(t.privateLinkTemporary)){
   const auto leaf=fs::path(requested).filename().string();const llvm::StringRef token(leaf);
   if(!intermediate.empty()||!token.starts_with("cce_all_link-")||!token.ends_with(".o")||token.size()!=21||!std::all_of(token.begin()+13,token.end()-2,[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');})||!ownedRegistrationObject(member.second,error)){if(error.empty())error="unmapped owned temporary registration output";return false;}
   intermediate=requested;intermediateBytes=member.second;continue;
  }
  const auto binding=t.bindings.find(requested);
  if(binding==t.bindings.end()||physicalPath(requested)!=binding->second||t.inputs.at(binding->second)!=hash(member.second)){error="reproducer selected an unsealed static input: "+requested;return false;}
 }
 if(!sawObject||!sawVersion||intermediate.empty()||response.empty()||t.ownedLinkPlan.empty()){error="incomplete owned linker reproduction";return false;}
 // This is the real observed normalization performed by this sealed LLD15
 // dialect, preserving the validated outer tuple's order and selected inputs.
 std::vector<std::string> expected={"--chroot","."};
 for(std::size_t i=2;i<t.ownedLinkPlan.size();++i){const std::string& arg=t.ownedLinkPlan[i];const llvm::StringRef token(arg);
  if(token=="-cce-aicore-arch=dav-c310-vec")continue;
  if(token=="-cce-stub-dir"||token=="-cce-install-dir"||token=="--reproduce"){++i;continue;}
  if(token=="-o"){expected.insert(expected.end(),{"-o",fs::path(t.ownedLinkPlan[++i]).filename().string()});continue;}
  if(token=="-dynamic-linker"){expected.insert(expected.end(),{"--dynamic-linker",t.ownedLinkPlan[++i]});continue;}
  if(token=="-rpath"){expected.insert(expected.end(),{"--rpath",t.ownedLinkPlan[++i].substr(1)});continue;}
  if(token=="-L"){expected.insert(expected.end(),{"-L",t.ownedLinkPlan[++i].substr(1)});continue;}
  if(token.starts_with("-L")){expected.insert(expected.end(),{"-L",token.drop_front(3).str()});continue;}
  if(token.starts_with("-l")){expected.insert(expected.end(),{"-l",token.drop_front(2).str()});continue;}
  if(token=="-x"){expected.push_back("--discard-all");continue;}
  if(token=="-pie"){expected.push_back("--pie");continue;}
  if(token=="-EL"){expected.push_back("--EL");continue;}
  if(token=="-t"){expected.push_back("--trace");continue;}
  if(token=="--hash-style=gnu"){expected.insert(expected.end(),{"--hash-style","gnu"});continue;}
  expected.push_back(fs::path(arg).is_absolute()?arg.substr(1):arg);
 }
 expected.push_back(intermediate.substr(1));
 for(const auto name:{"ascendc_runtime","ascendcl","runtime","error_manager","profapi","unified_dlog","mmpa","ascend_dump","c_sec","stdc++"})expected.insert(expected.end(),{"-l",name});
 expected.insert(expected.end(),{"-L",t.cann.substr(1)+"/lib64","-L",t.cann.substr(1)+"/lib64","-l","profapi"});
 llvm::BumpPtrAllocator alloc;llvm::StringSaver saver(alloc);llvm::SmallVector<const char*,128> parsed;llvm::cl::TokenizeGNUCommandLine(response,saver,parsed);
 std::vector<std::string> actual;for(const auto* a:parsed)actual.emplace_back(a);
 if(actual!=expected){error="actual internal linker response differs from its sealed outer tuple and native registration recipe";return false;}
 std::string trace;if(!read(directory/"link.stdout.log",trace)){error="missing actual linker input trace";return false;}
 std::istringstream lines(trace);std::string line;unsigned generatedTraces=0;
 while(std::getline(lines,line)){
  if(line.empty())continue;const auto paren=line.find('(');if(paren!=std::string::npos)line.resize(paren);
  if(!safePath(line)||!fs::path(line).is_absolute()||!members.count(prefix+line.substr(1))){error="actual linker trace is absent from its reproduction: "+line;return false;}
  if(line==intermediate)++generatedTraces;
 }
 if(generatedTraces!=1){error="owned registration output was not a unique actual linker input";return false;}
 const auto capture=directory/"wrapper-generated.o";std::error_code ec;
 {llvm::raw_fd_ostream out(capture.string(),ec,llvm::sys::fs::CD_CreateNew);if(ec){error="owned registration capture path was already present/uncreatable";return false;}out.write(intermediateBytes.data(),intermediateBytes.size());out.close();if(out.has_error()){out.clear_error();error="cannot seal owned registration output bytes";return false;}}
 generated[capture.string()]=hash(intermediateBytes);
 llvm::json::Object proof,memberHashes;for(const auto& member:members)memberHashes[member.first]=hash(member.second);
 proof["schema"]="ascify.owned-linker-reproduction.v1";proof["actual_wrapper_generated_input"]=intermediate;
 proof["capture_relative"]=capture.filename().string();proof["capture_sha256"]=hash(intermediateBytes);
 proof["temporary_requested_directory"]=t.privateLinkTemporary;proof["temporary_physical_directory"]=t.privateTemporaryPhysical;
 proof["temporary_directory_uid"]=std::to_string(t.privateTemporaryIdentity->getDevice())+":"+std::to_string(t.privateTemporaryIdentity->getFile());proof["temporary_directory_permissions"]="0700";
 proof["captured_source_role"]="only generated AArch64 registration object from the sealed cce-ld/ld.lld tuple and fresh private TMPDIR";
 proof["not_a_caller_or_native_input_capability"]=true;proof["actual_internal_response_sha256"]=hash(response);proof["member_sha256"]=std::move(memberHashes);
 {std::ofstream out(directory/"link-inputs.proof.json");out<<llvm::formatv("{0:2}",llvm::json::Value(std::move(proof))).str()<<"\n";if(!out){error="cannot seal actual owned linker reproduction proof";return false;}}
 if(!t.matchesPrivateTemporary()||!fs::is_empty(t.privateLinkTemporary,ec)||ec||!fs::remove(t.privateLinkTemporary,ec)||ec){error="owned linker temporary directory changed or retained unplanned outputs";return false;}
 t.privateLinkTemporary.clear();t.privateTemporaryPhysical.clear();t.privateTemporaryIdentity.reset();
 return true;
}
bool verifyCallerIncludeTree(const Target& t,const fs::path& tracePath,const std::map<std::string,std::string>& mapped,std::string& error){
 const auto artifact=[&](const std::string& source){const auto found=mapped.find(source);return found==mapped.end()?source:found->second;};
 std::set<std::string> callerArtifacts;for(const auto& caller:t.callers)callerArtifacts.insert(artifact(caller));
 std::map<std::string,std::vector<std::string>> expected,actual;
 for(const auto& edge:t.edges)expected[artifact(edge.parent)].push_back(artifact(edge.child));
 std::string trace;if(!read(tracePath,trace)){error="missing actual target include-edge trace";return false;}
 std::vector<std::string> stack={artifact(t.root)};bool sawRoot=false;
 std::istringstream lines(trace);std::string line;
 while(std::getline(lines,line)){
  std::size_t depth=0;while(depth<line.size()&&line[depth]=='.')++depth;
  if(!depth||depth>=line.size()||line[depth]!=' ')continue;
  const auto spelling=line.substr(depth+1),child=physicalPath(spelling);
  if(child.empty()){error="include trace has no physical selected provider";return false;}
  // The forced SDK wrapper has no printed parent. Its early native children
  // precede the first depth-one source include and cannot grant caller edges.
  if(!sawRoot&&depth!=1){if(callerArtifacts.count(child)){error="caller input entered through an unobservable forced include";return false;}continue;}
  if(depth==1)sawRoot=true;
  if(depth>stack.size()){error="incomplete actual target include parent stack";return false;}
  const auto parent=stack[depth-1];stack.resize(depth);stack.push_back(child);
  if(callerArtifacts.count(child))actual[parent].push_back(child);
 }
 if(!sawRoot||actual!=expected){error="actual target caller include edges differ from original offset-ordered physical selections";return false;}
 return true;
}
} // namespace
bool RecordStandaloneCallerEdges(const std::vector<StandaloneDeviceIncludeEdge>& edges,std::string& error){
 if(!target||!target->bound||target->edgesBound||!target->matches()){error="original caller edge authority is absent or replayed";return false;}
 for(const auto& edge:edges){auto value=edge;value.parent=physicalPath(edge.parent);value.child=physicalPath(edge.child);
  if(!target->inputs.count(value.parent)||!target->callers.count(value.child)||!target->remember(edge.parent)||!target->remember(edge.child)||edge.spelling.empty()){error="original caller edge lacks physical parser provenance";return false;}
  if(!edge.search.empty()&&!target->rememberDirectory(edge.search)){error="original caller search role changed";return false;}
  target->edges.push_back(std::move(value));
 }
 std::stable_sort(target->edges.begin(),target->edges.end(),[](const auto& a,const auto& b){return std::tie(a.parent,a.offset)<std::tie(b.parent,b.offset);});
 target->edgesBound=true;return true;
}
bool RecordStandaloneParserArguments(const std::string& source,const std::vector<std::string>& arguments,std::string& error){
 if(!target){error="owned target does not authorize this parser command";return false;}
 auto& t=*target;const auto physical=physicalPath(source);
 if(physical.empty()||t.parserArguments.count(physical)){error="standalone parser command lacks a unique physical source";t.argumentFailure=error;return false;}
 if(arguments.empty()||fs::path(arguments.front()).filename()!="clang-tool"||(fs::path(arguments.front()).has_parent_path()&&physicalPath(fs::path(arguments.front()).parent_path())!=physicalPath(t.product+"/bin"))){error="standalone parser requires the owned GNU CUDA tool entry";t.argumentFailure=error;return false;}
 if(Verbose){llvm::json::Array argv;for(const auto& a:arguments)argv.push_back(a);llvm::errs()<<"Ascify standalone pre-driver argv: "<<llvm::formatv("{0}",llvm::json::Value(std::move(argv)))<<"\n";}
 unsigned sourceInputs=0,languages=0,standards=0,architectures=0;std::string architecture;
 const auto fail=[&](const std::string& reason){error=reason;t.argumentFailure=error;return false;};
 const auto sameDirectory=[&](const std::string& spelling,const std::string& expected){const auto selected=physicalPath(spelling),required=physicalPath(expected);return safePath(spelling)&&!selected.empty()&&!required.empty()&&selected==required&&t.rememberDirectory(spelling);};
 for(std::size_t i=1;i<arguments.size();++i){const auto& option=arguments[i];const llvm::StringRef arg(option);
  const auto value=[&]()->std::string{return i+1<arguments.size()?arguments[++i]:std::string();};
  if(arg=="--")continue;
  if(arg=="-I-")return fail("legacy include search controls are outside standalone mapping");
  if(arg=="-x"){if(value()!="cuda")return fail("standalone parser language must be CUDA");++languages;continue;}
  if(arg=="-std=c++17"){++standards;continue;}
  if(arg=="-fsyntax-only"||arg=="--cuda-host-only"||arg=="-fno-delayed-template-parsing"||arg=="-Wno-pragma-once-outside-header"||arg=="-fgpu-rdc"||arg=="-v")continue;
  if(arg=="-I"||arg=="-iquote"||arg=="-isystem"||arg=="-idirafter"){if(!safePath(value()))return fail("invalid standalone include path spelling");continue;}
  bool include=false;for(const auto prefix:{"-iquote","-isystem","-idirafter","-I"})if(arg.starts_with(prefix)&&arg.size()>std::strlen(prefix)){if(!safePath(arg.drop_front(std::strlen(prefix))))return fail("invalid standalone include path spelling");include=true;break;}
  if(include)continue;
  if(arg=="--cuda-path"||arg.starts_with("--cuda-path=")){const auto path=arg=="--cuda-path"?value():arg.split('=').second.str();if(!sameDirectory(path,CudaPath.getValue()))return fail("standalone CUDA SDK argument differs from the physical configured SDK");continue;}
  if(arg=="-resource-dir"||arg.starts_with("-resource-dir=")){const auto path=arg=="-resource-dir"?value():arg.split('=').second.str();if(!sameDirectory(path,t.product+"/libexec/ascify/clang/23"))return fail("standalone parser resource is not this installed LLVM23 product resource");continue;}
  if(arg=="-include"){const auto path=value();if(path!="cuda_runtime.h"&&!(physicalPath(path)==physicalPath(t.product+"/libexec/ascify/frontend-compat/ascify-admitted-v1/host_math.h")&&t.remember(path)))return fail("forced include is not an owned SDK/profile input");continue;}
  if(arg.starts_with("--cuda-gpu-arch=sm_")){const auto number=arg.drop_front(llvm::StringRef("--cuda-gpu-arch=sm_").size());if(++architectures!=1||number.empty()||!std::all_of(number.begin(),number.end(),[](char c){return c>='0'&&c<='9';}))return fail("standalone parser supports one literal numerical CUDA architecture");architecture=number.str()+"0";continue;}
  if(!arg.empty()&&arg.front()!='-'){
   if(++sourceInputs!=1||!safePath(option)||!fs::path(option).is_absolute()||option!=source||physicalPath(option)!=physical||
      (physical!=t.root&&(!t.bound||!t.callers.count(physical)||!t.inputs.count(physical)))||!t.remember(option))return fail("standalone parser command must retain the exact selected physical source spelling and identity");
   t.parserInputSpellings[physical]=option;continue;
  }
  return fail("unmapped standalone parser argument: "+arg.split('=').first.str());
 }
 if(sourceInputs!=1||!languages||!standards||architectures!=1)return fail("standalone parser command lacks its explicit source/CUDA/C++17/sm_NN boundary");
 t.parserArchitectures[physical]=architecture;
 t.parserArguments.emplace(physical,arguments);return true;
}
bool PrepareStandaloneDeviceTarget(const std::string& root,const std::string& output,const std::string& bundle,const char* executable,std::string& error){
 if(target){error="one standalone executable target per invocation";return false;}
 auto t=std::make_unique<Target>();t->root=physicalPath(root);t->cann=physicalPath(StandaloneCannRoot.getValue());
 std::error_code ec;t->output=fs::absolute(output,ec).lexically_normal().string();t->bundle=fs::absolute(bundle,ec).lexically_normal().string();
 const auto binary=physicalPath(executable);t->product=fs::path(binary).parent_path().parent_path().string();t->compiler=t->cann+"/tools/bisheng_compiler/bin/ccec";
 if(ec||!safePath(t->output)||!safePath(t->bundle)||t->root.empty()||t->cann.empty()||output.empty()||!t->remember(t->root)||!t->remember(t->compiler)||!t->remember(t->product+"/include/ascify/ascify_cuda_compat.hpp")){error="installed product, physical source and actual CANN compiler required";return false;}
 if(!rememberTree(*t,t->cann+"/tools/bisheng_compiler",error))return false;
 for(const auto& name:libraries)if(!t->remember(t->cann+"/lib64/lib"+name+".so")){error="missing actual CANN provider: "+name;return false;}
 for(fs::recursive_directory_iterator it(t->product+"/include",ec),end;!ec&&it!=end;it.increment(ec)){if(it->is_regular_file(ec)&&!t->remember(it->path())){error="unreadable installed public header";return false;}}
 if(ec){error="cannot enumerate installed public headers";return false;}
 // This mode creates a new fixed executable target. It does not infer the
 // closure of any caller parent CMake project or imported device archive.
 target=std::move(t);return true;
 }
bool BindStandaloneDeviceInputs(clang::CompilerInstance& ci,const std::string& current,const std::string& root,bool rootParse,const std::function<bool(clang::SourceLocation)>& native,std::string& error){
 if(!target||target->root!=physicalPath(root)||!target->matches()){error="owned standalone target missing or changed";return false;}
 auto& t=*target;auto& sm=ci.getSourceManager();
 const auto& pp=ci.getPreprocessorOpts();const auto& lang=ci.getLangOpts();
 if(!t.argumentFailure.empty()||!t.parserArguments.count(physicalPath(current))){error="standalone parser arguments were not independently recorded before the Clang driver: "+t.argumentFailure;return false;}
 if(!pp.MacroIncludes.empty()||!lang.CPlusPlus17||lang.CPlusPlus20||lang.CPlusPlus23){error="standalone builder requires unmodified C++17 preprocessing options";return false;}
 // This LLVM23 CUDA driver synthesizes an architecture-list -D after the
 // product's raw argv boundary. Caller -D/-U, including the same spelling and
 // value, has already been rejected before the driver sees it.
 // A host-only invocation has no auxiliary CPU in CompilerInstance. The
 // complete allowlisted pre-driver argv selected one sm_NN explicitly. Every caller macro/config/ABI
 // override and unsupported equal-form forwarding option is outside this mode.
 const auto expected=std::string("__CUDA_ARCH_LIST__=")+t.parserArchitectures.at(physicalPath(current));
 if(pp.Macros.size()!=1||pp.Macros.front().second||pp.Macros.front().first!=expected){
  error="unproved compiler-synthesized CUDA architecture-list macro";return false;
 }
 t.parserDriverMacros[physicalPath(current)]=expected;
 const bool profiled=FrontendCompat.getValue()=="ascify-admitted-v1";
 const auto wrapper=physicalPath(t.product+"/libexec/ascify/clang/23/include/__clang_cuda_runtime_wrapper.h");
 const auto profileMath=physicalPath(t.product+"/libexec/ascify/frontend-compat/ascify-admitted-v1/host_math.h");
 if(pp.Includes.size()!=(profiled?3u:2u)||pp.Includes[0]!="__clang_cuda_runtime_wrapper.h"||pp.Includes[1]!="cuda_runtime.h"||(profiled&&physicalPath(pp.Includes[2])!=profileMath)){
  error="forced includes do not match the owned driver/SDK/profile recipe";return false;
 }
 bool directWrapper=false;
 for(auto it=sm.fileinfo_begin();it!=sm.fileinfo_end();++it){const auto id=sm.translateFile(it->first);if(id.isInvalid())continue;const auto location=sm.getLocForStartOfFile(id);
  if(physicalPath(sm.getFilename(location).str())!=wrapper)continue;
  const auto include=sm.getIncludeLoc(id);const auto* entry=sm.getFileEntryForID(id);std::string bytes;bool bad=false;llvm::sys::fs::UniqueID uid;
  if(!entry||include.isInvalid()||sm.getFileID(sm.getSpellingLoc(include))!=ci.getPreprocessor().getPredefinesFileID()||sm.isFileOverridden(entry)||!read(wrapper,bytes)||sm.getBufferData(id,&bad)!=bytes||bad||llvm::sys::fs::getUniqueID(wrapper,uid)||uid!=entry->getUniqueID()||!t.remember(wrapper)){
   error="compiler-synthesized CUDA wrapper lacks direct physical installed-resource provenance";return false;
  }
  directWrapper=true;
 }
 if(!directWrapper){error="installed CUDA wrapper was not the actual direct driver input";return false;}
 const auto currentPath=physicalPath(current);std::string actual;
 bool invalid=false;const auto mainFile=sm.getMainFileID();const auto main=sm.getBufferData(mainFile,&invalid);
 const auto* entry=sm.getFileEntryForID(mainFile);llvm::sys::fs::UniqueID mainUID;
 const auto spelling=t.parserInputSpellings.find(currentPath);
 if(invalid||spelling==t.parserInputSpellings.end()||!entry||sm.isFileOverridden(entry)||
    sm.getFilename(sm.getLocForStartOfFile(mainFile))!=spelling->second||physicalPath(spelling->second)!=currentPath||
    llvm::sys::fs::getUniqueID(spelling->second,mainUID)||mainUID!=entry->getUniqueID()||
    mainUID!=t.identities.at(currentPath)||!read(currentPath,actual)||main!=actual){error="standalone main lacks the exact selected physical path/UID/bytes witness";return false;}
 if(rootParse&&t.bound){error="standalone root proof cannot be replayed with a different include context";return false;}
 const auto bind=[&](const std::string& path)->bool{if(rootParse)return t.remember(path);return t.inputs.count(physicalPath(path))&&t.remember(path);};
 if(!bind(currentPath)){error="unobserved/changed selected source";return false;}
 if(rootParse)t.callers.insert(currentPath);
 for(auto it=sm.fileinfo_begin();it!=sm.fileinfo_end();++it){const auto id=sm.translateFile(it->first);if(id.isInvalid()||id==sm.getMainFileID())continue;const auto loc=sm.getLocForStartOfFile(id);const auto path=physicalPath(sm.getFilename(loc).str());if(path.empty())continue;
  bool bad=false;const auto parsed=sm.getBufferData(id,&bad);std::string physical;llvm::sys::fs::UniqueID uid;
  const auto* entry=sm.getFileEntryForID(id);
  if(rootParse&&!native(loc))t.callers.insert(path);
  if(!entry||bad||sm.isFileOverridden(entry)||!read(path,physical)||parsed!=physical||llvm::sys::fs::getUniqueID(path,uid)||uid!=entry->getUniqueID()||physicalPath(sm.getFilename(loc).str())!=path||!bind(sm.getFilename(loc).str())){error="standalone input ledger is changed/remapped/unobserved: "+path;return false;}
   }
 if(rootParse){
  const auto cwd=ci.getFileManager().getVirtualFileSystem().getCurrentWorkingDirectory();
  if(!cwd){error="actual parser search working directory unavailable";return false;}
  const auto resource=physicalPath(ci.getHeaderSearchOpts().ResourceDir),sdk=physicalPath(CudaPath.getValue());
  for(const auto& entry:ci.getHeaderSearchOpts().UserEntries){
   if(entry.IsFramework){error="framework search is outside standalone target mapping";return false;}
   const fs::path raw(entry.Path);const auto spelling=(raw.is_absolute()?raw:fs::path(*cwd)/raw).string();const auto physical=physicalPath(spelling);
   if(physical==sdk||within(physical,sdk)||physical==resource||within(physical,resource)||physical==t.product||within(physical,t.product))continue;
   std::string flag;
   switch(entry.Group){case clang::frontend::Quoted:flag="-iquote";break;case clang::frontend::Angled:flag="-I";break;
    case clang::frontend::System:case clang::frontend::CXXSystem:case clang::frontend::ExternCSystem:flag="-isystem";break;
    case clang::frontend::After:flag="-idirafter";break;default:error="unmapped effective parser search group";return false;}
   if(!t.rememberDirectory(spelling)){error="changed/nonphysical retained search directory: "+spelling;return false;}
   t.searches.push_back({flag,spelling});
  }
  t.bound=true;
 }
 return t.matches();
 }
bool StandaloneDeviceSnapshot(const std::string& root,std::string& stamp,std::string& error){if(!target||!target->bound||target->root!=physicalPath(root)||!target->matches()){error="owned standalone source/provider snapshot missing or changed";return false;}stamp=target->stamp();return true;}
bool BuildStandaloneDeviceTarget(const std::string& stagedRoot,const std::string& stagedBundle,const std::vector<std::pair<std::string,std::string>>& translatedHeaders,std::string& error){
 if(!target||!target->bound||!target->edgesBound||target->consumed||!target->matches()){error="standalone capability is absent, changed or already consumed";return false;}
 auto& t=*target;std::error_code ec;const fs::path bundle(stagedBundle),directory=bundle/".ascify-device";fs::create_directories(directory,ec);if(ec){error="cannot create owned build directory";return false;}
 // The unpublished root keeps its final relative include spellings. A private
 // sibling alias resolves those bytes without rewriting the generated source.
 const auto alias=fs::path(stagedRoot).parent_path()/fs::path(t.bundle).filename();if(!fs::exists(alias,ec)){fs::create_directory_symlink(bundle,alias,ec);}if(ec||physicalPath(alias)!=physicalPath(bundle)||!t.rememberDirectory(alias)||!t.rememberDirectory(bundle)){error="cannot bind private staged include alias";return false;}
 std::map<std::string,std::string> generated;
 const auto rememberGenerated=[&](const fs::path& path){std::string bytes;if(!t.rememberGeneratedPhysical(path)||!read(path,bytes))return false;generated[path.string()]=hash(bytes);return true;};
 if(!rememberGenerated(stagedRoot)){error="missing generated standalone root";return false;}
 for(fs::recursive_directory_iterator it(bundle,ec),end;!ec&&it!=end;it.increment(ec)){if(it->is_regular_file(ec)&&it->path().extension()==".dpp"&&!rememberGenerated(it->path()))return false;}if(ec){error="cannot enumerate generated headers";return false;}
 t.activeStagedRoot=stagedRoot;t.activeBundle=physicalPath(bundle);
 const auto object=(directory/"target.o").string(),program=(directory/"target").string();
 std::vector<std::string> compile={t.compiler,"-x","dpp","--cce-aicore-arch=dav-c310-vec","-std=c++17","-O2","-c","-I"+fs::path(stagedRoot).parent_path().string(),"-I"+bundle.string(),"-I"+t.product+"/include"};
 for(const auto& search:t.searches){compile.push_back(search.first);compile.push_back(search.second);}
 for(const auto& suffix:{"/include","/asc/include","/asc","/aarch64-linux/asc","/aarch64-linux/asc/include","/include/ascendc/host_api","/compiler/ascendc/include/highlevel_api","/compiler/tikcpp/tikcfw","/compiler/tikcpp/tikcfw/lib","/compiler/tikcpp/tikcfw/lib/matmul","/compiler/tikcpp/tikcfw/impl","/compiler/tikcpp/tikcfw/interface",""})compile.push_back("-I"+t.cann+suffix);
 compile.insert(compile.end(),{stagedRoot,"-o",object});
 unsigned providers=0;
 if(!rememberDynamicProviders(t,directory,t.compiler,providers,error)||!rememberDriverPlan(t,directory,"compile-plan",compile,providers,error))return false;
 for(const auto& name:libraries)if(!rememberDynamicProviders(t,directory,t.cann+"/lib64/lib"+name+".so",providers,error))return false;
 auto dependencies=compile;dependencies.erase(std::find(dependencies.begin(),dependencies.end(),"-c"));dependencies.resize(dependencies.size()-2);
 dependencies.insert(dependencies.end(),{"-M","-MF",(directory/"dependencies-before.d").string(),"-MT","ascify-owned-target"});
 if(!execute(t,directory,"dependency-plan",dependencies,error))return false;
 std::set<std::string> beforeDependencies;
 if(!dependencyList(directory/"dependencies-before.d",beforeDependencies,error))return false;
 for(const auto& path:beforeDependencies){const auto physical=physicalPath(path);
  if(physical==physicalPath(stagedRoot)||within(physical,t.activeBundle)){
   if(!generated.count(physical)){error="target compiler selected an unplanned generated input: "+path;return false;}
  }else{
   const bool native=within(physical,t.cann)||within(physical,t.product)||within(physical,"/usr/include")||within(physical,"/usr/lib")||within(physical,"/lib");
   if(!native&&!t.inputs.count(physical)){error="retained caller header differs from actual parser selection: "+path;return false;}
   if(!t.remember(path)){error="target dependency changed during discovery: "+path;return false;}
  }
 }
 std::set<std::string> physicalDependencies;for(const auto& path:beforeDependencies)physicalDependencies.insert(physicalPath(path));
 std::map<std::string,std::string> mapped;mapped[t.root]=physicalPath(stagedRoot);
 for(const auto& item:translatedHeaders){const auto source=physicalPath(item.first),artifact=physicalPath(bundle/item.second);
  if(!t.callers.count(source)||!within(artifact,t.activeBundle)||!generated.count(artifact)||!mapped.emplace(source,artifact).second){error="translation mapping is not an owned physical input/artifact";return false;}
 }
 for(const auto& caller:t.callers){const auto found=mapped.find(caller);const auto& required=found==mapped.end()?caller:found->second;
  if(!physicalDependencies.count(required)){error="target omitted or changed a retained caller input: "+caller;return false;}
 }
 std::vector<std::set<std::string>> expectedViewDependencies;
 // Check both real preprocessor views independently. A file-set equality
 // cannot establish that each include still resolves in its original parent.
 for(const auto& view:std::vector<std::pair<std::string,std::string>>{{"device","--cce-aicore-only"},{"host","--cce-host-only"}}){
  auto query=dependencies;const auto dep=directory/("dependencies-"+view.first+".d");
  query[std::find(query.begin(),query.end(),"-MF")-query.begin()+1]=dep.string();
  query.insert(query.end(),{"-H","-fshow-skipped-includes",view.second});
  const auto label="include-edges-"+view.first;
  if(!execute(t,directory,label,query,error)||!verifyCallerIncludeTree(t,directory/(label+".stderr.log"),mapped,error))return false;
  std::set<std::string> paths;if(!dependencyList(dep,paths,error))return false;
  expectedViewDependencies.push_back(paths);
  for(const auto& path:paths){const auto physical=physicalPath(path);
   if(physical==physicalPath(stagedRoot)||within(physical,t.activeBundle)){if(!generated.count(physical)){error="unplanned target-view generated dependency";return false;}}
   else if(!t.remember(path)){error="target-view dependency changed";return false;}
  }
 }
 if(!t.matches()){error="target source/provider changed after dependency discovery";return false;}
 // The CANN driver runs device cc1 before host cc1. A shared -MF file is
 // overwritten by the host job; stdout retains both real dependency rules.
 compile.insert(compile.end(),{"-MD","-MF","-","-MT","ascify-owned-target"});
 if(!rememberDriverPlan(t,directory,"actual-compile-plan",compile,providers,error,true)||!t.matches()||!execute(t,directory,"compile",compile,error))return false;
 std::vector<std::set<std::string>> actualViews;
 if(!dependencyRules(directory/"compile.stdout.log",actualViews,error))return false;
 if(actualViews.size()!=2||actualViews!=expectedViewDependencies){error="actual device/host compiler dependencies differ from their sealed view plans";return false;}
 std::set<std::string> actualDependencies;for(const auto& paths:actualViews)actualDependencies.insert(paths.begin(),paths.end());
 if(actualDependencies!=beforeDependencies){error="actual compiler dependency union differs from sealed plan";return false;}
 if(!rememberGenerated(object)){error="owned compiler produced no object";return false;}
 // The fixed wrapper's actual native registration recipe selects these two
 // providers in addition to the explicit outer driver libraries.
 if(!t.remember(t.cann+"/lib64/libascendc_runtime.a")||!t.remember(t.cann+"/lib64/libunified_dlog.so")||!rememberDynamicProviders(t,directory,t.cann+"/lib64/libunified_dlog.so",providers,error)){if(error.empty())error="missing fixed native registration provider";return false;}
 const auto temporary=directory/"link-tmp";
 if(llvm::sys::fs::create_directory(temporary.string(),false,llvm::sys::fs::owner_all)){error="cannot create fresh private owned linker temporary directory";return false;}
 t.privateLinkTemporary=temporary.string();
 t.privateTemporaryPhysical=physicalPath(temporary);llvm::sys::fs::UniqueID temporaryUID;
 if(t.privateTemporaryPhysical.empty()||llvm::sys::fs::getUniqueID(temporary.string(),temporaryUID)){error="owned linker temporary has no physical identity";return false;}
 t.privateTemporaryIdentity=temporaryUID;
 if(!t.matchesPrivateTemporary()){error="owned linker temporary is not its fresh private physical directory";return false;}
 std::vector<std::string> link={t.compiler,"--cce-aicore-arch=dav-c310-vec","-O2",object,t.cann+"/lib64/libmsprofiler.so","-o",program,"-L"+t.cann+"/lib64","-L"+t.cann+"/aarch64-linux/lib64"};
 for(const auto& name:libraries)if(name!="msprofiler")link.push_back("-l"+name);
 link.insert(link.end(),{"-lstdc++","-lm","-pthread","-Wl,-rpath,"+t.cann+"/lib64","-Wl,-rpath,"+t.cann+"/aarch64-linux/lib64"});
 // Query the exact trace-enabled command that will actually link.
 link.insert(link.end(),{"-Wl,-t","-Wl,--reproduce,"+(directory/"link-inputs.tar").string()});
 if(!rememberDriverPlan(t,directory,"link-plan",link,providers,error))return false;
 // The trace records the actual selected linker files; no user object/archive
 // can enter through an argv, environment, response file or parent build.
 if(!t.matches()||!execute(t,directory,"link",link,error)||!rememberGenerated(program))return false;
 if(!verifyOwnedLinkReproduction(t,directory,object,generated,error))return false;
 for(const auto& input:generated){std::string bytes;if(!read(input.first,bytes)||hash(bytes)!=input.second){error="generated build input changed";return false;}}
 if(!t.matches()){error="source/provider inputs changed during actual target build";return false;}
 llvm::json::Object before,artifacts;for(const auto& p:t.inputs)before[p.first]=p.second;for(const auto& p:generated){
  const auto relative=p.first==stagedRoot?std::string("../")+fs::path(t.output).filename().string():fs::path(p.first).lexically_relative(bundle).string();artifacts[relative]=p.second;
 }
 llvm::json::Object receipt;receipt["schema"]="ascify.owned-standalone-device-target.v1";receipt["domain"]="new independent executable, not an arbitrary parent CMake closure";receipt["root_source"]=t.root;receipt["source_inputs"]=std::move(before);receipt["actual_artifacts"]=std::move(artifacts);llvm::json::Array searches;for(const auto& p:t.searches){llvm::json::Array row;row.push_back(p.first);row.push_back(p.second);searches.push_back(std::move(row));}receipt["effective_retained_search_order"]=std::move(searches);llvm::json::Object bindings,identities,mapping;
 for(const auto& p:t.bindings)bindings[p.first]=p.second;
 for(const auto& p:t.directoryBindings){bindings[p.first]=p.second;identities[p.first]=std::to_string(t.directoryIdentities.at(p.first).getDevice())+":"+std::to_string(t.directoryIdentities.at(p.first).getFile());}
 for(const auto& p:t.identities)identities[p.first]=std::to_string(p.second.getDevice())+":"+std::to_string(p.second.getFile());
 for(const auto& p:mapped)mapping[p.first]=p.second==physicalPath(stagedRoot)?std::string("../")+fs::path(t.output).filename().string():fs::path(p.second).lexically_relative(bundle).string();
 llvm::json::Array absentSearches;for(const auto& p:t.absentDirectories)absentSearches.push_back(p);receipt["search_directories_observed_ENOENT"]=std::move(absentSearches);
 receipt["physical_requested_bindings"]=std::move(bindings);receipt["physical_unique_ids"]=std::move(identities);receipt["caller_artifact_mapping"]=std::move(mapping);
 llvm::json::Array edges;
 for(const auto& edge:t.edges){llvm::json::Object item;item["parent"]=edge.parent;item["directive_offset"]=edge.offset;item["child"]=edge.child;item["literal_spelling"]=edge.spelling;item["angled"]=edge.angled;item["search_role"]=edge.search;item["parent_sha256"]=t.inputs.at(edge.parent);item["child_sha256"]=t.inputs.at(edge.child);edges.push_back(std::move(item));}
 receipt["original_caller_include_edges"]=std::move(edges);receipt["target_include_edge_views"]="independent host and dav-c310 device -H -fshow-skipped-includes";receipt["actual_dependency_rules"]="ordered device/host stdout rules, each exact against its preprocessor view; union exact against -M";
 llvm::json::Object parserCommands,driverMacros;
 for(const auto& command:t.parserArguments){llvm::json::Array argv;for(const auto& a:command.second)argv.push_back(a);parserCommands[command.first]=std::move(argv);}
 llvm::json::Object parserSources;for(const auto& source:t.parserInputSpellings)parserSources[source.first]=source.second;receipt["exact_selected_parser_input_spellings"]=std::move(parserSources);
 for(const auto& item:t.parserDriverMacros)driverMacros[item.first]=item.second;
 receipt["pre_driver_parser_argv"]=std::move(parserCommands);receipt["compiler_synthesized_macros"]=std::move(driverMacros);
 llvm::json::Object generatedTuples;
 for(const auto& item:t.generatedPhysical){llvm::json::Object row;row["canonical"]=item.second.canonical;
  row["uid"]=std::to_string(item.second.uid.getDevice())+":"+std::to_string(item.second.uid.getFile());
  row["size"]=item.second.size;row["sha256"]=item.second.digest;generatedTuples[item.first]=std::move(row);}
 receipt["actual_generated_physical_tuples"]=std::move(generatedTuples);
 receipt["actual_target_resource_dir"]=t.compilerResource;receipt["compile_units"]=1;receipt["owned_device_objects_linked"]=1;receipt["published_bundle"]=t.bundle;receipt["actual_argv_records_unpublished_paths"]=true;receipt["user_device_archives"]=0;receipt["optimization"]="O2";receipt["executed_device_program"]=false;
 {std::ofstream out(directory/"build.receipt.json");out<<llvm::formatv("{0:2}",llvm::json::Value(std::move(receipt))).str()<<"\n";if(!out){error="cannot seal actual standalone build";return false;}}
 for(fs::recursive_directory_iterator it(directory,ec),end;!ec&&it!=end;it.increment(ec)){
  if(it->is_symlink(ec)||!it->is_regular_file(ec)){error="unexpected nonregular owned build artifact";return false;}
  std::string bytes;if(!read(it->path(),bytes)){error="unreadable owned build artifact";return false;}
  t.sealedArtifacts[it->path().lexically_relative(bundle).string()]=hash(bytes);
 }
 if(ec||t.sealedArtifacts.empty()){error="cannot seal complete owned build artifacts";return false;}
 t.builtInputs=generated;
 if(!t.matches()){error="source/provider changed while sealing actual build artifacts";return false;}
 t.consumed=true;return true;
 }
bool GetStandaloneDeviceArtifacts(const std::string& bundle,std::vector<std::pair<std::string,std::string>>& artifacts,std::string& error){
 if(!target||!target->consumed||target->sealedArtifacts.empty()||!target->matches()){error="owned device build artifacts were not sealed";return false;}
 const auto root=physicalPath(bundle);std::set<std::string> actual;std::error_code ec;
 for(const auto& input:target->builtInputs){std::string bytes;const auto status=fs::symlink_status(input.first,ec);
  if(ec||fs::is_symlink(status)||!fs::is_regular_file(status)||!read(input.first,bytes)||hash(bytes)!=input.second){error="generated root/header/object/program changed after actual build";return false;}
 }
 for(fs::recursive_directory_iterator it(fs::path(bundle)/".ascify-device",ec),end;!ec&&it!=end;it.increment(ec)){
  if(it->is_symlink(ec)||!it->is_regular_file(ec)){error="nonregular owned device artifact";return false;}
  const auto relative=it->path().lexically_relative(bundle).string();std::string bytes;const auto expected=target->sealedArtifacts.find(relative);
  if(expected==target->sealedArtifacts.end()||!within(physicalPath(it->path()),root)||!read(it->path(),bytes)||hash(bytes)!=expected->second){error="changed/unrecorded owned device artifact";return false;}
  actual.insert(relative);
 }
 if(ec||actual.size()!=target->sealedArtifacts.size()){error="owned device artifact inventory differs";return false;}
 if(!target->matches()){error="source/provider changed while checking publication artifacts";return false;}
 artifacts.assign(target->sealedArtifacts.begin(),target->sealedArtifacts.end());return true;
}
} // namespace ascify
