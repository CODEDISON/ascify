#include "DeviceAggregateInlining.h"
#include "StandaloneDeviceTarget.h"
#include "RegisterTile32Lowering.h"
#include "CudaExecutionCoordinates.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/MacroArgs.h"
#include <cctype>
#include "llvm/Support/SHA256.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/ADT/SmallString.h"
#include "ArgParse.h"
#include <optional>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <tuple>
#include <regex>
#include <algorithm>
#include <memory>
namespace ascify {
bool DeviceRegisterTile32Fact::operator==(const DeviceRegisterTile32Fact& other) const {
 return std::tie(projectStamp,sourcePath,sourceSha256,beginOffset,endOffset)==
        std::tie(other.projectStamp,other.sourcePath,other.sourceSha256,other.beginOffset,other.endOffset);
}
bool DeviceAggregateInlineFact::operator==(const DeviceAggregateInlineFact& o) const {
 return std::tie(projectStamp,sourcePath,sourceSha256,recordName,layout,bodyGraph,signature,referenceSites,functionOffset,insertOffset,guardOffset,recordBytes,recordAlign,sharedBytes,callerOffset,calls)==
        std::tie(o.projectStamp,o.sourcePath,o.sourceSha256,o.recordName,o.layout,o.bodyGraph,o.signature,o.referenceSites,o.functionOffset,o.insertOffset,o.guardOffset,o.recordBytes,o.recordAlign,o.sharedBytes,o.callerOffset,o.calls);
}
void ObserveDeviceNativeMacro(clang::CompilerInstance& ci,const clang::Token& name,
    const clang::MacroDefinition& definition,clang::SourceRange invocation,
    const clang::MacroArgs* arguments,DeviceNativeMacroEvidence& evidence){
 const auto* macro=definition.getMacroInfo();
 if(!name.getIdentifierInfo()||!macro||(!macro->isBuiltinMacro()&&!macro->isFunctionLike()))return;
 DeviceNativeMacroUse use;use.definition=macro;use.name=name.getIdentifierInfo()->getName().str();
 use.nameLocation=name.getLocation();use.invocation=invocation;
 use.unambiguous=!definition.isAmbiguous()&&definition.getModuleMacros().empty();
 if(macro->isFunctionLike()&&arguments&&!macro->isVariadic()&&arguments->getNumMacroArguments()==macro->getNumParams()){
  const auto tokens=macro->tokens();
  for(unsigned i=0;i+1<tokens.size();++i){
   if(!tokens[i].is(clang::tok::hash)||!tokens[i+1].getIdentifierInfo())continue;
   const int parameter=macro->getParameterNum(tokens[i+1].getIdentifierInfo());if(parameter<0)continue;
   const auto* raw=arguments->getUnexpArgument(static_cast<unsigned>(parameter));
   if(!raw[0].is(clang::tok::identifier)||raw[0].needsCleaning()||!raw[1].is(clang::tok::eof))continue;
   use.stringArguments.push_back({tokens[i].getLocation(),tokens[i+1].getLocation(),
       raw[0].getLocation(),ci.getPreprocessor().getSpelling(raw[0])});
  }
 }
 evidence.uses.push_back(std::move(use));
}
namespace {
using namespace clang; namespace fs=std::filesystem;
std::string digest(llvm::StringRef bytes) {
 llvm::SHA256 h;h.update(bytes);const auto raw=h.final();std::string out;const char hex[]="0123456789abcdef";
 for(auto b:raw){out+=hex[(static_cast<unsigned char>(b)>>4)&15];out+=hex[static_cast<unsigned char>(b)&15];}return out;
}
bool read(const fs::path& p,std::string& out){std::ifstream f(p,std::ios::binary);if(!f)return false;out.assign(std::istreambuf_iterator<char>(f),{});return !f.bad();}
std::string canon(const fs::path& p){std::error_code ec;auto q=fs::canonical(p,ec);return ec?std::string():q.string();}
bool within(const std::string& file,const std::string& dir){return file.size()>dir.size()&&file.compare(0,dir.size(),dir)==0&&file[dir.size()]=='/';}
std::string lower(std::string s){for(char& c:s)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));return s;}
std::string trim(std::string s){const auto a=s.find_first_not_of(" \t\r\n");return a==std::string::npos?std::string():s.substr(a,s.find_last_not_of(" \t\r\n")-a+1);}
struct Project {std::string root,dir,stamp;};
bool project(const std::string& root,Project& p,std::string& error){
 p.root=canon(root);p.dir=fs::path(p.root).parent_path().string();
 return !p.root.empty()&&StandaloneDeviceSnapshot(root,p.stamp,error);
}
const Expr* plain(const Expr* e){for(;;){if(const auto* x=dyn_cast_or_null<ExprWithCleanups>(e)){if(x->cleanupsHaveSideEffects())return nullptr;e=x->getSubExpr();}else if(e){const auto* q=e->IgnoreParenImpCasts();if(q==e)return e;e=q;}else return nullptr;}}
// A SourceLocExpr in a native default argument is evaluated at its use,
// not at the default expression's native spelling. Follow both original and
// rewritten defaults explicitly: these AST nodes have no ordinary children.
bool containsTypedSourceObservation(const Stmt* statement,std::set<const Stmt*>& seen){
 if(!statement||!seen.insert(statement).second)return false;
 if(isa<SourceLocExpr>(statement)||isa<PredefinedExpr>(statement))return true;
 if(const auto* value=dyn_cast<CXXDefaultArgExpr>(statement)){
  if(containsTypedSourceObservation(value->getParam()->getDefaultArg(),seen)||
     containsTypedSourceObservation(value->getExpr(),seen))return true;
 }
 if(const auto* value=dyn_cast<CXXDefaultInitExpr>(statement)){
  if(containsTypedSourceObservation(value->getField()->getInClassInitializer(),seen)||
     containsTypedSourceObservation(value->getExpr(),seen))return true;
 }
 for(const auto* child:statement->children())if(containsTypedSourceObservation(child,seen))return true;
 return false;
}
class Inventory:public RecursiveASTVisitor<Inventory>{
 const FunctionDecl* owner=nullptr;
 public:
 std::set<const FunctionDecl*> functions;std::vector<const DeclRefExpr*> refs;std::vector<std::pair<const CallExpr*,const FunctionDecl*>> calls;
 std::set<const Expr*> direct;
 struct AssemblyUse {SourceLocation first;const FunctionDecl* second=nullptr;const Decl* declaration=nullptr;const GCCAsmStmt* gnu=nullptr;const char* kind="unknown";};
 std::vector<AssemblyUse> assembly;
 std::vector<SourceRange> sourceObservations;
 bool shouldVisitTemplateInstantiations()const{return true;}
 bool TraverseFunctionDecl(FunctionDecl* f){const auto* old=owner;owner=f;functions.insert(f->getCanonicalDecl());const bool r=RecursiveASTVisitor::TraverseFunctionDecl(f);owner=old;return r;}
 bool VisitDeclRefExpr(DeclRefExpr* r){if(isa<FunctionDecl>(r->getDecl()))refs.push_back(r);return true;}
 bool VisitCallExpr(CallExpr* c){calls.push_back({c,owner});if(c->getDirectCallee())direct.insert(plain(c->getCallee()));return true;}
 bool VisitDecl(Decl* d){
  if(d->hasAttr<AliasAttr>()||d->hasAttr<AsmLabelAttr>()||d->hasAttr<IFuncAttr>()||d->hasAttr<WeakAttr>()||d->hasAttr<WeakRefAttr>()||d->hasAttr<UsedAttr>()||d->hasAttr<RetainAttr>())assembly.push_back({d->getLocation(),dyn_cast<FunctionDecl>(d),d,nullptr,"declaration"});
  return true;
 }
 bool VisitCXXDefaultArgExpr(CXXDefaultArgExpr* e){
  std::set<const Stmt*> seen;
  if(containsTypedSourceObservation(e->getParam()->getDefaultArg(),seen)||containsTypedSourceObservation(e->getExpr(),seen))
   sourceObservations.push_back(SourceRange(e->getUsedLocation(),e->getUsedLocation()));
  return true;
 }
 bool VisitCXXDefaultInitExpr(CXXDefaultInitExpr* e){
  std::set<const Stmt*> seen;
  if(containsTypedSourceObservation(e->getField()->getInClassInitializer(),seen)||containsTypedSourceObservation(e->getExpr(),seen))
   sourceObservations.push_back(SourceRange(e->getUsedLocation(),e->getUsedLocation()));
  return true;
 }
 bool VisitSourceLocExpr(SourceLocExpr* e){sourceObservations.push_back(e->getSourceRange());return true;}
 bool VisitPredefinedExpr(PredefinedExpr* e){sourceObservations.push_back(e->getSourceRange());return true;}
 bool VisitGCCAsmStmt(GCCAsmStmt* s){assembly.push_back({s->getBeginLoc(),owner,nullptr,s,"GCCAsmStmt"});return true;}bool VisitMSAsmStmt(MSAsmStmt* s){assembly.push_back({s->getBeginLoc(),owner,nullptr,nullptr,"MSAsmStmt"});return true;}
 bool VisitFileScopeAsmDecl(FileScopeAsmDecl* d){assembly.push_back({d->getLocation(),nullptr,d,nullptr,"FileScopeAsmDecl"});return true;}
};
class Proof {
 CompilerInstance& ci;ASTContext& context;SourceManager& sm;const Project& p;const std::string current;
 const std::function<bool(SourceLocation)>& owned;const std::function<bool(const Decl*)>& sdk;const std::function<bool(SourceLocation)>& native;Inventory inventory;std::vector<std::string> graphParts;
 const std::vector<DeviceAggregateInlineFact>* witnesses;
 const DeviceNativeMacroEvidence& macroEvidence;
 std::set<const FunctionDecl*> helperDeclarations;
 std::set<std::string> helperSymbols;
 bool symbolsDisjoint(llvm::StringRef text)const {
  const auto identifier=[](char c){return std::isalnum(static_cast<unsigned char>(c))||c=='_'||c=='$';};
  for(const auto& symbol:helperSymbols){std::size_t at=0;while((at=text.find(symbol,at))!=llvm::StringRef::npos){
   const auto end=at+symbol.size();if((at==0||!identifier(text[at-1]))&&(end==text.size()||!identifier(text[end])))return false;at=end;
  }}return true;
 }
 bool fixedNative(SourceLocation location)const {
  return location.isValid()&&native(sm.getSpellingLoc(location))&&native(sm.getExpansionLoc(location));
 }
 bool nativeMacroUse(const DeviceNativeMacroUse& use)const {
  const auto* macro=use.definition;
  if(!macro||!use.unambiguous||macro->isBuiltinMacro()||!macro->isFunctionLike()||macro->isVariadic()||
      !fixedNative(use.nameLocation)||!fixedNative(use.invocation.getBegin())||!fixedNative(use.invocation.getEnd())||
      !macro->getDefinitionLoc().isFileID()||!macro->getDefinitionEndLoc().isFileID()||
      !native(macro->getDefinitionLoc())||!native(macro->getDefinitionEndLoc()))return false;
  Token name;if(Lexer::getRawToken(macro->getDefinitionLoc(),name,sm,ci.getLangOpts(),true)||
      !name.is(tok::raw_identifier)||name.getRawIdentifier()!=use.name)return false;
  for(const auto& token:macro->tokens())if(!token.getLocation().isFileID()||!native(token.getLocation()))return false;
  return true;
 }
 bool literalBenignPragma(SourceLocation location)const {
  if(!location.isFileID())return false;bool bad=false;
  const auto bytes=sm.getBufferData(sm.getFileID(location),&bad);if(bad)return false;
  const auto offset=sm.getFileOffset(location);if(offset>=bytes.size())return false;
  auto first=offset;while(first&&bytes[first-1]!='\n'&&bytes[first-1]!='\r')--first;
  auto last=offset;while(last<bytes.size()&&bytes[last]!='\n'&&bytes[last]!='\r')++last;
  const auto line=bytes.slice(first,last);
  if(line.contains('\\')||line.contains("??/")||line.contains("/*"))return false;
  return std::regex_match(line.str(),std::regex(R"re(^[ \t]*#[ \t]*pragma[ \t]+(once|unroll([ \t]+[0-9]+)?)[ \t]*(//[^\r\n]*)?$)re"));
 }
 bool compilerCacheLinePragma(SourceLocation location,const std::set<std::string>& usedNames)const {
  auto& pp=ci.getPreprocessor();const auto file=pp.getPredefinesFileID();
  if(!location.isValid()||!location.isFileID()||file.isInvalid()||sm.getFileID(location)!=file||sm.getFileEntryForID(file))return false;
  bool bad=false;const auto buffer=sm.getBufferData(file,&bad);
  if(bad||buffer!=pp.getPredefines())return false;
  const auto offset=sm.getFileOffset(location);if(offset>=buffer.size())return false;
  auto begin=offset;while(begin&&buffer[begin-1]!='\n'&&buffer[begin-1]!='\r')--begin;
  auto end=offset;while(end<buffer.size()&&buffer[end]!='\n'&&buffer[end]!='\r')++end;
  const auto line=buffer.slice(begin,end);
  for(const auto* name:{"__GCC_DESTRUCTIVE_SIZE","__GCC_CONSTRUCTIVE_SIZE"}){
   if(usedNames.count(name))continue;
   for(const auto* operation:{"push_macro","pop_macro"}){
    const std::string expected=std::string("#pragma ")+operation+"(\""+name+"\")";
    if(line!=expected)continue;
    if(Verbose){llvm::json::Object proof;proof["role"]="owned compiler predefines cacheline stack event";proof["predefines_buffer_sha256"]=digest(buffer);proof["whole_buffer_matches_Preprocessor"]=true;proof["actual_predefines_FileID"]=file.getHashValue();proof["offset"]=offset;proof["raw_line"]=line.str();proof["macro"]=name;proof["operation"]=operation;proof["participating_names_disjoint"]=true;llvm::errs()<<"Ascify compiler cacheline pragma proof: "<<llvm::formatv("{0}",llvm::json::Value(std::move(proof)))<<"\n";}
    return true;
   }
  }
  return false;
 }
 bool nativeStringification(SourceLocation token)const {
  const auto reject=[&](const char* reason){if(Verbose)llvm::errs()<<"Ascify native stringification witness rejected: "<<reason<<"\n";return false;};
  if(!token.isMacroID())return reject("scratch token lacks an expansion chain");
  std::set<std::string> usedNames;auto node=token;unsigned depth=0;
  while(node.isMacroID()&&sm.isMacroArgExpansion(node)){
   if(++depth>32)return reject("macro argument chain exceeds the bounded proof");
   const auto range=sm.getImmediateExpansionRange(node);const auto begin=sm.getSpellingLoc(range.getBegin()),end=sm.getSpellingLoc(range.getEnd());
   if(begin!=end||!begin.isFileID()||!native(begin))return reject("argument carrier is not one physical native parameter token");
   const DeviceNativeMacroUse* carrier=nullptr;
   for(const auto& use:macroEvidence.uses){
    if(sm.getExpansionLoc(use.invocation.getBegin())!=sm.getExpansionLoc(token))continue;
    for(const auto& replacement:use.definition->tokens()){
     if(replacement.getLocation()!=begin||!replacement.getIdentifierInfo()||use.definition->getParameterNum(replacement.getIdentifierInfo())<0)continue;
     if(!nativeMacroUse(use)||(carrier&&carrier->definition!=use.definition))return reject("carrier MacroInfo has no unique physical native definition");carrier=&use;
    }
   }
   if(!carrier)return reject("argument carrier has no observed active MacroInfo");usedNames.insert(carrier->name);
   node=sm.getImmediateSpellingLoc(node);
  }
  if(!node.isMacroID())return reject("argument chain does not reach stringization");
  const auto scratch=sm.getImmediateSpellingLoc(node);
  if(!scratch.isFileID()||!sm.isWrittenInScratchSpace(scratch))return reject("stringization does not have a scratch spelling");
  const auto operation=sm.getImmediateExpansionRange(node);
  const auto hash=sm.getSpellingLoc(operation.getBegin()),parameter=sm.getSpellingLoc(operation.getEnd());
  if(!hash.isFileID()||!parameter.isFileID()||!native(hash)||!native(parameter))return reject("stringization operator and parameter are not physical native tokens");
  const auto outer=sm.getImmediateMacroCallerLoc(node);
  if(!outer.isMacroID()||sm.isMacroArgExpansion(outer)||sm.getImmediateSpellingLoc(outer)!=hash)return reject("stringization has no direct defining macro frame");
  const auto invocation=sm.getImmediateExpansionRange(outer);
  if(!invocation.getBegin().isFileID()||!invocation.getEnd().isFileID()||!native(invocation.getBegin())||!native(invocation.getEnd()))return reject("defining invocation is not a direct physical native call");
  const DeviceNativeMacroUse* matched=nullptr;const DeviceNativeStringArgument* argument=nullptr;
  for(const auto& use:macroEvidence.uses){
   if(use.invocation!=invocation.getAsRange()||use.nameLocation!=invocation.getBegin())continue;
   for(const auto& candidate:use.stringArguments){
    if(candidate.hash!=hash||candidate.parameter!=parameter)continue;
    if(matched||!nativeMacroUse(use))return reject("stringization does not have one observed active native MacroInfo");matched=&use;argument=&candidate;
   }
  }
  if(!matched||!argument)return reject("no observed native #parameter/unexpanded argument witness");
  if(!argument->argument.isFileID()||!native(argument->argument))return reject("stringized argument is not an original native token");
  Token raw;if(Lexer::getRawToken(argument->argument,raw,sm,ci.getLangOpts(),true)||!raw.is(tok::raw_identifier)||raw.needsCleaning()||raw.getRawIdentifier()!=argument->spelling)return reject("original argument spelling is not one literal identifier");
  Token generated;if(Lexer::getRawToken(scratch,generated,sm,ci.getLangOpts(),true)||!generated.is(tok::string_literal)||ci.getPreprocessor().getSpelling(generated)!="\""+argument->spelling+"\"")return reject("scratch token differs from exact single-identifier stringization");
  usedNames.insert(matched->name);
  for(const auto& control:macroEvidence.controls)if(usedNames.count(control.first)&&!fixedNative(control.second))return reject("participating macro has caller define/undef history");
  for(const auto location:macroEvidence.pragmas)if(!fixedNative(location)&&!literalBenignPragma(location)&&!compilerCacheLinePragma(location,usedNames)){
   if(Verbose){llvm::json::Object diagnostic;diagnostic["location"]=location.printToString(sm);diagnostic["macro_location"]=location.isMacroID();
    const auto add=[&](const char* role,SourceLocation at){llvm::json::Object item;item["location"]=at.printToString(sm);item["file"]=sm.getFilename(at).str();item["fixed_native"]=at.isValid()&&at.isFileID()&&native(at);bool bad=false;
     if(at.isValid()){const auto buffer=sm.getBufferData(sm.getFileID(at),&bad);const auto offset=sm.getFileOffset(at);if(!bad&&offset<=buffer.size()){auto begin=offset;while(begin&&buffer[begin-1]!='\n'&&buffer[begin-1]!='\r')--begin;auto end=offset;while(end<buffer.size()&&buffer[end]!='\n'&&buffer[end]!='\r')++end;item["physical_line"]=buffer.slice(begin,std::min<std::size_t>(end,begin+1024)).str();item["offset"]=offset;}}
     diagnostic[role]=std::move(item);
    };add("spelling",sm.getSpellingLoc(location));add("expansion",sm.getExpansionLoc(location));llvm::errs()<<"Ascify native stringification pragma evidence: "<<llvm::formatv("{0}",llvm::json::Value(std::move(diagnostic)))<<"\n";
   }return reject("macro stack history contains an unproved pragma");
  }
  if(Verbose){
   const auto at=matched->definition->getDefinitionLoc();const auto file=sm.getFileID(at);const auto* entry=sm.getFileEntryForID(file);if(!entry)return false;
   std::string body;for(const auto& part:matched->definition->tokens())body+=ci.getPreprocessor().getSpelling(part)+"\n";
   llvm::json::Object proof;proof["macro"]=matched->name;proof["definition_file"]=sm.getFilename(at).str();proof["definition_offset"]=sm.getFileOffset(at);proof["definition_file_sha256"]=digest(sm.getBufferData(file));proof["definition_uid"]=std::to_string(entry->getUniqueID().getDevice())+":"+std::to_string(entry->getUniqueID().getFile());proof["replacement_tokens_sha256"]=digest(body);proof["hash_offset"]=sm.getFileOffset(hash);proof["parameter_offset"]=sm.getFileOffset(parameter);proof["argument_file"]=sm.getFilename(argument->argument).str();proof["argument_offset"]=sm.getFileOffset(argument->argument);proof["argument"]=argument->spelling;proof["invocation_begin"]=sm.getFileOffset(invocation.getBegin());proof["invocation_end"]=sm.getFileOffset(invocation.getEnd());llvm::json::Array names;for(const auto& name:usedNames)names.push_back(name);proof["history_macro_names"]=std::move(names);proof["whole_callback_history_native_or_literal_benign_pragmas"]=true;llvm::errs()<<"Ascify physical native stringification proof: "<<llvm::formatv("{0}",llvm::json::Value(std::move(proof)))<<"\n";
  }
  return true;
 }
 bool nativeAssemblyDisjoint(const Inventory::AssemblyUse& use)const {
  if(!fixedNative(use.first))return false;
  if(use.gnu){
   const auto* text=dyn_cast<StringLiteral>(use.gnu->getAsmStringExpr()->IgnoreParenImpCasts());
   if(!text||use.gnu->isAsmGoto()||!symbolsDisjoint(text->getString()))return false;
   bool nativeString=fixedNative(text->getBeginLoc())&&fixedNative(text->getEndLoc())&&text->getNumConcatenated();
   llvm::json::Array stringTokens;
   for(unsigned i=0;i<text->getNumConcatenated();++i){const auto at=text->getStrTokenLoc(i),spelling=sm.getSpellingLoc(at),expansion=sm.getExpansionLoc(at);
    const bool scratch=spelling.isValid()&&sm.isWrittenInScratchSpace(spelling),tokenNative=fixedNative(at)&&!scratch,stringized=scratch&&nativeStringification(at);nativeString&=tokenNative||stringized;
    if(Verbose){llvm::json::Object token;token["spelling_file"]=sm.getFilename(spelling).str();token["spelling_offset"]=spelling.isValid()?sm.getFileOffset(spelling):0;token["expansion_file"]=sm.getFilename(expansion).str();token["expansion_offset"]=expansion.isValid()?sm.getFileOffset(expansion):0;token["fixed_native"]=tokenNative;token["certified_stringification"]=stringized;token["scratch"]=scratch;
     if(!tokenNative){llvm::json::Array frames;auto node=at;unsigned depth=0;
      while(node.isMacroID()&&depth++<32){const auto range=sm.getImmediateExpansionRange(node);const auto spell=sm.getImmediateSpellingLoc(node),caller=sm.getImmediateMacroCallerLoc(node);llvm::json::Object frame;
       frame["name"]=Lexer::getImmediateMacroName(node,sm,ci.getLangOpts()).str();frame["macro_argument"]=sm.isMacroArgExpansion(node);
       frame["immediate_spelling"]=spell.printToString(sm);frame["expansion_begin"]=range.getBegin().printToString(sm);frame["expansion_end"]=range.getEnd().printToString(sm);frame["caller"]=caller.printToString(sm);frames.push_back(std::move(frame));if(caller==node)break;node=caller;
      }token["macro_frames"]=std::move(frames);token["frame_diagnostic_truncated"]=node.isMacroID();
     }stringTokens.push_back(std::move(token));}
   }
   if(Verbose){llvm::json::Object evidence;evidence["asm"]=text->getString().str();evidence["tokens"]=std::move(stringTokens);evidence["literal_begin_native"]=fixedNative(text->getBeginLoc());evidence["literal_end_native"]=fixedNative(text->getEndLoc());evidence["all_native"]=nativeString;llvm::errs()<<"Ascify fixed-native assembly string tokens: "<<llvm::formatv("{0}",llvm::json::Value(std::move(evidence)))<<"\n";}
   if(!nativeString)return false;
   class Operands:public RecursiveASTVisitor<Operands>{
    const std::set<const FunctionDecl*>& helpers;
    public:bool safe=true;explicit Operands(const std::set<const FunctionDecl*>& h):helpers(h){}
    bool VisitDeclRefExpr(DeclRefExpr* reference){if(const auto* f=dyn_cast<FunctionDecl>(reference->getDecl()))if(helpers.count(f->getCanonicalDecl()))safe=false;return true;}
    bool VisitMemberExpr(MemberExpr* member){if(const auto* f=dyn_cast<FunctionDecl>(member->getMemberDecl()))if(helpers.count(f->getCanonicalDecl()))safe=false;return true;}
   } operands(helperDeclarations);
   for(unsigned i=0;i<use.gnu->getNumOutputs();++i)operands.TraverseStmt(const_cast<Expr*>(use.gnu->getOutputExpr(i)));
   for(unsigned i=0;i<use.gnu->getNumInputs();++i)operands.TraverseStmt(const_cast<Expr*>(use.gnu->getInputExpr(i)));
   return operands.safe;
  }
  const auto* declaration=use.declaration;
  if(!declaration||declaration->isInvalidDecl()||std::string(use.kind)!="declaration"||declaration->hasAttr<AliasAttr>()||declaration->hasAttr<IFuncAttr>()||declaration->hasAttr<WeakRefAttr>())return false;
  const auto* function=dyn_cast<FunctionDecl>(declaration);const auto* variable=dyn_cast<VarDecl>(declaration);
  if(!function&&!variable)return false;
  if(declaration->getDeclContext()->isDependentContext())return false;
  if(function){
   if(isa<CXXConstructorDecl>(function)||isa<CXXDestructorDecl>(function))return false;
   if(function->getTemplatedKind()!=FunctionDecl::TK_NonTemplate||function->getType()->isDependentType()||helperDeclarations.count(function->getCanonicalDecl()))return false;
   for(const auto* parameter:function->parameters())if(parameter->getType()->isDependentType())return false;
  }
  if(variable&&(variable->getType()->isDependentType()||variable->getDescribedVarTemplate()||isa<VarTemplateSpecializationDecl>(variable)))return false;
  if(const auto* label=declaration->getAttr<AsmLabelAttr>())if(!symbolsDisjoint(label->getLabel()))return false;
  std::unique_ptr<MangleContext> mangler(context.createMangleContext());std::string symbol;llvm::raw_string_ostream out(symbol);
  const auto* named=cast<NamedDecl>(declaration);
  if(mangler->shouldMangleDeclName(named))mangler->mangleName(function?GlobalDecl(function):GlobalDecl(variable),out);else out<<named->getName();out.flush();
  return !symbol.empty()&&symbolsDisjoint(symbol);
 }
 std::string file(SourceLocation loc)const{if(!loc.isFileID())return {};return sm.isWrittenInMainFile(loc)?current:canon(sm.getFilename(loc).str());}
 bool literal(SourceLocation loc)const{return loc.isFileID()&&owned(loc)&&within(file(loc),p.dir);}
 bool vector(QualType type,std::string& tag){
  const auto* record=type->getAsCXXRecordDecl();if(!record||!sdk(record)||record->getQualifiedNameAsString()!="uint4"||!record->getDefinition())return false;
  record=record->getDefinition();if(record->getNumBases()||record->isUnion()||std::distance(record->field_begin(),record->field_end())!=4)return false;
  const auto& layout=context.getASTRecordLayout(record);if(layout.getSize().getQuantity()!=16||layout.getAlignment().getQuantity()!=16)return false;
  unsigned i=0;const char* fields[]={"x","y","z","w"};for(const auto* f:record->fields()){if(f->getName()!=fields[i]||!context.hasSameType(f->getType(),context.UnsignedIntTy)||layout.getFieldOffset(i)!=32*i)return false;++i;}
  tag="uint4:16:16";return true;
 }
 bool record(const CXXRecordDecl* r,DeviceAggregateInlineFact& fact){
  r=r?r->getDefinition():nullptr;if(!r||!literal(r->getLocation())||!r->getDeclContext()->isTranslationUnit()||!r->getIdentifier()||r->isUnion()||r->hasAttrs()||r->getNumBases()||r->isPolymorphic()||!r->isStandardLayout()||!r->isTriviallyCopyable()||!r->hasTrivialDestructor()||!r->hasTrivialDefaultConstructor())return false;
  const auto& l=context.getASTRecordLayout(r);fact.recordBytes=l.getSize().getQuantity();fact.recordAlign=l.getAlignment().getQuantity();fact.recordName=r->getNameAsString();fact.layout=std::to_string(fact.recordBytes)+":"+std::to_string(fact.recordAlign);
  if(fact.recordBytes!=64||fact.recordAlign!=16||std::distance(r->field_begin(),r->field_end())!=4||!r->getBeginLoc().isFileID()||file(r->getLocation())!=fact.sourcePath)return false;
  unsigned index=0;for(const auto* f:r->fields()){
   if(f->hasAttrs()||f->isBitField()||f->hasInClassInitializer()||!literal(f->getLocation())||f->getType().isVolatileQualified())return false;
   std::string tag;if(!vector(f->getType(),tag)||l.getFieldOffset(index)!=128*index)return false;
   fact.layout+=";"+tag+":"+std::to_string(l.getFieldOffset(index++));
  }
  auto end=Lexer::getLocForEndOfToken(r->getEndLoc(),0,sm,ci.getLangOpts());if(!literal(end))return false;
  bool invalid=false;const auto data=sm.getBufferData(sm.getFileID(end),&invalid);unsigned offset=sm.getFileOffset(end);while(!invalid&&offset<data.size()&&std::isspace(static_cast<unsigned char>(data[offset])))++offset;
  if(invalid||offset>=data.size()||data[offset]!=';')return false;fact.guardOffset=offset+1;return true;
 }
  bool compilerImplicitBlockBarrier(const CallExpr* call) const {
   const auto* callee=call?call->getDirectCallee():nullptr;
   if(!ci.getLangOpts().CUDA||!callee||!callee->isImplicit()||
      call->getNumArgs()!=0||!call->getType()->isVoidType())return false;
   const unsigned raw=callee->getBuiltinID();if(!raw)return false;
   auto& builtins=context.BuiltinInfo;
   const auto* target=&context.getTargetInfo();unsigned id=raw;
   // Host CUDA registers NVPTX builtins in the auxiliary-ID interval. Compare
   // the normalized target enum, never the observed host-specific raw number.
   if(builtins.isAuxBuiltinID(raw)){
    target=context.getAuxTargetInfo();if(!target)return false;
    id=builtins.getAuxBuiltinID(raw);
   }
   const auto arch=target->getTriple().getArch();
   if((arch!=llvm::Triple::nvptx&&arch!=llvm::Triple::nvptx64)||
      id!=clang::NVPTX::BI__syncthreads)return false;
   for(const auto* declaration:callee->redecls()){
    const auto* prototype=declaration->getType()->getAs<FunctionProtoType>();
    if(!declaration->isImplicit()||declaration->getBuiltinID()!=raw||
       declaration->doesThisDeclarationHaveABody()||declaration->getDefinition()||
       declaration->isDeleted()||declaration->isDefaulted()||isa<CXXMethodDecl>(declaration)||
       declaration->getTemplatedKind()!=FunctionDecl::TK_NonTemplate||
       !prototype||prototype->isVariadic()||prototype->getNumParams()!=0||
       declaration->getNumParams()!=0||!prototype->getReturnType()->isVoidType()||
       !declaration->hasAttr<BuiltinAttr>()||!declaration->hasAttr<CUDADeviceAttr>())return false;
    for(const auto* attribute:declaration->attrs())
     if(!isa<BuiltinAttr>(attribute)&&!isa<CUDADeviceAttr>(attribute))return false;
   }
   const auto* reference=dyn_cast_or_null<DeclRefExpr>(plain(call->getCallee()));
   if(!reference||reference->hasQualifier()||reference->hasExplicitTemplateArgs()||
      reference->getDecl()->getCanonicalDecl()!=callee->getCanonicalDecl()||
      !literal(reference->getLocation())||!literal(call->getBeginLoc())||
      !literal(call->getEndLoc())||call->getBeginLoc()!=reference->getLocation()||
      sm.getFileID(call->getBeginLoc())!=sm.getFileID(call->getEndLoc()))return false;
   bool invalid=false;
   const auto text=Lexer::getSourceText(CharSourceRange::getTokenRange(call->getSourceRange()),sm,ci.getLangOpts(),&invalid);
   if(invalid||text.contains('\\'))return false;
   // The raw lexer reads its end sentinel; a SourceManager substring is
   // not NUL-terminated at the call boundary. Keep an owned terminated copy.
   const std::string tokenBuffer=text.str();
   Lexer rawTokens(call->getBeginLoc(),ci.getLangOpts(),tokenBuffer.data(),tokenBuffer.data(),tokenBuffer.data()+tokenBuffer.size());Token token;
   rawTokens.LexFromRawLexer(token);
   if(!token.is(tok::raw_identifier)||token.needsCleaning()||token.getRawIdentifier()!="__syncthreads")return false;
   rawTokens.LexFromRawLexer(token);if(!token.is(tok::l_paren)||token.needsCleaning())return false;
   rawTokens.LexFromRawLexer(token);if(!token.is(tok::r_paren)||token.needsCleaning())return false;
   rawTokens.LexFromRawLexer(token);return token.is(tok::eof);
  }
  bool noRecursion(const FunctionDecl* f,std::set<const FunctionDecl*>& active,std::set<const FunctionDecl*>& done,bool& usesCompilerBlockBarrier){
  f=f->getCanonicalDecl();if(done.count(f))return true;if(!active.insert(f).second)return false;
  for(const auto& site:inventory.calls)if(site.second&&site.second->getCanonicalDecl()==f){const auto* callee=site.first->getDirectCallee();if(!callee)return false;if(sdk(callee))continue;
   if(compilerImplicitBlockBarrier(site.first)){usesCompilerBlockBarrier=true;continue;}
    if(callee->getBuiltinID()){const auto name=callee->getName();if(name=="memcpy"||name=="__builtin_memcpy")continue;return false;}const auto* def=callee->getDefinition();if(!def||!literal(def->getLocation())||!noRecursion(callee,active,done,usesCompilerBlockBarrier))return false;}
  const auto* definition=f->getDefinition();if(!definition||!definition->getBody())return false;
  class Observations:public RecursiveASTVisitor<Observations>{public:bool safe=true;
   bool VisitPredefinedExpr(PredefinedExpr*){safe=false;return true;}
   bool VisitAddrLabelExpr(AddrLabelExpr*){safe=false;return true;}
   bool VisitVarDecl(VarDecl* v){if(v->isLocalVarDecl()&&!v->hasAttr<CUDASharedAttr>()&&(v->isStaticLocal()||v->getTLSKind()!=VarDecl::TLS_None))safe=false;return true;}
  } observations;observations.TraverseStmt(definition->getBody());if(!observations.safe)return false;
  std::string printed;llvm::raw_string_ostream out(printed);definition->getBody()->printPretty(out,nullptr,context.getPrintingPolicy());out.flush();
  graphParts.push_back(file(definition->getLocation())+":"+std::to_string(sm.getFileOffset(definition->getLocation()))+":"+digest(printed));
  active.erase(f);done.insert(f);return true;
 }
 bool implicitSharedStorage(const VarDecl* variable) const {
  if(variable->getStorageClass()==SC_None)return true; // Preserve the old path.
  if(variable->getStorageClass()!=SC_Static)return false;
  const auto* attribute=variable->getAttr<CUDASharedAttr>();if(!attribute)return false;
  const auto begin=sm.getExpansionLoc(variable->getBeginLoc());
  const auto end=sm.getExpansionLoc(variable->getEndLoc());
  if(!literal(begin)||!literal(end)||sm.getFileID(begin)!=sm.getFileID(end)||
     begin!=sm.getExpansionLoc(attribute->getLocation()))return false;
  bool invalid=false;
  const auto text=Lexer::getSourceText(CharSourceRange::getTokenRange(begin,end),sm,ci.getLangOpts(),&invalid);
  if(invalid||text.contains('\\'))return false;
  const std::string buffer=text.str();
  Lexer raw(begin,ci.getLangOpts(),buffer.data(),buffer.data(),buffer.data()+buffer.size());
  Token tokens[7];
  for(auto& token:tokens){raw.LexFromRawLexer(token);if(token.needsCleaning())return false;}
  if(!tokens[0].is(tok::raw_identifier)||tokens[0].getRawIdentifier()!="__shared__"||
     !tokens[1].is(tok::raw_identifier)||tokens[1].getRawIdentifier()!="unsigned"||
     !tokens[2].is(tok::raw_identifier)||tokens[2].getRawIdentifier()!="int"||
     !tokens[3].is(tok::raw_identifier)||tokens[3].getRawIdentifier()!=variable->getName()||
     !tokens[4].is(tok::l_square)||!tokens[5].is(tok::numeric_constant)||!tokens[6].is(tok::r_square))return false;
  Token tail;raw.LexFromRawLexer(tail);if(!tail.is(tok::eof))return false;
  // A caller must not inject a written storage class through a replacement
  // for __shared__ or a nested qualifier macro. These are data/provenance
  // checks only; no macro file obtains SDK declaration authority here.
  auto& preprocessor=ci.getPreprocessor();std::vector<std::string> pending{"__shared__"};
  std::set<std::string> used;
  while(!pending.empty()){
   const auto name=pending.back();pending.pop_back();if(!used.insert(name).second)continue;
   if(used.size()>32)return false;
   const auto definition=preprocessor.getMacroDefinitionAtLoc(preprocessor.getIdentifierInfo(name),begin);
   if(!definition){if(name=="__shared__")return false;continue;}
   const auto* macro=definition.getMacroInfo();
   if(!macro||definition.isAmbiguous()||!definition.getModuleMacros().empty()||macro->isBuiltinMacro()||macro->isVariadic()||
      !fixedNative(macro->getDefinitionLoc())||!fixedNative(macro->getDefinitionEndLoc()))return false;
   for(const auto& token:macro->tokens()){
    if(!token.getLocation().isFileID()||!native(token.getLocation()))return false;
    const auto word=preprocessor.getSpelling(token);if(word=="static"||word=="extern")return false;
    if(const auto* identifier=token.getIdentifierInfo())
     if(!macro->isFunctionLike()||macro->getParameterNum(identifier)<0)pending.push_back(identifier->getName().str());
   }
  }
  for(const auto& event:macroEvidence.controls)
   if(used.count(event.first)&&!fixedNative(event.second))return false;
  return true;
 }
 bool shared(const FunctionDecl* f,DeviceAggregateInlineFact& fact){
  class Shared:public RecursiveASTVisitor<Shared>{public:std::vector<const VarDecl*> vars;std::vector<const DeclRefExpr*> refs;bool VisitVarDecl(VarDecl*v){if(v->hasAttr<CUDASharedAttr>())vars.push_back(v);return true;}bool VisitDeclRefExpr(DeclRefExpr*r){refs.push_back(r);return true;}} v;v.TraverseStmt(f->getBody());
  if(v.vars.empty())return true;if(v.vars.size()!=1||fact.calls!=1)return false;const auto* var=v.vars.front();const auto* array=context.getAsConstantArrayType(var->getType());
  if(!literal(var->getLocation())||!implicitSharedStorage(var)||var->hasInit()||!array||!context.hasSameType(array->getElementType(),context.UnsignedIntTy)||array->getSize().ugt(4096))return false;
  for(const auto* a:var->attrs())if(!isa<CUDASharedAttr>(a))return false;fact.sharedBytes=static_cast<unsigned>(array->getSize().getZExtValue()*4);if(!fact.sharedBytes)return false;
  for(const auto* r:v.refs)if(r->getDecl()==var){const auto parents=context.getParents(*r);const auto* cast=parents.size()==1?parents[0].get<ImplicitCastExpr>():nullptr;if(!cast||cast->getCastKind()!=CK_ArrayToPointerDecay)return false;const auto up=context.getParents(*cast);const auto* sub=up.size()==1?up[0].get<ArraySubscriptExpr>():nullptr;if(!sub||sub->getBase()!=cast)return false;
   const auto uses=context.getParents(*sub);if(uses.size()!=1)return false;
   const auto* read=uses[0].get<ImplicitCastExpr>();const auto* write=uses[0].get<BinaryOperator>();
   if(!(read&&read->getCastKind()==CK_LValueToRValue)&&!(write&&write->isAssignmentOp()&&write->getLHS()==sub))return false;
  }
  for(const auto& site:inventory.calls)if(site.first->getDirectCallee()&&site.first->getDirectCallee()->getCanonicalDecl()==f->getCanonicalDecl()){
   const auto* caller=site.second;if(!caller||!caller->hasAttr<CUDAGlobalAttr>()||caller->getTemplatedKind()!=FunctionDecl::TK_NonTemplate||!literal(caller->getLocation()))return false;
   const auto parents=context.getParents(*site.first);const auto* variable=parents.size()==1?parents[0].get<VarDecl>():nullptr;if(!variable||plain(variable->getInit())!=site.first)return false;
   const auto up=context.getParents(*variable);const auto* decl=up.size()==1?up[0].get<DeclStmt>():nullptr;if(!decl||!decl->isSingleDecl())return false;const auto top=context.getParents(*decl);if(top.size()!=1||top[0].get<CompoundStmt>()!=caller->getBody())return false;
   // A top-level call can still be skipped by an earlier divergent return or
   // repeated through a control transfer. Only straight declaration setup
   // with authenticated non-terminating SDK/builtin calls precedes it.
   const auto* body=cast<CompoundStmt>(caller->getBody());
   class Setup:public RecursiveASTVisitor<Setup>{
    const std::function<bool(const Decl*)>& sdk;ASTContext& context;
    const std::function<bool(SourceLocation)>& native;
    public:bool safe=true;explicit Setup(const std::function<bool(const Decl*)>& s,ASTContext& c,const std::function<bool(SourceLocation)>& n):sdk(s),context(c),native(n){}
    bool TraversePseudoObjectExpr(PseudoObjectExpr* expression){
     const auto coordinate=ProveCudaExecutionCoordinate(expression,context,native);
     if(coordinate&&coordinate->expression==expression)return true;
     return RecursiveASTVisitor<Setup>::TraversePseudoObjectExpr(expression);
    }
    bool VisitCallExpr(CallExpr* c){const auto* f=c->getDirectCallee();safe&=f&&!f->isNoReturn()&&(sdk(f)||f->getBuiltinID());return true;}
    bool VisitLambdaExpr(LambdaExpr*){safe=false;return true;}
    bool VisitCXXThrowExpr(CXXThrowExpr*){safe=false;return true;}
   } setup(sdk,context,native);
   for(const auto* statement:body->body()){if(statement==decl)break;if(!isa<DeclStmt>(statement))return false;setup.TraverseStmt(const_cast<Stmt*>(statement));}
   if(!setup.safe)return false;

   fact.callerOffset=sm.getFileOffset(caller->getLocation());
  }return true;
 }
 public:
 Proof(CompilerInstance& c,const Project& project,const std::string& source,const std::function<bool(SourceLocation)>& o,const std::function<bool(const Decl*)>& s,const std::function<bool(SourceLocation)>& n,const DeviceNativeMacroEvidence& m,const std::vector<DeviceAggregateInlineFact>* w):ci(c),context(c.getASTContext()),sm(c.getSourceManager()),p(project),current(source),owned(o),sdk(s),native(n),witnesses(w),macroEvidence(m){inventory.TraverseDecl(context.getTranslationUnitDecl());}
 bool run(DeviceAggregateInlineAnalysis& out,std::string& error){
  out.projectStamp=p.stamp;
   for(const auto& range:inventory.sourceObservations)
    if(!fixedNative(range.getBegin())||!fixedNative(range.getEnd())){
     error="caller typed source-location/function observation has no owned-target identity";return false;
    }
  std::unique_ptr<MangleContext> mangler(context.createMangleContext());
  for(const auto* canonical:inventory.functions){const auto* function=canonical->getDefinition();
   if(!function||!function->hasAttr<CUDADeviceAttr>()||function->hasAttr<CUDAGlobalAttr>()||!literal(function->getLocation())||!function->getIdentifier())continue;
   const auto* record=function->getReturnType()->getAsCXXRecordDecl();if(!record||!record->isCompleteDefinition()||function->getReturnType()->isDependentType()||context.getTypeSize(function->getReturnType())<256)continue;
   if(function->isInvalidDecl()||function->getTemplatedKind()!=FunctionDecl::TK_NonTemplate||function->getType()->isDependentType()||function->getDeclContext()->isDependentContext()){
    error="aggregate helper template/dependent context has no bounded symbol proof";return false;
   }
   for(const auto* parameter:function->parameters())if(parameter->getType()->isDependentType()){error="aggregate helper dependent parameter has no bounded symbol proof";return false;}
   helperDeclarations.insert(canonical);helperSymbols.insert(function->getNameAsString());helperSymbols.insert(function->getQualifiedNameAsString());std::string symbol;llvm::raw_string_ostream out(symbol);
   if(mangler->shouldMangleDeclName(function))mangler->mangleName(GlobalDecl(function),out);else out<<function->getName();out.flush();if(symbol.empty()){error="aggregate helper has no exact compiler symbol";return false;}helperSymbols.insert(symbol);
  }
  for(const auto& a:inventory.assembly)if(!nativeAssemblyDisjoint(a)){
   llvm::json::Array operands;
   const auto operand=[&](const Expr* expression){std::string text;llvm::raw_string_ostream stream(text);expression->printPretty(stream,nullptr,context.getPrintingPolicy());stream.flush();llvm::json::Object row;row["type"]=expression->getType().getAsString(context.getPrintingPolicy());row["expression"]=text;operands.push_back(std::move(row));};
   if(a.gnu){for(unsigned i=0;i<a.gnu->getNumOutputs();++i)operand(a.gnu->getOutputExpr(i));for(unsigned i=0;i<a.gnu->getNumInputs();++i)operand(a.gnu->getInputExpr(i));}
   const auto detail=llvm::formatv("{0}",llvm::json::Value(std::move(operands))).str();
   error="unproved assembly can reference helper symbols: expansion="+sm.getFilename(sm.getExpansionLoc(a.first)).str()+", spelling="+sm.getFilename(sm.getSpellingLoc(a.first)).str()+", decl="+(a.second?a.second->getQualifiedNameAsString():std::string("<global>"))+", label="+(a.second&&a.second->hasAttr<AsmLabelAttr>()?a.second->getAttr<AsmLabelAttr>()->getLabel().str():std::string())+", native_spelling="+std::to_string(native(sm.getSpellingLoc(a.first)))+", native_expansion="+std::to_string(native(sm.getExpansionLoc(a.first)))+", kind="+a.kind+", asm="+(a.gnu?a.gnu->getAsmString():std::string())+", asm_kind="+(a.gnu?a.gnu->getAsmStringExpr()->getStmtClassName():std::string())+", operands="+detail;return false;}
  for(const auto* canonical:inventory.functions){const auto* f=canonical->getDefinition();if(!f||!f->hasAttr<CUDADeviceAttr>()||f->hasAttr<CUDAGlobalAttr>()||!literal(f->getLocation()))continue;
   const auto* r=f->getReturnType()->getAsCXXRecordDecl();if(!r||!r->isCompleteDefinition()||f->getReturnType()->isDependentType()||context.getTypeSize(f->getReturnType())<256)continue;
   DeviceAggregateInlineFact fact;fact.projectStamp=p.stamp;fact.sourcePath=file(f->getLocation());fact.functionOffset=sm.getFileOffset(f->getLocation());fact.signature=f->getType().getCanonicalType().getAsString(context.getPrintingPolicy());
   std::string bytes;if(!read(fact.sourcePath,bytes)){error="candidate source unreadable";return false;}fact.sourceSha256=digest(bytes);
   bool valid=f->getIdentifier()&&!isa<CXXMethodDecl>(f)&&f->getDeclContext()->isTranslationUnit()&&f->getTemplatedKind()==FunctionDecl::TK_NonTemplate&&!f->isVariadic()&&!f->isInlineSpecified()&&!f->getReturnType()->isDependentType();
   for(const auto* arg:f->parameters())valid&=!arg->getType()->isDependentType()&&!arg->getType()->isFunctionPointerType()&&!arg->hasDefaultArg()&&!arg->hasAttrs()&&literal(arg->getLocation());
   unsigned declarations=0;for(const auto* d:f->redecls()){++declarations;valid&=d==f;}valid&=declarations==1;
   for(const auto* a:f->attrs())valid&=isa<CUDADeviceAttr>(a);
   valid&=record(r,fact);
   const auto returnRange=f->getReturnTypeSourceRange();valid&=literal(returnRange.getBegin())&&literal(returnRange.getEnd());
   if(valid){const auto* attribute=f->getAttr<CUDADeviceAttr>();const auto begin=attribute?sm.getExpansionLoc(attribute->getLocation()):SourceLocation();
    const auto prefix=Lexer::getSourceText(CharSourceRange::getCharRange(begin,returnRange.getBegin()),sm,ci.getLangOpts());
    valid=begin.isFileID()&&trim(prefix.str())=="__device__";
   }
   if(valid){const auto spelling=Lexer::getSourceText(CharSourceRange::getTokenRange(returnRange),sm,ci.getLangOpts());valid=spelling==fact.recordName;fact.insertOffset=sm.getFileOffset(returnRange.getBegin());}
   std::set<const Expr*> direct;
   for(const auto& site:inventory.calls)if(site.first->getDirectCallee()&&site.first->getDirectCallee()->getCanonicalDecl()==canonical){++fact.calls;direct.insert(plain(site.first->getCallee()));valid&=site.first->getExprLoc().isFileID()&&site.second&&literal(site.second->getLocation())&&(site.second->hasAttr<CUDADeviceAttr>()||site.second->hasAttr<CUDAGlobalAttr>());}
   for(const auto* reference:inventory.refs)if(cast<FunctionDecl>(reference->getDecl())->getCanonicalDecl()==canonical){
    const auto location=reference->getLocation();valid&=location.isFileID()&&direct.count(reference);
    if(location.isFileID())fact.referenceSites.push_back(file(location)+":"+std::to_string(sm.getFileOffset(location)));
   }
   std::sort(fact.referenceSites.begin(),fact.referenceSites.end());
   const DeviceAggregateInlineFact* witness=nullptr;
   if(witnesses){
    for(const auto& item:*witnesses)if(item.sourcePath==fact.sourcePath&&item.functionOffset==fact.functionOffset){if(witness){valid=false;break;}witness=&item;}
    valid&=witness&&fact.calls<=witness->calls;
    if(witness){
     for(const auto& site:fact.referenceSites)valid&=std::binary_search(witness->referenceSites.begin(),witness->referenceSites.end(),site);
     // Full-root caller authority is inherited only after every locally
     // observed reference has been found in that same immutable root fact.
     fact.calls=witness->calls;fact.callerOffset=witness->callerOffset;fact.referenceSites=witness->referenceSites;
    }
   }
   std::set<const FunctionDecl*> active,done;bool usesCompilerBlockBarrier=false;graphParts.clear();valid&=fact.calls>0&&noRecursion(f,active,done,usesCompilerBlockBarrier)&&shared(f,fact)&&(!usesCompilerBlockBarrier||fact.sharedBytes>0);
   std::sort(graphParts.begin(),graphParts.end());std::string graph;for(const auto& part:graphParts)graph+=part+"\n";fact.bodyGraph=digest(graph);
   if(witness)valid&=fact==*witness;
   if(!valid){error="aggregate helper does not have a closed POD/direct-call/shared proof";return false;}out.facts.push_back(fact);
  }return true;
 }
};
} // namespace
bool ValidateDeviceInlineProjectSnapshot(const std::string& root,const std::string& expected,std::string& error){Project p;return project(root,p,error)&&p.stamp==expected;}
bool AnalyzeDeviceAggregateInlining(CompilerInstance& ci,const std::string& originalRoot,const std::string& currentSource,const std::function<bool(SourceLocation)>& owned,const std::function<bool(const Decl*)>& sdk,const std::function<bool(SourceLocation)>& native,const DeviceNativeMacroEvidence& macroEvidence,const std::function<bool(const DeviceNativeMacroUse&)>& helperDiagnostic,const std::function<bool(FileID)>& completeGuard,const std::vector<DeviceAggregateInlineFact>* rootWitnesses,const std::vector<DeviceRegisterTile32Fact>* rootRegisterWitnesses,DeviceAggregateInlineAnalysis& out,std::string& error){
 Project p;const bool rootParse=canon(currentSource)==canon(originalRoot);
 if(!BindStandaloneDeviceInputs(ci,currentSource,originalRoot,rootParse,native,error)||!project(originalRoot,p,error))return false;const auto current=canon(currentSource);if(current.empty()){error="current physical input missing";return false;}
 const auto& pp=ci.getPreprocessorOpts();if(!ci.getLangOpts().CUDA||ci.getLangOpts().Modules||pp.RetainExcludedConditionalBlocks||!pp.RemappedFiles.empty()||!pp.RemappedFileBuffers.empty()||!pp.ImplicitPCHInclude.empty()){error="ordinary non-remapped CUDA AST required";return false;}
 bool invalid=false;auto& sm=ci.getSourceManager();const auto main=sm.getBufferData(sm.getMainFileID(),&invalid);std::string physical;if(invalid||!read(current,physical)||main!=physical){error="current parsed input differs from physical project source";return false;}
  // A native spelling expanded in caller code is still caller-observable.
  // Only the exact two non-parameter slots of a pending helper transaction
  // have a narrow callback; its commit is mandatory before publication.
  static const std::set<std::string> observedPositionMacros={
      "__FILE__","__FILE_NAME__","__BASE_FILE__","__LINE__",
      "__INCLUDE_LEVEL__","__COUNTER__","__TIMESTAMP__","__DATE__","__TIME__"};
  for(const auto& use:macroEvidence.uses){
   if(!use.definition||!use.definition->isBuiltinMacro()||!observedPositionMacros.count(use.name))continue;
   const auto spelling=sm.getSpellingLoc(use.nameLocation),expansion=sm.getExpansionLoc(use.nameLocation);
   if(spelling.isFileID()&&expansion.isFileID()&&native(spelling)&&native(expansion))continue;
   if(helperDiagnostic(use))continue;
   error="caller builtin source observation has no owned-target identity: "+use.name;return false;
  }
  // Raw caller directive controls are checked even in inactive branches. The
 // live/historical macro flag is additionally checked by AscifyAction.
 for(auto it=sm.fileinfo_begin();it!=sm.fileinfo_end();++it){const auto id=sm.translateFile(it->first);if(id.isInvalid())continue;const auto at=sm.getLocForStartOfFile(id);if(native(at))continue;
  bool bad=false;const auto bytes=sm.getBufferData(id,&bad);if(bad)continue;

  const std::string data=bytes.str();
  // The emitted compiler may have different architecture/version macros.
  // No caller conditional can hide a later address/call/shared reuse. This
  // first contract accepts unguarded caller files and literal unroll pragmas.
  Lexer raw(sm.getLocForStartOfFile(id),ci.getLangOpts(),bytes.begin(),bytes.begin(),bytes.end());
  const bool guarded=completeGuard(id);unsigned conditions=0;
  Token token;for(;;){raw.LexFromRawLexer(token);if(token.is(tok::eof))break;
   if(token.is(tok::hash)&&token.isAtStartOfLine()){raw.LexFromRawLexer(token);if(token.is(tok::numeric_constant)){error="caller numeric line remapping has no physical include-edge identity";return false;}if(token.is(tok::raw_identifier)){
    const auto directive=token.getRawIdentifier();if(directive=="line"){error="caller line remapping has no physical include-edge identity";return false;}if(directive=="if"||directive=="ifdef"||directive=="ifndef"||directive=="elif"||directive=="else"||directive=="endif"){++conditions;if(!guarded||conditions>2||(conditions==1&&directive!="ifndef")||(conditions==2&&directive!="endif")){error="caller conditional preprocessing is not a solely controlled whole-file outer guard: file="+sm.getFilename(at).str()+", file_id="+std::to_string(id.getHashValue())+", directive="+directive.str()+", offset="+std::to_string(sm.getFileOffset(token.getLocation()))+", guarded="+std::to_string(guarded);return false;}}
   }}
  }
  if(conditions!=0&&conditions!=2){error="incomplete caller guard";return false;}
  if(std::regex_search(data,std::regex(R"re(#[ \t]*(define|undef)[ \t]+(inline|__attribute__|always_inline|static_assert|sizeof|alignof)\b)re"))||data.find("push_macro")!=std::string::npos||data.find("pop_macro")!=std::string::npos||data.find("_Pragma")!=std::string::npos||data.find("__pragma")!=std::string::npos){error="caller raw preprocessing controls introduced inline/layout tokens";return false;}
 }
  if(!Proof(ci,p,current,owned,sdk,native,macroEvidence,rootWitnesses).run(out,error))return false;
  std::vector<RegisterTile32Factory> factories;
  if(!ProveRegisterTile32Factories(ci.getASTContext(),
      [&](SourceLocation at){return at.isFileID()&&owned(at);},sdk,factories,error))return false;
  if(!factories.empty()){
   const std::set<std::string> controlled={"ascify_cg","register_partition32_v1","register_tile32_v1","ASCIFY_REGISTER_TILE32_COMPAT_HPP"};
   for(const auto& event:macroEvidence.controls)if(controlled.count(event.first)){
    error="register adapter identifier has macro definition/undefinition history";return false;
   }
   for(const auto& word:controlled)if(ci.getPreprocessor().getMacroDefinition(ci.getPreprocessor().getIdentifierInfo(word))){
    error="register adapter identifier is a current macro";return false;
   }
   // Inactive directives are not MacroDefined callbacks. Inspect the actual
   // caller buffers too; cleaned directive/name tokens never gain authority.
   for(auto it=sm.fileinfo_begin();it!=sm.fileinfo_end();++it){
    const auto id=sm.translateFile(it->first);if(id.isInvalid())continue;
    const auto at=sm.getLocForStartOfFile(id);if(native(at))continue;
    bool invalid=false;const auto bytes=sm.getBufferData(id,&invalid);
    if(invalid){error="register caller raw input is unavailable";return false;}
    Lexer raw(at,ci.getLangOpts(),bytes.begin(),bytes.begin(),bytes.end());Token token;
    for(;;){raw.LexFromRawLexer(token);if(token.is(tok::eof))break;
     if(!token.is(tok::hash)||!token.isAtStartOfLine())continue;
     raw.LexFromRawLexer(token);const auto directive=Lexer::getSpelling(token,sm,ci.getLangOpts());
     if(directive!="define"&&directive!="undef")continue;
     raw.LexFromRawLexer(token);const auto word=Lexer::getSpelling(token,sm,ci.getLangOpts());
     if(controlled.count(word)){error="register adapter identifier has caller raw directive history";return false;}
    }
   }
  }
  out.registerFactories.clear();
  for(const auto& factory:factories){
   const auto file=sm.getFileID(factory.begin);bool invalid=false;
   const auto bytes=sm.getBufferData(file,&invalid);const auto path=canon(sm.getFilename(factory.begin).str());std::string physical;
   if(invalid||path.empty()||!read(path,physical)||bytes!=physical){error="register factory physical input changed";return false;}
   DeviceRegisterTile32Fact fact{p.stamp,path,digest(physical),sm.getFileOffset(factory.begin),sm.getFileOffset(factory.end)};
   if(fact.beginOffset>=fact.endOffset||fact.endOffset>physical.size()){error="register factory has no complete physical range";return false;}
   if(rootRegisterWitnesses&&std::find(rootRegisterWitnesses->begin(),rootRegisterWitnesses->end(),fact)==rootRegisterWitnesses->end()){
    error="register factory differs from complete original root AST proof";return false;
   }
   out.registerFactories.push_back(std::move(fact));
  }
  return true;
}
} // namespace ascify
