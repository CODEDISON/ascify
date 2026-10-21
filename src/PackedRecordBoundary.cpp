#include "PackedRecordBoundary.h"

#include <algorithm>
#include <map>
#include <regex>
#include <set>
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"

namespace ascify {
namespace {
// Obtain the record type through the available ASTContext API. Newer Clang
// lazily creates tag types and explicitly deletes TagDecl::getTypeForDecl.
template <class Context, class Record>
auto packedRecordType(Context &context, const Record *record, int)
    -> decltype(context.getRecordType(record), clang::QualType()) {
  return context.getRecordType(record);
}
template <class Context, class Record>
auto packedRecordType(Context &context, const Record *record, long)
    -> decltype(context.getCanonicalTagType(record), clang::QualType()) {
  return context.getCanonicalTagType(record);
}
unsigned lineBegin(llvm::StringRef b, unsigned p) {
  const auto at = b.rfind('\n', p);
  return at == llvm::StringRef::npos ? 0 : at + 1;
}
unsigned lineEnd(llvm::StringRef b, unsigned p) {
  const auto at = b.find('\n', p);
  return at == llvm::StringRef::npos ? b.size() : at + 1;
}
bool literal(llvm::StringRef line, const char *expression) {
  return !line.contains('\\') && !line.contains("??") &&
      std::regex_match(line.str(), std::regex(expression));
}
const char *Open = R"(^[ \t]*#[ \t]*pragma[ \t]+pack[ \t]*\([ \t]*1[ \t]*\)[ \t]*\r?\n$)";
const char *Reset = R"(^[ \t]*#[ \t]*pragma[ \t]+pack[ \t]*\([ \t]*\)[ \t]*\r?\n$)";
const char *Pop = R"(^[ \t]*#[ \t]*pragma[ \t]+pack[ \t]*\([ \t]*pop[ \t]*\)[ \t]*\r?\n$)";
const char *Else = R"(^[ \t]*#[ \t]*else[ \t]*\r?\n$)";
const char *Endif = R"(^[ \t]*#[ \t]*endif[ \t]*\r?\n$)";
bool footerNames(llvm::StringRef line, std::vector<std::string> &names) {
  std::smatch matched;
  const std::string text = line.str();
  if (text.find('\\') != std::string::npos || text.find("??") != std::string::npos ||
      !std::regex_match(text, matched, std::regex(R"(^[ \t]*#[ \t]*if[ \t]+([^\r\n]+)\r?\n$)"))) return false;
  std::string remaining = matched[1];
  const std::regex term(R"(^[ \t]*defined[ \t]*\([ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*\)[ \t]*)");
  std::set<std::string> unique;
  for (;;) {
    std::smatch part;
    if (!std::regex_search(remaining, part, term)) return false;
    const std::string name = part[1];
    if (!unique.insert(name).second) return false;
    names.push_back(name);
    remaining.erase(0, part.length());
    if (remaining.empty()) return !names.empty();
    if (remaining.substr(0, 2) != "||") return false;
    remaining.erase(0, 2);
  }
}
bool ordinary(clang::CompilerInstance &ci) {
  const auto &pp = ci.getPreprocessorOpts();
  const auto &lang = ci.getLangOpts();
  return !pp.RetainExcludedConditionalBlocks &&
      !lang.Modules && !lang.CPlusPlusModules && !lang.PackStruct &&
      !lang.ApplePragmaPack && !lang.XLPragmaPack &&
      pp.ImplicitPCHInclude.empty() && pp.PCHThroughHeader.empty() &&
      pp.PrecompiledPreambleBytes.first == 0 && !pp.PrecompiledPreambleBytes.second &&
      ci.getFrontendOpts().ModuleFiles.empty() && ci.getASTContext().getExternalSource() == nullptr;
}
bool undefined(clang::CompilerInstance &ci, const std::vector<std::string> &names) {
  auto &pp = ci.getPreprocessor();
  for (const auto &name : names)
    if (pp.getMacroDefinition(pp.getIdentifierInfo(name)).getMacroInfo()) return false;
  return true;
}
}

bool PackedRecordBoundary::fail(const std::string &reason) {
  if (failure.empty()) failure = reason;
  return false;
}

bool PackedRecordBoundary::pragma(clang::CompilerInstance &ci,
    clang::SourceLocation loc, bool open, bool reset, bool selected,
    bool governedVirtualMain) {
  if (!open && !reset) return false;
  auto &sm = ci.getSourceManager();
  if (!loc.isFileID() || !selected || !ordinary(ci))
    return fail("packed region requires selected ordinary physical input and default packing options");
  const auto file = sm.getFileID(loc);
  bool invalid = false;
  const auto bytes = sm.getBufferData(file, &invalid);
  const auto begin = lineBegin(bytes, sm.getFileOffset(loc));
  if (invalid || begin >= bytes.size()) return fail("packed region has no physical source line");
  if (reset) {
    if (!active || regions.back().file != file || regions.back().reset != begin ||
        !undefined(ci, regions.back().undefinedNames))
      return fail("packed reset has no identical active source region");
    regions.back().closed = true;
    active = false;
    return true;
  }
  if (active) return fail("nested packed regions are outside the preserved-record proof");
  const auto *entry = sm.getFileEntryForID(file);
  if (!entry || (sm.isFileOverridden(entry) && !governedVirtualMain))
    return fail("packed input is not an authenticated physical or governed virtual source");
  if (!governedVirtualMain) {
    const auto disk = llvm::MemoryBuffer::getFile(sm.getFilename(loc));
    llvm::sys::fs::UniqueID physical;
    if (!disk || (*disk)->getBuffer() != bytes ||
        llvm::sys::fs::getUniqueID(sm.getFilename(loc), physical) ||
        physical != entry->getUniqueID())
      return fail("packed physical input differs from parsed source");
  }
  Region r;
  r.file = file; r.path = sm.getFilename(loc).str(); r.bytes = bytes.str(); r.begin = begin; r.open = begin;
  r.bodyBegin = lineEnd(bytes, begin);
  if (!literal(bytes.slice(begin, r.bodyBegin), Open)) return fail("nonliteral packed opener");
  r.pragmaLines.push_back(begin);
  const auto start = sm.getLocForStartOfFile(file);
  clang::Lexer lexer(start, ci.getLangOpts(), bytes.begin(), bytes.begin()+r.bodyBegin, bytes.end());
  clang::Token token;
  unsigned directive = bytes.size();
  for (;;) {
    lexer.LexFromRawLexer(token);
    if (token.is(clang::tok::eof)) break;
    if (token.needsCleaning()) return fail("packed body uses continued or cleaned tokens");
    if (token.is(clang::tok::hash) && token.isAtStartOfLine()) {
      directive = lineBegin(bytes, sm.getFileOffset(token.getLocation())); break;
    }
  }
  if (directive == bytes.size()) return fail("packed region has no same-file reset");
  r.bodyEnd = directive;
  unsigned cursor = directive;
  auto nextLine = [&]() {
    const unsigned next = lineEnd(bytes, cursor);
    const auto line = bytes.slice(cursor, next); cursor = next; return line;
  };
  const auto first = nextLine();
  if (literal(first, Reset)) {
    r.reset = directive;
  } else {
    if (!footerNames(first, r.undefinedNames) || !undefined(ci, r.undefinedNames))
      return fail("packed footer is not a proved inactive defined-name disjunction");
    r.pragmaLines.push_back(cursor);
    if (!literal(nextLine(), Pop) || !literal(nextLine(), Else))
      return fail("packed conditional footer differs from literal pop/else/reset/endif");
    r.reset = cursor;
    if (!literal(nextLine(), Reset) || !literal(nextLine(), Endif))
      return fail("packed conditional footer is incomplete");
  }
  // Optional inactive push prefix uses the exact same condition as the
  // footer. It is preserved, never executed in this admitted configuration.
  if (!r.undefinedNames.empty()) {
    unsigned previous = begin;
    std::vector<std::pair<unsigned, llvm::StringRef>> lines;
    while (previous && lines.size() < 3) {
      const unsigned before = previous - 1;
      const unsigned first = before ? lineBegin(bytes, before - 1) : 0;
      const auto line = bytes.slice(first, previous);
      previous = first;
      if (!line.trim().empty()) lines.emplace_back(first, line);
    }
    if (lines.size() == 3 && literal(lines[0].second, Endif) &&
        literal(lines[1].second, R"(^[ \t]*#[ \t]*pragma[ \t]+pack[ \t]*\([ \t]*push[ \t]*\)[ \t]*\r?\n$)")) {
      std::vector<std::string> prefixNames;
      if (!footerNames(lines[2].second, prefixNames) || prefixNames != r.undefinedNames)
        return fail("packed push prefix differs from the proved inactive footer");
      r.begin = lines[2].first;
      r.pragmaLines.push_back(lines[1].first);
    }
  }
  r.pragmaLines.push_back(r.reset);
  r.end = cursor;
  regions.push_back(std::move(r)); active = true;
  return true;
}

void PackedRecordBoundary::includeOrFileTransition() {
  if (active) fail("include or file transition inside active packed region");
}

bool PackedRecordBoundary::prove(clang::CompilerInstance &ci,
    bool unrelatedPragmaFailure, std::vector<clang::tooling::Replacement> &owned) {
  owned.clear();
  if (regions.empty()) return failure.empty();
  if (!failure.empty() || active || unrelatedPragmaFailure || !ordinary(ci))
    return fail("packed region has unresolved preprocessing state");
  auto &ast = ci.getASTContext(); auto &sm = ci.getSourceManager();
  const auto ownedLines = ownedPragmaLines();
  std::set<clang::FileID> scanned;
  for (const auto &r : regions) {
    if (!scanned.insert(r.file).second) continue;
    const auto buffer = sm.getBufferData(r.file);
    const auto start = sm.getLocForStartOfFile(r.file);
    clang::Lexer raw(start, ci.getLangOpts(), buffer.begin(), buffer.begin(), buffer.end());
    clang::Token token;
    unsigned directive = 0, phase = 0;
    for (;;) {
      raw.LexFromRawLexer(token);
      if (token.is(clang::tok::eof)) break;
      const auto word = clang::Lexer::getSpelling(token, sm, ci.getLangOpts());
      if (token.needsCleaning() || token.is(clang::tok::hashhash) ||
          (token.isAnyIdentifier() && (word == "_Pragma" || word == "__pragma")))
        return fail("packed source contains opaque pragma construction or cleaned tokens");
      if (phase == 2 && word == "pack" &&
          std::find(ownedLines.begin(), ownedLines.end(), std::make_pair(r.file, directive)) == ownedLines.end())
        return fail("packed source contains an unowned active or inactive pack directive");
      phase = phase == 1 && word == "pragma" ? 2 : 0;
      if (token.is(clang::tok::hash) && token.isAtStartOfLine()) {
        phase = 1; directive = lineBegin(buffer, sm.getFileOffset(token.getLocation()));
      }
    }
  }
  for (const auto &r : regions) {
    if (!r.closed || sm.getBufferData(r.file) != r.bytes)
      return fail("packed region changed or did not close");
    class Records : public clang::RecursiveASTVisitor<Records> {
      clang::ASTContext &ast; clang::SourceManager &sm;
      const Region &r;
      bool contained(const clang::Decl *d) {
        return d->getBeginLoc().isFileID() && d->getEndLoc().isFileID() &&
            sm.getFileID(d->getBeginLoc()) == r.file && sm.getFileID(d->getEndLoc()) == r.file &&
            sm.getFileOffset(d->getBeginLoc()) >= r.bodyBegin &&
            sm.getFileOffset(d->getEndLoc()) < r.bodyEnd;
      }
      bool scalar(clang::QualType type) {
        if (type.hasLocalQualifiers()) return false;
        if (const auto *alias = llvm::dyn_cast<clang::TypedefType>(type.getTypePtr())) {
          if (!contained(alias->getDecl())) return false;
          return scalar(alias->getDecl()->getUnderlyingType());
        }
        const auto *builtin = llvm::dyn_cast<clang::BuiltinType>(type.getTypePtr());
        if (!builtin) return false;
        const auto kind = builtin->getKind();
        const unsigned bits = ast.getTypeSize(type);
        return ((kind == clang::BuiltinType::Char_S || kind == clang::BuiltinType::Char_U ||
                 kind == clang::BuiltinType::SChar || kind == clang::BuiltinType::UChar) && bits == 8) ||
            ((kind == clang::BuiltinType::Short || kind == clang::BuiltinType::UShort) && bits == 16) ||
            ((kind == clang::BuiltinType::Int || kind == clang::BuiltinType::UInt) && bits == 32);
      }
     public:
      bool good = true;
      unsigned count = 0;
      std::string witnesses;
      std::set<std::string> names;
      Records(clang::ASTContext &a, const Region &region) : ast(a), sm(a.getSourceManager()), r(region) {}
      bool VisitDecl(clang::Decl *d) {
        if (d->isImplicit() || llvm::isa<clang::TranslationUnitDecl>(d)) return true;
        const auto b = sm.getExpansionLoc(d->getBeginLoc()), e = sm.getExpansionLoc(d->getEndLoc());
        if (b.isInvalid() || e.isInvalid() || sm.getFileID(b) != r.file || sm.getFileID(e) != r.file ||
            sm.getFileOffset(e) < r.bodyBegin || sm.getFileOffset(b) >= r.bodyEnd) return true;
        if (!contained(d)) { good = false; return true; }
        if (const auto *field = llvm::dyn_cast<clang::FieldDecl>(d)) {
          if (field->getName().empty() || field->hasAttrs() || field->isBitField() ||
              field->hasInClassInitializer() || !scalar(field->getType())) good = false;
          names.insert(field->getName().str()); return true;
        }
        if (!d->getDeclContext()->isTranslationUnit()) { good = false; return true; }
        if (const auto *alias = llvm::dyn_cast<clang::TypedefNameDecl>(d)) {
          if (alias->hasAttrs() || alias->getName().empty()) { good = false; return true; }
          const auto t = alias->getUnderlyingType();
          if (t->isDependentType()) { good = false; return true; }
          if (!t->isRecordType() && !scalar(t)) { good = false; return true; }
          if (const auto *record = t->getAs<clang::RecordType>())
            if (!contained(record->getDecl()) || !record->getDecl()->isCompleteDefinition()) {
              good = false; return true;
            }
          const auto name = alias->getName().str(); names.insert(name);
          witnesses += "static_assert(sizeof(" + name + ")==" + std::to_string(ast.getTypeSize(t)/8) +
              " && alignof(" + name + ")==" + std::to_string(ast.getTypeAlign(t)/8) +
              ",\"Ascify preserved packed type layout\");\n";
          if (!t->isRecordType())
            witnesses += "static_assert((static_cast<" + name + ">(-1)<0)==" +
                std::to_string(t->isSignedIntegerType()) + ",\"Ascify preserved integer signedness\");\n";
          return true;
        }
        const auto *record = llvm::dyn_cast<clang::RecordDecl>(d);
        if (!record || !record->isCompleteDefinition() || !record->isStruct() ||
            record->isDependentContext()) {
          good = false; return true;
        }
        const auto type = packedRecordType(ast, record, 0);
        if (type.isNull() || type->isDependentType()) { good = false; return true; }
        for (const auto *attribute : record->attrs()) {
          const auto *pack = llvm::dyn_cast<clang::MaxFieldAlignmentAttr>(attribute);
          if (!pack || !pack->isImplicit() || pack->getAlignment() != 8) good = false;
        }
        if (!record->hasAttr<clang::MaxFieldAlignmentAttr>()) good = false;
        if (const auto *cxx = llvm::dyn_cast<clang::CXXRecordDecl>(record))
          if (cxx->getNumBases() || !cxx->isStandardLayout() || !cxx->isTriviallyCopyable() || !cxx->isAggregate()) good = false;
        std::string name = record->getName().str();
        if (name.empty()) if (const auto *alias = record->getTypedefNameForAnonDecl()) name = alias->getName().str();
        if (name.empty()) { good = false; return true; }
        names.insert(name);
        for (const auto *field : record->fields())
          if (!scalar(field->getType())) { good = false; return true; }
        const auto &layout = ast.getASTRecordLayout(record);
        if (layout.getAlignment().getQuantity() != 1) good = false;
        unsigned offset = 0, index = 0;
        for (const auto *field : record->fields()) {
          if (!scalar(field->getType()) || layout.getFieldOffset(index++) != offset || field->getName().empty()) good = false;
          witnesses += "static_assert(__builtin_offsetof(" + name + "," + field->getName().str() + ")==" +
              std::to_string(offset/8) + ",\"Ascify preserved packed field offset\");\n";
          witnesses += "static_assert(sizeof(decltype(((" + name + "*)nullptr)->" + field->getName().str() + "))==" +
              std::to_string(ast.getTypeSize(field->getType())/8) +
              " && (static_cast<decltype(((" + name + "*)nullptr)->" + field->getName().str() + ")>(-1)<0)==" +
              std::to_string(field->getType()->isSignedIntegerType()) +
              ",\"Ascify preserved packed integer field\");\n";
          offset += ast.getTypeSize(field->getType());
        }
        if (!index || layout.getSize().getQuantity()*8 != offset) good = false;
        witnesses += "static_assert(sizeof(" + name + ")==" + std::to_string(offset/8) +
            " && alignof(" + name + ")==1,\"Ascify preserved packed record\");\n";
        ++count; return true;
      }
    } records(ast, r);
    records.TraverseDecl(ast.getTranslationUnitDecl());
    if (!records.good || !records.count) return fail("packed body is not unchanged plain fixed-integer records and typedefs");
    const auto start = sm.getLocForStartOfFile(r.file);
    llvm::StringRef bytes(r.bytes);
    clang::Lexer lexer(start, ci.getLangOpts(), bytes.begin(), bytes.begin()+r.bodyBegin, bytes.end());
    clang::Token token;
    const std::set<std::string> keywords = {"typedef", "struct", "char", "short", "int", "signed", "unsigned"};
    for (;;) {
      lexer.LexFromRawLexer(token);
      if (token.is(clang::tok::eof) || sm.getFileOffset(token.getLocation()) >= r.bodyEnd) break;
      if (token.needsCleaning()) return fail("packed raw body requires literal tokens");
      if (token.isAnyIdentifier()) {
        const auto word = clang::Lexer::getSpelling(token, sm, ci.getLangOpts());
        if ((!keywords.count(word) && !records.names.count(word)) ||
            ci.getPreprocessor().getMacroDefinitionAtLoc(ci.getPreprocessor().getIdentifierInfo(word), token.getLocation()).getMacroInfo())
          return fail("packed body contains a macro or an unproved identifier");
      } else if (!token.isOneOf(clang::tok::l_brace, clang::tok::r_brace, clang::tok::semi)) {
        return fail("packed body contains nondeclaration syntax or source attributes");
      }
    }
    std::string guard;
    if (!r.undefinedNames.empty()) {
      guard = "#if ";
      for (size_t i=0;i<r.undefinedNames.size();++i) {
        if (i) guard += " || ";
        guard += "defined(" + r.undefinedNames[i] + ")";
      }
      guard += "\n#error Ascify packed region requires the original inactive configuration\n#endif\n";
      owned.emplace_back(sm, start.getLocWithOffset(r.open), 0, guard);
    }
    const char *witnessSyntax[] = {"static_assert", "sizeof", "alignof", "static_cast",
                                   "__builtin_offsetof", "decltype", "nullptr"};
    std::string syntaxGuard = "#if ";
    for (size_t i=0; i<sizeof(witnessSyntax)/sizeof(witnessSyntax[0]); ++i) {
      const auto *identifier = ci.getPreprocessor().getIdentifierInfo(witnessSyntax[i]);
      if (ci.getPreprocessor().getMacroDefinitionAtLoc(identifier, start.getLocWithOffset(r.end)).getMacroInfo())
        return fail("generated packed layout witness syntax is shadowed at insertion");
      if (i) syntaxGuard += " || ";
      syntaxGuard += std::string("defined(") + witnessSyntax[i] + ")";
    }
    syntaxGuard += "\n#error Ascify packed layout witness syntax must remain unshadowed\n#endif\n";
    owned.emplace_back(sm, start.getLocWithOffset(r.end), 0, syntaxGuard + records.witnesses);
  }
  return true;
}

bool PackedRecordBoundary::auditFinalEdits(const clang::tooling::Replacements &edits,
    const std::vector<clang::tooling::Replacement> &owned) {
  for (const auto &edit : edits) {
    bool ours = false;
    for (const auto &allowed : owned)
      if (edit.getFilePath() == allowed.getFilePath() && edit.getOffset() == allowed.getOffset() &&
          edit.getLength() == allowed.getLength() && edit.getReplacementText() == allowed.getReplacementText()) ours = true;
    if (ours) continue;
    for (const auto &r : regions) {
      if (edit.getFilePath() != r.path &&
          !llvm::sys::fs::equivalent(edit.getFilePath(), r.path)) continue;
      const uint64_t begin = edit.getOffset(), end = begin + edit.getLength();
      if ((edit.getLength() == 0 && begin >= r.begin && begin <= r.end) ||
          (begin < r.end && end > r.begin))
        return fail("final replacement overlaps protected original packed source");
    }
  }
  return failure.empty();
}

std::vector<std::pair<clang::FileID, unsigned>> PackedRecordBoundary::ownedPragmaLines() const {
  std::vector<std::pair<clang::FileID, unsigned>> result;
  if (!failure.empty() || active) return result;
  for (const auto &r : regions) if (r.closed)
    for (const auto offset : r.pragmaLines) result.emplace_back(r.file, offset);
  return result;
}
}
