#ifndef INCLUDE_LEXER_H
#define INCLUDE_LEXER_H
#include "Diagnostics.h"
#include "Token.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/SourceMgr.h"
namespace frontendgen {

/// Manage all keywords.
class KeyWordManager {
  llvm::StringMap<tokenKinds> keywordMap;
  void addKeyWords();

public:
  KeyWordManager() { addKeyWords(); }
  void addKeyWord(llvm::StringRef name, tokenKinds kind);
  tokenKinds getKeyWord(llvm::StringRef name, tokenKinds kind);
};

class Lexer {
  llvm::SourceMgr &srcMgr;
  DiagnosticEngine &diagnostic;
  const char *curPtr;
  llvm::StringRef curBuffer;
  KeyWordManager keywordManager;

public:
  Lexer(llvm::SourceMgr &srcMgr, DiagnosticEngine &diagnostic)
      : srcMgr(srcMgr), diagnostic(diagnostic) {
    curBuffer = srcMgr.getMemoryBuffer(srcMgr.getMainFileID())->getBuffer();
    curPtr = curBuffer.begin();
  }
  DiagnosticEngine &getDiagnostic() { return diagnostic; }
  void next(Token &token);
  void identifier(Token &token);
  void number(Token &token);
  void formToken(Token &token, const char *tokenEnd, tokenKinds kind);
  llvm::StringRef getMarkContent(std::string start, std::string end);
};

} // namespace frontendgen
#endif