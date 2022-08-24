#ifndef INCLUDE_TOKEN
#define INCLUDE_TOKEN
#include "Lexer.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
namespace frontendgen {
enum tokenKinds {
#define TOK(ID) ID,
#include "Token.def"
  NUM_TOKENS
};
/// store token names.
static const char *tokenNameMap[] = {
#define TOK(ID) #ID,
#define KEYWORD(ID, FLAG) #ID,
#include "Token.def"
    nullptr};

class Token {
  friend class Lexer;

private:
  tokenKinds tokenKind;
  const char *start;
  int length;

public:
  void setTokenKind(tokenKinds kind) { tokenKind = kind; }
  void setLength(int len) { length = len; }

  llvm::StringRef getContent() { return llvm::StringRef(start, length); }
  tokenKinds getKind() { return tokenKind; }
  const char *getTokenName() { return tokenNameMap[tokenKind]; }
  bool is(tokenKinds kind);
  llvm::SMLoc getLocation();
};

} // namespace frontendgen
#endif