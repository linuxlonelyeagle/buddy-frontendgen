#ifndef INCLUDE_PARSER_H
#define INCLUDE_PARSER_H
#include "AST.h"
#include "Lexer.h"
#include "Sema.h"
#include "Terminator.h"
#include "Token.h"
namespace frontendgen {

/// A class for parsing tokens.
class Parser {
  Lexer &lexer;
  Token token;
  Sema &action;
  Terminators &terminators;

public:
  Parser(Lexer &lexer, Sema &action, Terminators &terminators)
      : lexer(lexer), action(action), terminators(terminators) {
    advance();
  }
  bool consume(tokenKinds kind);
  bool consumeNoAdvance(tokenKinds kind);
  void advance();
  Module *parser();
  void compilEngine(Module *module);
  void parserRules(Rule *rule);
  void parserGenerator(GeneratorAndOthers* generatorAndOthers);
  void lookToken();
  AntlrBase::baseKind getAntlrBaseKind(llvm::StringRef name);
  void parserIdentifier(GeneratorAndOthers* generatorAndOthers);
  void parserTerminator(GeneratorAndOthers* generatorAndOthers);
  void parserPBExpression(GeneratorAndOthers* generatorAndOthers);
  void parserDialect(Dialect *&dialect, llvm::StringRef defName);
  bool parserOp(std::vector<Op *> &ops, llvm::StringRef opName);
  bool parserOpinterface(std::vector<Opinterface *> &opInterfaces);
  void parserCurlyBracketOpen(GeneratorAndOthers* generatorAndOthers);
  void parserDAG(DAG*& dag);
  void parserBuilders(std::vector<Builder*>& builders);
  void parserCode(llvm::StringRef& code);
  void parserCArg(llvm::StringRef& operand, llvm::StringRef& value);
};
} // namespace frontendgen

#endif