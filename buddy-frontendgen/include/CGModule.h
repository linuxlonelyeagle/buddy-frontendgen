#ifndef INCLUDE_CGMODULE_H
#define INCLUDE_CGMODULE_H

#include "AST.h"
#include "Terminator.h"
#include "llvm/Support/raw_ostream.h"
namespace frontendgen {
/// The class for code generation.
class CGModule {
  Terminators &terminators;
  Module *module;
  llvm::raw_fd_ostream &os;
  llvm::StringMap<llvm::StringRef> typeMap;

public:
  CGModule(Module *module, llvm::raw_fd_ostream &os, Terminators &terminators)
      : module(module), os(os), terminators(terminators) { initTypeMap();}
  void initTypeMap();
  void lookTypeMap();
  llvm::StringRef findType(llvm::StringRef key);
  void emitAST();
  void emitAntlr(llvm::StringRef grammarName);
  void emit(const std::vector<Rule *> &rules);
  void emit(const std::vector<GeneratorAndOthers*> &generators);
  void emit(const std::vector<AntlrBase *> &generator);
  void emitGrammar(llvm::StringRef grammarName);
  void emitTerminators();
  void emitCustomTerminators();
  void emitWSAndComment();
  void emitTd();
  void emitDialect();
  void emitOps();
  void emitOpInterfaces();
  void emitIncludes();
  void emitIncludes(llvm::StringRef grammarName);
  void emitMLIRVisitor(llvm::StringRef grammarName);
  void emitClass(llvm::StringRef grammarName);
  void emitRuleVisitor(llvm::StringRef grammarName, Rule* rule);
  void emitBuilders(Rule* rule);
  void emitBuilder(llvm::StringRef buildrOp, int index);
  Op* findOp(llvm::StringRef opName);
  void emitOp(Op* op, int index);
};
} // namespace frontendgen

#endif