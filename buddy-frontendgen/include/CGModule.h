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

public:
  CGModule(Module *module, llvm::raw_fd_ostream &os, Terminators &terminators)
      : module(module), os(os), terminators(terminators) {}
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
};
} // namespace frontendgen

#endif