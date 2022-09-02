#ifndef INCLUDE_CGMODULE_H
#define INCLUDE_CGMODULE_H

#include "AST.h"
#include "Terminator.h"
#include "llvm/Support/raw_ostream.h"
namespace frontendgen {
class TypeMap{
  llvm::StringMap<llvm::StringRef> cppMap;
  llvm::StringMap<llvm::StringRef> argmentsMap;
  llvm::StringMap<llvm::StringRef> resultsMap;
  public:
  TypeMap() {
    #define CPPMAP(key, value) cppMap.insert(std::pair(key, value));
    #define RESULTSMAP(key, value) resultsMap.insert(std::pair(key, value));
    #define ARGUMENTSMAP(key, value) argmentsMap.insert(std::pair(key, value));
    #include "TypeMap.def"
  } 
  llvm::StringRef findCppMap(llvm::StringRef value);
  llvm::StringRef findArgumentMap(llvm::StringRef value);
  llvm::StringRef findResultsMap(llvm::StringRef value);
};

/// The class for code generation.
class CGModule {
  Terminators &terminators;
  Module *module;
  llvm::raw_fd_ostream &os;
  TypeMap typeMap;
public:
  CGModule(Module *module, llvm::raw_fd_ostream &os, Terminators &terminators)
      : module(module), os(os), terminators(terminators) { }
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