#ifndef INCLUDE_SEMA_H
#define INCLUDE_SEMA_H
#include "AST.h"

namespace frontendgen {

class Sema {
public:
  void actOnModule(Module *module, std::vector<Rule *> &rules,
                   Dialect *&dialect, std::vector<Op *> &ops,
                   std::vector<Opinterface *> &opInterfaces);
  void actOnRule(Rule *rule, std::vector<GeneratorAndOthers *> &generators);
  void actOnDialect(Dialect *dialect, llvm::StringRef defName,
                    llvm::StringRef name, llvm::StringRef emitAccessorPrefix,
                    llvm::StringRef cppNamespace);
  void actOnOps(std::vector<Op *> &ops, llvm::StringRef opName,
                llvm::StringRef mnemonic, llvm::StringRef traits,
                llvm::StringRef summary, llvm::StringRef description,
                DAG *arguments, DAG *results, bool hasCustomAssemblyFormat,
                std::vector<Builder *> &builders, bool hasVerifier,
                llvm::StringRef assemblyFormat, llvm::StringRef regions,
                llvm::StringRef extraClassDeclaration, bool skipDefaultBuilders,
                bool hasCanonicalizer);
  void actOnOpInterfaces(std::vector<Opinterface *> &opInterfaces,
                         llvm::StringRef defName, llvm::StringRef name,
                         llvm::StringRef methods, llvm::StringRef description);
  void actOnDag(DAG *&arguments, DAG &dag);
};
} // namespace frontendgen
#endif