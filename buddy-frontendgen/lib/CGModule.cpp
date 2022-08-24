#include "CGModule.h"

using namespace frontendgen;
/// Emit the ast,currently only antlr's ast are supported.
void CGModule::emitAST() {
  for (auto i : module->getRules()) {
    llvm::outs() << "rule name: " << i->getName() << '\n';
    for (auto j : i->getGenerators()) {
      llvm::outs() << "  generator: " << '\n' << "    ";
      for (auto k : j) {
        if (k->getKind() == AntlrBase::baseKind::rule)
          llvm::outs() << "\"" << k->getName() << "\"(rule) ";
        else if (k->getKind() == AntlrBase::baseKind::terminator)
          llvm::outs() << "\"" << k->getName() << "\"(terminator) ";
        else if (k->getKind() == AntlrBase::baseKind::pbexpression)
          llvm::outs() << "\"" << k->getName() << "\"(bpExpression) ";
      }
      llvm::outs() << '\n';
    }
  }
}

/// Emit the code of antlr , emit the generative formula first, then emit
/// user-defined terminator , and finally emit the system-defined terminator.
void CGModule::emitAntlr(llvm::StringRef grammarName) {
  emitGrammar(grammarName);
  emit(module->getRules());
  emitCustomTerminators();
  emitTerminators();
}
/// Emit the system-defined terminator.
void CGModule::emitTerminators() {
  for (auto start = terminators.terminators.begin();
       start != terminators.terminators.end(); start++) {
    os << start->first() << '\n';
    os << "  : " << start->second << "\n  ;\n\n";
  }
  emitWSAndComment();
}

void CGModule::emitGrammar(llvm::StringRef grammarNmae) {
  os << "grammar " << grammarNmae << ";\n\n";
}

/// Emit user-defined terminator.
void CGModule::emitCustomTerminators() {
  for (auto terminator : terminators.customTerminators) {
    std::string tmp = terminator.str();
    if (tmp[0] >= 'a' && tmp[0] <= 'z')
      tmp[0] -= 32;
    llvm::StringRef name(tmp);
    os << name << '\n';
    os << "  : \'" << terminator.str() << "\'\n  ;\n\n";
  }
}

/// Emit the generative formula.
void CGModule::emit(const std::vector<Rule *> &rules) {
  for (Rule *rule : rules) {
    os << rule->getName() << '\n';
    emit(rule->getGenerators());
    os << '\n';
  }
}

void CGModule::emit(const std::vector<std::vector<AntlrBase *>> &generators) {
  for (auto generator : generators) {
    if (generator == generators[0])
      os << "  : ";
    else
      os << "  | ";
    emit(generator);
  }
  os << "  ;\n";
}

/// Output the elements of the generated formula.
void CGModule::emit(const std::vector<AntlrBase *> &generator) {
  for (AntlrBase *base : generator) {
    if (base->getKind() == AntlrBase::baseKind::terminator) {
      std::string tmp = base->getName().str();
      // The terminator in antlr must be capitalized.
      if (tmp[0] >= 'a' && tmp[0] <= 'z')
        tmp[0] -= 32;
      llvm::StringRef name(tmp);
      os << name << " ";
    } else if (base->getKind() == AntlrBase::baseKind::rule) {
      os << base->getName() << " ";
    } else if (base->getKind() == AntlrBase::baseKind::pbexpression) {
      os << base->getName();
    }
  }
  os << '\n';
}

/// TODO: Supports user-defined comment whitespace.
void CGModule::emitWSAndComment() {
  os << "WS\n  : [ \\r\\n\\t] -> skip\n  ;\n\n";
  os << "Comment\n  : '#' .*? \'\\n\' ->skip\n  ;\n";
}

/// Emit tablegen file.
void CGModule::emitTd() {
  emitIncludes();
  emitOpInterfaces();
  emitDialect();
  emitOps();
}

/// output user-defined dialect to tablegen's form.
void CGModule::emitDialect() {
  Dialect *dialect = module->getDialect();
  os << "def " << dialect->getDefName() << " : Dialect {\n";
  os << "  let name = " << dialect->getName() << ";\n";
  os << "  let cppNamespace = " << dialect->getCppNamespace() << ";\n";
  os << "  let emitAccessorPrefix = " << dialect->getEmitAccessorPrefix()
     << ";\n";
  os << "}\n\n";
}

/// output user-defined ops to tablegen's form.
void CGModule::emitOps() {
  std::vector<Op *> &ops = module->getOps();
  llvm::StringRef baseDilectName = module->getDialect()->getDefName();
  size_t idx = baseDilectName.find("_");
  std::string opBaseName = baseDilectName.substr(0, idx).str();
  opBaseName += "_Op";
  os << "class " << opBaseName
     << "<string mnemonic, list<Trait> traits = []> :\n";
  os << "  Op<" << baseDilectName << ", mnemonic, traits>;\n\n";

  for (Op *op : ops) {
    os << "def " << op->getOpName() << " : " << opBaseName << "<"
       << op->getMnemonic();
    if (!op->getTraits().empty())
      os << ", " << op->getTraits();
    os << "> { \n";
    if (!op->getSummary().empty())
      os << "  let summary = " << op->getSummary() << ";\n";
    if (!op->getArguments().empty())
      os << "  let arguments = " << op->getArguments() << ";\n";
    if (!op->getResults().empty())
      os << "  let results = " << op->getResults() << ";\n";
    if (op->getHasConstantMaterializer())
      os << "  let hasCustomAssemblyFormat = 1;\n";
    if (op->getHasVerifier())
      os << "  let hasVerifier = 1;\n";
    if (!op->getDescription().empty())
      os << "  let description = " << op->getDescription() << ";\n";
    if (!op->getBuilders().empty())
      os << "  let builders = " << op->getBuilders() << ";\n";
    if (!op->getExtraClassDeclaration().empty())
      os << "  let extraClassDeclaration = " << op->getExtraClassDeclaration()
         << ";\n";
    if (op->getSkipDefaultBuilders())
      os << "  let  skipDefaultBuilders = 1;\n";
    if (!op->getRegions().empty())
      os << "  let regions = " << op->getRegions() << ";\n";
    if (op->getHasCanonicalizer())
      os << "  let hasCanonicalizer = 1;\n";
    if (!op->getAssemblyFormat().empty())
      os << "  let assemblyFormat = " << op->getAssemblyFormat() << ";\n";
    os << "}\n\n";
  }
}

void CGModule::emitOpInterfaces() {
  std::vector<Opinterface *> opInterfaces = module->getOpInterfaces();
  for (Opinterface *opInterface : opInterfaces) {
    os << "def " << opInterface->getDefName() << " : OpInterface<"
       << opInterface->getName() << ">{\n";
    if (!opInterface->getDescription().empty())
      os << "  let description = " << opInterface->getDescription() << ";\n";
    if (!opInterface->getMethods().empty())
      os << "  let methods = " << opInterface->getMethods() << ";\n";
    os << "}\n\n";
  }
}

void CGModule::emitIncludes() {
  if (!module->getOps().empty()) {
    os << "include \"mlir/IR/FunctionInterfaces.td\"\n"
       << "include \"mlir/IR/SymbolInterfaces.td\"\n"
       << "include \"mlir/Interfaces/CallInterfaces.td\"\n"
       << "include \"mlir/Interfaces/CastInterfaces.td\"\n"
       << "include \"mlir/Interfaces/SideEffectInterfaces.td\"\n";
  }
  if (!module->getOpInterfaces().empty()) {
    os << "include \"mlir/IR/OpBase.td\"\n";
  }
}
