#include "CGModule.h"
#include "llvm/Support/Casting.h"

using namespace frontendgen;
/// Emit the ast,currently only antlr's ast are supported.
void CGModule::emitAST() {
  for (auto i : module->getRules()) {
    llvm::outs() << "rule name: " << i->getName() << '\n';
    for (auto j : i->getGeneratorsAndOthers()) {
      llvm::outs() << "  generator: " << '\n' << "    ";
      for (auto k : j->getGenerator()) {
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
    emit(rule->getGeneratorsAndOthers());
    os << '\n';
  }
}

void CGModule::emit(const std::vector<GeneratorAndOthers*> &generatorsAndOthers) {
  for (GeneratorAndOthers* generatorAndOthers : generatorsAndOthers) {
    if (generatorAndOthers == generatorsAndOthers[0])
      os << "  : ";
    else
      os << "  | ";
    emit(generatorAndOthers->getGenerator());
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
    if (op->getArguments()) {
      DAG* dag = op->getArguments();
      auto operands = dag->getOperands();
      os << "  let arguments = (";
      os << dag->getDagOperatpr() <<" ";
      auto operandNames = dag->getOperandNames();
      for (auto start = operands.begin(); start != operands.end(); start++) {
        os << *start;
        if (operandNames.find(*start) != operandNames.end())
          os << ":$" << operandNames[*start];
        if (start + 1 != operands.end()) 
          os << ", ";
      }
        os << ")\n";
    }
    if (op->getResults()) {
      DAG* dag = op->getResults();
      auto operands = dag->getOperands();
      os << "  let results = (";
      os << dag->getDagOperatpr() <<" ";
      auto operandNames = dag->getOperandNames();
      for (auto start = operands.begin(); start != operands.end(); start++) {
        os << *start;
        if (operandNames.find(*start) != operandNames.end())
          os << ":$" << operandNames[*start];
        if (start + 1 != operands.end()) 
          os << ", ";
      }
        os << ")\n";
    }
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

void CGModule::emitMLIRVisitor(llvm::StringRef grammarName) {
  emitIncludes(grammarName);
  emitClass(grammarName);
}

void CGModule::emitIncludes(llvm::StringRef grammarName) {
  os << "#include \"" << grammarName << "BaseVisitor.h\"\n";
  os << "#include \"" << grammarName <<  "Lexer.h\"\n";
  os << "#include \"" << grammarName <<"Parser.h\"\n";
  os << "#include \"mlir/IR/Attributes.h\"\n";
  os << "#include \"mlir/IR/Builders.h\"\n";
  os << "#include \"mlir/IR/BuiltinOps.h\"\n";
  os << "#include \"mlir/IR/BuiltinTypes.h\"\n";
  os << "#include \"mlir/IR/MLIRContext.h\"\n";
  os << "#include \"mlir/IR/Verifier.h\"\n";
  os << "#include \"llvm/ADT/STLExtras.h\"\n";
  os << "#include \"llvm/ADT/ScopedHashTable.h\"\n";
  os << "#include \"llvm/ADT/StringRef.h\"\n";
  os << "#include \"llvm/Support/raw_ostream.h\"\n";
  os << "\n";
}

void CGModule::emitClass(llvm::StringRef gramarName) {
  os << "class MLIRToy" << gramarName << "Visitor : public " <<  gramarName << "BaseVisitor {\n";

  os << "public:\n";
  os << "MLIR" << gramarName << "Visitor(std::string filename, mlir::MLIRContext &context)\n"
     << ": builder(&context), fileName(filename) " << "{\n theModule = mlir::ModuleOp::create(builder.getUnknownLoc()); \n}\n\n";
  os << "mlir::ModuleOp getModule() { return theModule; }\n\n";

  os << "private:\n";
  os << "mlir::ModuleOp theModule;\n";
  os << "mlir::OpBuilder builder;\n";
  os << "std::string fileName;\n\n";
  auto rules = module->getRules();
  for (auto rule : rules) {
    emitRuleVisitor(gramarName, rule);
  }
  os << "}\n";
}


void CGModule::emitRuleVisitor(llvm::StringRef grammarName, Rule* rule) {
  std::string ruleName = rule->getName().str();
  ruleName[0] = ruleName[0] - 32;
  os << "virtual std::any visit" << ruleName;
  os << "(" << grammarName << "Parser::" << ruleName << "Context *ctx) {\n";
  emitBuilder(rule);
  os << "  return visitChildren(ctx);\n";
  os << "}\n\n";
}


void CGModule::emitBuilder(Rule* rule) {
  for (GeneratorAndOthers* generatorAndOthers : rule->getGeneratorsAndOthers()) {
    if (generatorAndOthers->getOpBulderIdx() != -1) {
      llvm::StringRef opName = generatorAndOthers->getBuilderOpName();
      
    }
  }
}