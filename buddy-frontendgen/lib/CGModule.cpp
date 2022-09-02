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

void CGModule::emit(
    const std::vector<GeneratorAndOthers *> &generatorsAndOthers) {
  for (GeneratorAndOthers *generatorAndOthers : generatorsAndOthers) {
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
      DAG *dag = op->getArguments();
      auto operands = dag->getOperands();
      os << "  let arguments = (";
      os << dag->getDagOperater() << " ";
      auto operandNames = dag->getOperandNames();
      int number = 0;
      for (auto start = operands.begin(); start != operands.end(); start++) {
        os << *start;
        if (!operandNames[number].empty())
          os << ":$" << operandNames[number];
        if (start + 1 != operands.end())
          os << ", ";
        number++;
      }
      os << ");\n";
    }
    if (op->getResults()) {
      DAG *dag = op->getResults();
      auto operands = dag->getOperands();
      os << "  let results = (";
      os << dag->getDagOperater() << " ";
      auto operandNames = dag->getOperandNames();
      int number = 0;
      for (auto start = operands.begin(); start != operands.end(); start++) {
        os << *start;
        if (!operandNames[number].empty())
          os << ":$" << operandNames[number];
        if (start + 1 != operands.end())
          os << ", ";
        number++;
      }
      os << ");\n";
    }
    if (op->getHasConstantMaterializer())
      os << "  let hasCustomAssemblyFormat = 1;\n";
    if (op->getHasVerifier())
      os << "  let hasVerifier = 1;\n";
    if (!op->getDescription().empty())
      os << "  let description = " << op->getDescription() << ";\n";
    if (!op->getBuilders().empty()) {
      os << "  let builders = [\n";
      std::vector<Builder *> builders = op->getBuilders();
      for (auto start = builders.begin(); start != builders.end(); start++) {
        DAG *dag = (*start)->getDag();
        os << "    OpBuilder<(" << dag->getDagOperater() << " ";
        int number = 0;
        auto operandNames = dag->getOperandNames();
        for (auto start = dag->getOperands().begin();
             start != dag->getOperands().end(); start++) {
          if (!dag->findValue(*start).empty()) {
            os << "CArg<" << *start << "," << dag->findValue(*start) << ">";
          } else
            os << *start;
          if (!operandNames[number].empty())
            os << ":$" << operandNames[number];
          if (start + 1 != dag->getOperands().end())
            os << ",";
          number++;
        }
        os << ")>";
        if (start + 1 != builders.end())
          os << ",\n";
        else
          os << "\n";
      }
      os << "  ];\n";
    }
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
  os << "#include \"" << grammarName << "Lexer.h\"\n";
  os << "#include \"" << grammarName << "Parser.h\"\n";
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
  os << "class MLIRToy" << gramarName << "Visitor : public " << gramarName
     << "BaseVisitor {\n";

  os << "public:\n";
  os << "MLIR" << gramarName
     << "Visitor(std::string filename, mlir::MLIRContext &context)\n"
     << ": builder(&context), fileName(filename) "
     << "{\n theModule = mlir::ModuleOp::create(builder.getUnknownLoc()); "
        "\n}\n\n";
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

void CGModule::emitRuleVisitor(llvm::StringRef grammarName, Rule *rule) {
  std::string ruleName = rule->getName().str();
  ruleName[0] = ruleName[0] - 32;
  os << "virtual std::any visit" << ruleName;
  os << "(" << grammarName << "Parser::" << ruleName << "Context *ctx) {\n";
  emitBuilders(rule);
  os << "  return visitChildren(ctx);\n";
  os << "}\n\n";
}

void CGModule::emitBuilders(Rule *rule) {
  for (GeneratorAndOthers *generatorAndOthers :
       rule->getGeneratorsAndOthers()) {
    llvm::SmallVector<llvm::StringRef> builderOpNames =
        generatorAndOthers->getBuilderNames();
    llvm::SmallVector<int> indexs = generatorAndOthers->getbulderIdxs();
    int size = builderOpNames.size();
    for (auto i = 0; i < size; i ++) {
      llvm::outs() << "das " <<  builderOpNames[i] << " " << indexs[i] << "\n";
    }
    for (int start = 0; start < size; start++)
      emitBuilder(builderOpNames[start], indexs[start]);
  }
}

void CGModule::emitBuilder(llvm::StringRef builderOp, int index) {
  Op *op = findOp(builderOp);
  if (op == nullptr)
    return;
  emitOp(op, index);
}

Op *CGModule::findOp(llvm::StringRef opName) {
  for (Op *op : module->getOps()) {
    if (op->getOpName() == opName)
      return op;
  }
  return nullptr;
}

void CGModule::emitOp(Op *op, int index) {
  if (index == 0) {
    DAG *arguments = op->getArguments();
    DAG *result = op->getResults();
    llvm::SmallVector<llvm::StringRef> argOperands;
    llvm::SmallVector<llvm::StringRef> argOperandNames;
    llvm::SmallVector<llvm::StringRef> resOperands;
    llvm::SmallVector<llvm::StringRef> resOperandNames;
    if (arguments) {
      argOperands = arguments->getOperands();
      argOperandNames = arguments->getOperandNames();
    }
    if (result) {
      resOperands = result->getOperands();
      resOperandNames = result->getOperandNames();
    }

    llvm::SmallVector<llvm::StringRef> opArgments;
    for (size_t index = 0; index < resOperands.size(); index++) {
      if (resOperands[index] == "F64Tensor" || resOperands[index] == "StaticShapeTensorOf<[F64]>") {
        if (resOperands[index] == "F64Tensor" || resOperands[index] == "StaticShapeTensorOf<[F64]>" )
          os << "  mlir::Type ";
        if (!resOperandNames[index].empty()) {
          os << resOperandNames[index] << ";\n";
          opArgments.push_back(resOperandNames[index]);
        } else {
          os << "res" << index << ";\n";
          llvm::StringRef arg("res" + std::to_string(index));
          opArgments.push_back(arg);
        }
      } else {
        llvm::errs() << resOperands[index] << "in" << op->getOpName() << "is not supported.\n";
        return;
      }
    }

    for (size_t index = 0; index < argOperands.size(); index++) {
      if (argOperands[index] == "F64ElementsAttr" ||
          argOperands[index] == "F64Tensor" ||
          argOperands[index] == "TypeAttrOf<FunctionType>" ||
          argOperands[index] == "SymbolNameAttr") {
        if (argOperands[index] == "F64Tensor" || argOperands[index] == "F64ElementsAttr")
          os << "  mlir::Value ";
        else if (argOperands[index] == "TypeAttrOf<FunctionType>")
          os << "  mlir::FunctionType ";
        else if (argOperands[index] == "SymbolNameAttr")
          os << "  llvm::StringRef ";
        if (!argOperandNames[index].empty()) {
          os << argOperandNames[index] << ";\n";
          opArgments.push_back(argOperandNames[index]);
        } else {
          os << "arg" << index << ";\n";
          llvm::StringRef arg("arg" + std::to_string(index));
          opArgments.push_back(arg);
        }
      } else {
        llvm::errs() << argOperands[index] << "in" << op->getOpName() << "is not supported.\n";
        return;
      }
    }

    llvm::StringRef cppNameSpace(
        module->getDialect()->getCppNamespace().data() + 1,
        module->getDialect()->getCppNamespace().size() - 2);
    os << "  "
       << "builder.create<" << cppNameSpace << "::" << op->getOpName()
       << ">(location";
    if (opArgments.size())
      os << ", ";
    for (size_t index = 0; index < opArgments.size(); index++) {
      os << opArgments[index];
      if (index + 1 != opArgments.size())
        os << ", ";
    }
    os << ");\n\n";
  } else if (index > 0) {
    index--;
    llvm::SmallVector<llvm::StringRef, 4> operands =
        op->getBuilders()[index]->getDag()->getOperands();
    llvm::SmallVector<llvm::StringRef, 4> operandNames =
        op->getBuilders()[index]->getDag()->getOperandNames();
    llvm::SmallVector<llvm::StringRef> opArguments;
    for (size_t index = 0; index < operands.size(); index++) {
      if (!findType(operands[index]).empty())
        os << "  " << findType(operands[index]);
      else
        os << "  " << operands[index];   // 如果在typemap中没有找到，那就默认输出tablegen里面的类型
      if (!operandNames[index].empty()) {
        os << " " << operandNames[index] << ";\n";
        opArguments.push_back(operandNames[index]);
      } else {
        os << " arg" << index << ";\n";
        opArguments.push_back("arg" + std::to_string(index));
      }
    }
    llvm::StringRef cppNameSpace(
        module->getDialect()->getCppNamespace().data() + 1,
        module->getDialect()->getCppNamespace().size() - 2);
    os << "  "
       << "builder.create<" << cppNameSpace << "::" << op->getOpName()
       << ">(location";
    if (!operandNames.empty()) {
      os << ", ";
      for (size_t index = 0; index < opArguments.size(); index++) {
        os << opArguments[index];
        if (index + 1 != opArguments.size())
          os << ", ";
      }
    }
    os << ");\n\n";
  }
}

void CGModule::initTypeMap() {
#define TYPEMAP(key, map) typeMap.insert(std::pair(key, map));
#include "TypeMap.def"
}

void CGModule::lookTypeMap() {
  for (auto start = typeMap.begin(); start != typeMap.end(); ++start) {
    llvm::outs() << start->first() << " " << start->second << "\n";
  }
}

llvm::StringRef CGModule::findType(llvm::StringRef key) {
  if (typeMap.find(key) == typeMap.end())
    return llvm::StringRef();
  return typeMap[key];
}