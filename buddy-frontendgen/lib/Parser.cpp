#include "Parser.h"
#include "AST.h"
#include "Lexer.h"
#include "Sema.h"
#include "unistd.h"
#include "llvm/Support/raw_ostream.h"
using namespace frontendgen;

void Parser::advance() { lexer.next(token); }

void Parser::lookToken() {
  while (token.getKind() != tokenKinds::eof) {
    llvm::outs() << token.getContent() << '\n';
    llvm::outs() << "token type:" << token.getTokenName() << '\n';
    advance();
  }
}

/// If current token's kind is expected kind, get next token.
/// If not, an error is reported.
bool Parser::consume(tokenKinds expectTok) {
  if (token.is(expectTok)) {
    advance();
    return true;
  }
  lexer.getDiagnostic().report(token.getLocation(),
                               DiagnosticEngine::err_expected,
                               tokenNameMap[expectTok], token.getTokenName());
  return false;
}

/// If current token's kind is expected kind, get next token.
/// If not, do nothing.
bool Parser::consumeNoAdvance(tokenKinds expectTok) {
  if (token.is(expectTok))
    return true;
  lexer.getDiagnostic().report(token.getLocation(),
                               DiagnosticEngine::err_expected,
                               tokenNameMap[expectTok], token.getTokenName());
  return false;
}

/// Parser the file, and return a Module, it store all information
/// to generate code.
Module *Parser::parser() {
  Module *module = new Module();
  compilEngine(module);
  return module;
}

/// Parse keyword op, dialect and rule.
void Parser::compilEngine(Module *module) {
  // rules store all rule ast.
  std::vector<Rule *> rules;
  // A file can only store one dialect.
  Dialect *dialect = nullptr;
  // ops store all op.
  std::vector<Op *> ops;
  std::vector<Opinterface *> opInterfaces;
  while (token.getKind() != tokenKinds::eof) {
    if (token.is(tokenKinds::kw_rule)) {
      advance();
      if (!consumeNoAdvance(tokenKinds::identifier))
        return;
      Rule *rule =
          new Rule(token.getContent(), token.getLocation(), AntlrBase::rule);
      advance();
      parserRules(rule);
      rules.push_back(rule);
      consume(tokenKinds::semi);
    } else if (token.is(tokenKinds::kw_dialect)) {
      advance();
      if (!consumeNoAdvance(tokenKinds::identifier))
        return;
      llvm::StringRef defName = token.getContent();
      advance();
      parserDialect(dialect, defName);
    } else if (token.is(tokenKinds::kw_op)) {
      advance();
      if (!parserOp(ops, token.getContent())) {
        action.actOnModule(module, rules, dialect, ops, opInterfaces);
        return;
      }
    } else if (token.is(tokenKinds::kw_opinterface)) {
      advance();
      if (!parserOpinterface(opInterfaces))
        action.actOnModule(module, rules, dialect, ops, opInterfaces);
    } else {
      lexer.getDiagnostic().report(
          token.getLocation(), DiagnosticEngine::err_expected,
          "keyword rule, dialect or op", token.getTokenName());
      action.actOnModule(module, rules, dialect, ops, opInterfaces);
      return;
    }
  }
  action.actOnModule(module, rules, dialect, ops, opInterfaces);
}

/// Parser the rule and fill nodes of rule ast.
void Parser::parserRules(Rule *rule) {
  if (!consumeNoAdvance(tokenKinds::colon))
    return;
  // A rule contains many generative.
  std::vector<std::vector<AntlrBase *>> generators;
  while (token.getKind() != tokenKinds::semi &&
         token.getKind() == tokenKinds::colon) {
    advance();
    std::vector<AntlrBase *> generator;
    parserGenerator(generator);
    generators.push_back(generator);
    if (!token.is(tokenKinds::colon) && !token.is(tokenKinds::semi)) {
      lexer.getDiagnostic().report(token.getLocation(),
                                   DiagnosticEngine::err_expected,
                                   "colon or semi", token.getTokenName());
      return;
    }
  }
  // Fill the rule ast.
  action.actOnRule(rule, generators);
}

/// Parser a generator and fill a node in generator.
void Parser::parserGenerator(std::vector<AntlrBase *> &generator) {
  while (token.is(tokenKinds::identifier) || token.is(tokenKinds::apostrophe) ||
         token.is(tokenKinds::plus) || token.is(tokenKinds::asterisk) ||
         token.is(tokenKinds::parentheseOpen) ||
         token.is(tokenKinds::parentheseClose) ||
         token.is(tokenKinds::questionMark)) {
    if (token.is(tokenKinds::identifier))
      parserIdentifier(generator);
    else if (token.is(tokenKinds::apostrophe))
      parserTerminator(generator);
    else
      parserPBExpression(generator);
  }
}

/// Check if the identifier is a terminator.
AntlrBase::baseKind Parser::getAntlrBaseKind(llvm::StringRef name) {
  if (terminators.isTerminator(name))
    return AntlrBase::baseKind::terminator;
  return AntlrBase::baseKind::rule;
}

/// processing the identifier, get the identifier's kind which stores
/// in the ast.
void Parser::parserIdentifier(std::vector<AntlrBase *> &generator) {
  AntlrBase::baseKind baseKind = getAntlrBaseKind(token.getContent());
  AntlrBase *r = nullptr;
  if (baseKind == AntlrBase::baseKind::rule)
    r = new Rule(token.getContent(), token.getLocation(), baseKind);
  else if (baseKind == AntlrBase::AntlrBase::terminator)
    r = new Terminator(token.getContent(), token.getLocation(), baseKind);
  generator.push_back(r);
  advance();
}

/// We support user-defined terminator.For example, we can write a 'terminator'
/// in a rule.
void Parser::parserTerminator(std::vector<AntlrBase *> &generator) {
  advance();
  AntlrBase *terminator = new Terminator(
      token.getContent(), token.getLocation(), AntlrBase::terminator);
  generator.push_back(terminator);
  terminators.addCustomTerminators(token.getContent());
  advance();
  consume(tokenKinds::apostrophe);
}

void Parser::parserPBExpression(std::vector<AntlrBase *> &generator) {
  AntlrBase *r = new Terminator(token.getContent(), token.getLocation(),
                                AntlrBase::pbexpression);
  generator.push_back(r);
  advance();
}
/// Parser dialect keyword and fill all information in the dialect.
void Parser::parserDialect(Dialect *&dialect, llvm::StringRef defName) {
  dialect = new Dialect();
  llvm::StringRef name;
  llvm::StringRef emitAccessorPrefix;
  llvm::StringRef cppNamespace;
  bool hasConstantMaterializer;
  while (token.is(tokenKinds::colon)) {
    advance();
    if (token.getContent().str() == "name") {
      advance();
      consumeNoAdvance(tokenKinds::equal);
      name = lexer.getMarkContent("\"", "\"");
      advance();
    } else if (token.getContent().str() == "emitAccessorPrefix") {
      advance();
      if (!consume(tokenKinds::equal))
        return;
      emitAccessorPrefix = token.getContent();
      advance();
    } else if (token.getContent().str() == "cppNamespace") {
      advance();
      consumeNoAdvance(tokenKinds::equal);
      cppNamespace = lexer.getMarkContent("\"", "\"");
      advance();
    } else if (token.getContent().str() == "hasConstantMaterializer") {
      advance();
      if (!consume(tokenKinds::equal))
        return;
      if (token.getContent().str() == "1")
        hasConstantMaterializer = true;
      advance();
    }
  }
  action.actOnDialect(dialect, defName, name, emitAccessorPrefix, cppNamespace);
  advance();
}

/// Parser op keyword and fill all information in the ops.
bool Parser::parserOp(std::vector<Op *> &ops, llvm::StringRef opName) {
  llvm::SMLoc loc = token.getLocation();
  llvm::StringRef mnemonic;
  llvm::StringRef traits;
  llvm::StringRef summary;
  llvm::StringRef description;
  llvm::StringRef arguments;
  llvm::StringRef results;
  bool hasCustomAssemblyFormat = false;
  llvm::StringRef builders;
  bool hasVerifier = false;
  llvm::StringRef assemblyFormat;
  llvm::StringRef regions;
  llvm::StringRef extraClassDeclaration;
  bool skipDefaultBuilders = false;
  bool hasCanonicalizer = false;
  advance();
  while (token.is(tokenKinds::colon)) {
    advance();
    if (token.getContent() == "mnemonic") {
      advance();
      if (!consumeNoAdvance(tokenKinds::equal))
        return false;
      mnemonic = lexer.getMarkContent("\"", "\"");
      advance();
    } else if (token.getContent() == "summary") {
      advance();
      if (!consumeNoAdvance(tokenKinds::equal))
        return false;
      summary = lexer.getMarkContent("\"", "\"");
      advance();
    } else if (token.getContent() == "traits") {
      advance();
      if (!consumeNoAdvance(tokenKinds::equal))
        return false;
      traits = lexer.getMarkContent("[", "]");
      advance();
    } else if (token.getContent() == "description") {
      advance();
      if (!consumeNoAdvance(tokenKinds::equal))
        return false;
      description = lexer.getMarkContent("[", "]");
      advance();
    } else if (token.getContent() == "arguments") {
      advance();
      if (!consumeNoAdvance(tokenKinds::equal))
        return false;
      arguments = lexer.getMarkContent("(", ")");
      advance();
    } else if (token.getContent() == "results") {
      advance();
      if (!consumeNoAdvance(tokenKinds::equal))
        return false;
      results = lexer.getMarkContent("(", ")");
      advance();
    } else if (token.getContent() == "hasCustomAssemblyFormat") {
      advance();
      if (!consume(tokenKinds::equal))
        return false;
      if (consumeNoAdvance(tokenKinds::number)) {
        if (token.getContent() == "1")
          hasCustomAssemblyFormat = true;
      }
      advance();
    } else if (token.getContent() == "builders") {
      advance();
      if (!consumeNoAdvance(tokenKinds::equal))
        return false;
      builders = lexer.getMarkContent("[", "]");
      advance();
    } else if (token.getContent() == "hasVerifier") {
      advance();
      if (!consume(tokenKinds::equal))
        return false;
      if (!consumeNoAdvance(tokenKinds::number)) {
        if (token.getContent() == "1")
          hasVerifier = true;
      }
      advance();
    } else if (token.getContent() == "assemblyFormat") {
      advance();
      if (!consumeNoAdvance(tokenKinds::equal))
        return false;
      assemblyFormat = lexer.getMarkContent("\"[", "\"]");
      advance();
    } else if (token.getContent() == "regions") {
      advance();
      if (!consumeNoAdvance(tokenKinds::equal))
        return false;
      regions = lexer.getMarkContent("(", ")");
      advance();
    } else if (token.getContent() == "extraClassDeclaration") {
      advance();
      if (!consumeNoAdvance(tokenKinds::equal))
        return false;
      extraClassDeclaration = lexer.getMarkContent("[", "]");
      advance();
    } else if (token.getContent() == "skipDefaultBuilders") {
      advance();
      if (!consume(tokenKinds::equal))
        return false;
      if (token.getKind() == tokenKinds::number) {
        if (token.getContent() == "1")
          skipDefaultBuilders = true;
      }
      advance();
    } else if (token.getContent() == "hasCanonicalizer") {
      advance();
      if (!consume(tokenKinds::equal))
        return false;
      if (token.getKind() == tokenKinds::number) {
        if (token.getContent() == "1")
          hasCanonicalizer = true;
      }
      advance();
    } else {
      lexer.getDiagnostic().report(token.getLocation(),
                                   DiagnosticEngine::err_not_supported_element,
                                   token.getContent());
      return false;
    }
  }
  if (mnemonic.empty()) {
    lexer.getDiagnostic().report(loc,
                                 DiagnosticEngine::diagKind::err_no_mnemonic);
    advance();
    return false;
  }
  if (!consume(tokenKinds::semi)) {
    llvm::outs() << token.getContent();
    return false;
  }
  action.actOnOps(ops, opName, mnemonic, traits, summary, description,
                  arguments, results, hasCustomAssemblyFormat, builders,
                  hasVerifier, assemblyFormat, regions, extraClassDeclaration,
                  skipDefaultBuilders, hasCanonicalizer);
  return true;
}

bool Parser::parserOpinterface(std::vector<Opinterface *> &opInterfaces) {
  llvm::StringRef defName;
  llvm::StringRef name;
  llvm::StringRef description;
  llvm::StringRef methods;
  if (!consumeNoAdvance(tokenKinds::identifier))
    return false;
  defName = token.getContent();
  llvm::SMLoc location = token.getLocation();
  advance();
  while (token.is(tokenKinds::colon)) {
    advance();
    if (token.getContent() == "name") {
      advance();
      if (!consumeNoAdvance(tokenKinds::equal))
        return false;
      name = lexer.getMarkContent("\"", "\"");
      advance();
    } else if (token.getContent() == "methods") {
      advance();
      if (!consumeNoAdvance(tokenKinds::equal))
        return false;
      methods = lexer.getMarkContent("[", "]");
      advance();
    } else if (token.getContent() == "description") {
      advance();
      if (!consumeNoAdvance(tokenKinds::equal))
        return false;
      description = lexer.getMarkContent("[", "]");
      advance();
    }
  }
  if (name.empty()) {
    lexer.getDiagnostic().report(location, DiagnosticEngine::err_no_name);
    return false;
  }
  action.actOnOpInterfaces(opInterfaces, defName, name, methods, description);
  if (!consume(semi))
    return false;
  return true;
}