#include "CGModule.h"
#include "Diagnostics.h"
#include "Lexer.h"
#include "Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

llvm::cl::opt<std::string> inputFileName("f", llvm::cl::desc("<input file>"));
llvm::cl::opt<std::string> outputFileName("o", llvm::cl::desc("<output file>"));

namespace {
enum Action { none, dumpAst, dumpAntlr, dumpTd, dumpAntlrAndTd };
}

llvm::cl::opt<Action> emitAction(
    "emit", llvm::cl::desc("Select the kind of output desired"),
    llvm::cl::values(clEnumValN(dumpAst, "ast", "Out put the ast")),
    llvm::cl::values(clEnumValN(dumpAntlr, "antlr", "Out put the antlr file")),
    llvm::cl::values(clEnumValN(dumpTd, "td", "put out the td file")),
    llvm::cl::values(clEnumValN(dumpAntlrAndTd, "all", "put out all file")));

/// Control generation of ast, tablegen files and antlr files.
void emit(frontendgen::Module *module, frontendgen::Terminators &terminators) {
  bool emitAst = emitAction == Action::dumpAst;
  bool emitAntlr =
      emitAction == Action::dumpAntlr || emitAction == Action::dumpAntlrAndTd;
  bool emitTd =
      emitAction == Action::dumpTd || emitAction == Action::dumpAntlrAndTd;
  if (emitAntlr) {
    if (outputFileName.empty()) {
      llvm::errs() << "if you want to emit g4 file you have to point out the "
                      "name of outputfile.\n";
      return;
    }
    if (!llvm::StringRef(outputFileName.c_str()).endswith(".g4")) {
      llvm::errs() << "outputfile name have to end with .g4.\n";
      return;
    }
    llvm::StringRef grammerName =
        llvm::StringRef(outputFileName.c_str(), outputFileName.size() - 3);
    std::error_code EC;
    llvm::sys::fs::OpenFlags openFlags = llvm::sys::fs::OpenFlags::OF_None;
    auto Out = llvm::ToolOutputFile(outputFileName.c_str(), EC, openFlags);
    frontendgen::CGModule CGmodule(module, Out.os(), terminators);
    CGmodule.emitAntlr(grammerName);
    Out.keep();
  }
  if (emitTd && module->getDialect()) {
    std::error_code EC;
    llvm::sys::fs::OpenFlags openFlags = llvm::sys::fs::OpenFlags::OF_None;
    auto Out = llvm::ToolOutputFile("Ops.td", EC, openFlags);
    frontendgen::CGModule CGmodule(module, Out.os(), terminators);
    CGmodule.emitTd();
    Out.keep();
  }
  if (emitAction == dumpAst && !module->getRules().empty()) {
    llvm::raw_fd_ostream os(-1, true);
    frontendgen::CGModule CGmodule(module, os, terminators);
    CGmodule.emitAST();
  }
  for (auto rule : module->getRules()) {
    for (auto generator : rule->getGenerators()) {
      for (auto element : generator) {
        delete element;
      }
    }
    delete rule;
  }
  for (auto* opinterface : module->getOpInterfaces())
    delete opinterface;
  delete module->getDialect();
  for (auto op : module->getOps()) {
    delete op;
  }
  delete module;
}

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file =
      llvm::MemoryBuffer::getFile(inputFileName.c_str());
  if (std::error_code bufferError = file.getError()) {
    llvm::errs() << "error read: " << bufferError.message() << '\n';
    exit(1);
  }
  llvm::SourceMgr srcMgr;
  srcMgr.AddNewSourceBuffer(std::move(*file), llvm::SMLoc());
  frontendgen::DiagnosticEngine diagnostic(srcMgr);
  frontendgen::Lexer lexer(srcMgr, diagnostic);
  frontendgen::Sema action;
  frontendgen::Terminators terminators;
  frontendgen::Parser parser(lexer, action, terminators);
  frontendgen::Module *module = parser.parser();
  emit(module, terminators);
  return 0;
}
