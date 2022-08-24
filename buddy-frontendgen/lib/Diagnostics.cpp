#include "Diagnostics.h"
#include "llvm/Support/SourceMgr.h"

using namespace frontendgen;
namespace {

/// Storage the message of the diagnostic.
const char *diagnosticText[] = {
#define DIAG(ID, Level, Msg) Msg,
#include "Diagnostics.def"
};

/// Storage the kind of the diagnostic.
llvm::SourceMgr::DiagKind diagnosticKind[] = {
#define DIAG(ID, Level, Msg) llvm::SourceMgr::DK_##Level,
#include "Diagnostics.def"
};
} // namespace

/// Get the message of the diagnostic.
const char *DiagnosticEngine::getDiagnosticText(unsigned diagID) {
  return diagnosticText[diagID];
}
/// Get the kind of the diagnostic.
llvm::SourceMgr::DiagKind DiagnosticEngine::getDiagnosticKind(unsigned DiagID) {
  return diagnosticKind[DiagID];
}
