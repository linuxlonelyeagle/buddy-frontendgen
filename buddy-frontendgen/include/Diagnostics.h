#ifndef INCLUDE_DIAGNOSTIC_H
#define INCLUDE_DIAGNOSTIC_H
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"

/// When there is an error in the user's code, we can diagnose the error through
/// the class.
namespace frontendgen {
class DiagnosticEngine {
  llvm::SourceMgr &SrcMgr;
  static const char *getDiagnosticText(unsigned diagID);
  llvm::SourceMgr::DiagKind getDiagnosticKind(unsigned diagID);
  bool hasReport = false;

public:
  enum diagKind {
#define DIAG(ID, Level, Msg) ID,
#include "Diagnostics.def"
  };
  DiagnosticEngine(llvm::SourceMgr &SrcMgr) : SrcMgr(SrcMgr) {}

  template <typename... Args>
  void report(llvm::SMLoc loc, unsigned diagID, Args &&...arguments) {
    if (!hasReport) {
      std::string Msg = llvm::formatv(getDiagnosticText(diagID),
                                      std::forward<Args>(arguments)...)
                            .str();
      SrcMgr.PrintMessage(loc, getDiagnosticKind(diagID), Msg);
      hasReport = true;
    }
  }
};

} // namespace frontendgen

#endif