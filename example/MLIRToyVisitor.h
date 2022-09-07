#include "ToyBaseVisitor.h"
#include "ToyLexer.h"
#include "ToyParser.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

class MLIRToyVisitor : public ToyBaseVisitor {
mlir::ModuleOp theModule;
mlir::OpBuilder builder;
std::string fileName;

public:
MLIRToyVisitor(std::string filename, mlir::MLIRContext &context)
: builder(&context), fileName(filename) {
 theModule = mlir::ModuleOp::create(builder.getUnknownLoc()); 
}

mlir::ModuleOp getModule() { return theModule; }

virtual std::any visitModule(ToyParser::ModuleContext *ctx) {
  return visitChildren(ctx);
}

virtual std::any visitExpression(ToyParser::ExpressionContext *ctx) {
  return visitChildren(ctx);
}

virtual std::any visitReturnExpr(ToyParser::ReturnExprContext *ctx) {
  return visitChildren(ctx);
}

virtual std::any visitIdentifierExpr(ToyParser::IdentifierExprContext *ctx) {
  {
  llvm::StringRef callee;
  llvm::ArrayRef<Value> arguments;
  mlir::Location location;
  builder.create<mlir::toy::GenericCallOp>(location, callee, arguments);
  }

  {
  mlir::Value input;
  mlir::Location location;
  builder.create<mlir::toy::PrintOp>(location, input);
  }

  return visitChildren(ctx);
}

virtual std::any visitTensorLiteral(ToyParser::TensorLiteralContext *ctx) {
  return visitChildren(ctx);
}

virtual std::any visitVarDecl(ToyParser::VarDeclContext *ctx) {
  {
  mlir::Type res0;
  mlir::Value input;
  mlir::Location location;
  builder.create<mlir::toy::ReshapeOp>(location, res0, input);
  }

  return visitChildren(ctx);
}

virtual std::any visitType(ToyParser::TypeContext *ctx) {
  return visitChildren(ctx);
}

virtual std::any visitFunDefine(ToyParser::FunDefineContext *ctx) {
  {
  mlir::Location location;
  builder.create<mlir::toy::ReturnOp>(location);
  }

  return visitChildren(ctx);
}

virtual std::any visitPrototype(ToyParser::PrototypeContext *ctx) {
  {
  llvm::StringRef sym_name;
  mlir::FunctionType function_type;
  mlir::Location location;
  builder.create<mlir::toy::FuncOp>(location, sym_name, function_type);
  }

  return visitChildren(ctx);
}

virtual std::any visitDeclList(ToyParser::DeclListContext *ctx) {
  return visitChildren(ctx);
}

virtual std::any visitBlock(ToyParser::BlockContext *ctx) {
  return visitChildren(ctx);
}

virtual std::any visitBlockExpr(ToyParser::BlockExprContext *ctx) {
  return visitChildren(ctx);
}

};
