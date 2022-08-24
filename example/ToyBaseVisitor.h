
// Generated from Toy.g4 by ANTLR 4.10.1

#pragma once


#include "antlr4-runtime.h"
#include "ToyVisitor.h"


/**
 * This class provides an empty implementation of ToyVisitor, which can be
 * extended to create a visitor which only needs to handle a subset of the available methods.
 */
class  ToyBaseVisitor : public ToyVisitor {
public:

  virtual std::any visitModule(ToyParser::ModuleContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitExpression(ToyParser::ExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitReturnExpr(ToyParser::ReturnExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitIdentifierExpr(ToyParser::IdentifierExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTensorLiteral(ToyParser::TensorLiteralContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitVarDecl(ToyParser::VarDeclContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitType(ToyParser::TypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFunDefine(ToyParser::FunDefineContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPrototype(ToyParser::PrototypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDeclList(ToyParser::DeclListContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBlock(ToyParser::BlockContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBlockExpr(ToyParser::BlockExprContext *ctx) override {
    return visitChildren(ctx);
  }


};

