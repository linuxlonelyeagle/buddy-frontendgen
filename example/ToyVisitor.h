
// Generated from Toy.g4 by ANTLR 4.10.1

#pragma once


#include "antlr4-runtime.h"
#include "ToyParser.h"



/**
 * This class defines an abstract visitor for a parse tree
 * produced by ToyParser.
 */
class  ToyVisitor : public antlr4::tree::AbstractParseTreeVisitor {
public:

  /**
   * Visit parse trees produced by ToyParser.
   */
    virtual std::any visitModule(ToyParser::ModuleContext *context) = 0;

    virtual std::any visitExpression(ToyParser::ExpressionContext *context) = 0;

    virtual std::any visitReturnExpr(ToyParser::ReturnExprContext *context) = 0;

    virtual std::any visitIdentifierExpr(ToyParser::IdentifierExprContext *context) = 0;

    virtual std::any visitTensorLiteral(ToyParser::TensorLiteralContext *context) = 0;

    virtual std::any visitVarDecl(ToyParser::VarDeclContext *context) = 0;

    virtual std::any visitType(ToyParser::TypeContext *context) = 0;

    virtual std::any visitFunDefine(ToyParser::FunDefineContext *context) = 0;

    virtual std::any visitPrototype(ToyParser::PrototypeContext *context) = 0;

    virtual std::any visitDeclList(ToyParser::DeclListContext *context) = 0;

    virtual std::any visitBlock(ToyParser::BlockContext *context) = 0;

    virtual std::any visitBlockExpr(ToyParser::BlockExprContext *context) = 0;


};

