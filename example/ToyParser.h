
// Generated from Toy.g4 by ANTLR 4.10.1

#pragma once


#include "antlr4-runtime.h"




class  ToyParser : public antlr4::Parser {
public:
  enum {
    Return = 1, ParentheseOpen = 2, SbracketOpen = 3, Identifier = 4, ParentheseClose = 5, 
    SbracketClose = 6, Var = 7, Add = 8, Sub = 9, Number = 10, Comma = 11, 
    Semi = 12, BracketOpen = 13, Def = 14, BracketClose = 15, AngleBracketOpen = 16, 
    Equal = 17, AngleBracketClose = 18, WS = 19, Comment = 20
  };

  enum {
    RuleModule = 0, RuleExpression = 1, RuleReturnExpr = 2, RuleIdentifierExpr = 3, 
    RuleTensorLiteral = 4, RuleVarDecl = 5, RuleType = 6, RuleFunDefine = 7, 
    RulePrototype = 8, RuleDeclList = 9, RuleBlock = 10, RuleBlockExpr = 11
  };

  explicit ToyParser(antlr4::TokenStream *input);

  ToyParser(antlr4::TokenStream *input, const antlr4::atn::ParserATNSimulatorOptions &options);

  ~ToyParser() override;

  std::string getGrammarFileName() const override;

  const antlr4::atn::ATN& getATN() const override;

  const std::vector<std::string>& getRuleNames() const override;

  const antlr4::dfa::Vocabulary& getVocabulary() const override;

  antlr4::atn::SerializedATNView getSerializedATN() const override;


  class ModuleContext;
  class ExpressionContext;
  class ReturnExprContext;
  class IdentifierExprContext;
  class TensorLiteralContext;
  class VarDeclContext;
  class TypeContext;
  class FunDefineContext;
  class PrototypeContext;
  class DeclListContext;
  class BlockContext;
  class BlockExprContext; 

  class  ModuleContext : public antlr4::ParserRuleContext {
  public:
    ModuleContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<FunDefineContext *> funDefine();
    FunDefineContext* funDefine(size_t i);


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ModuleContext* module();

  class  ExpressionContext : public antlr4::ParserRuleContext {
  public:
    ExpressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Number();
    TensorLiteralContext *tensorLiteral();
    IdentifierExprContext *identifierExpr();


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ExpressionContext* expression();

  class  ReturnExprContext : public antlr4::ParserRuleContext {
  public:
    ReturnExprContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Return();
    ExpressionContext *expression();


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ReturnExprContext* returnExpr();

  class  IdentifierExprContext : public antlr4::ParserRuleContext {
  public:
    IdentifierExprContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *ParentheseOpen();
    antlr4::tree::TerminalNode *ParentheseClose();
    std::vector<ExpressionContext *> expression();
    ExpressionContext* expression(size_t i);
    std::vector<antlr4::tree::TerminalNode *> Comma();
    antlr4::tree::TerminalNode* Comma(size_t i);


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  IdentifierExprContext* identifierExpr();

  class  TensorLiteralContext : public antlr4::ParserRuleContext {
  public:
    TensorLiteralContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *SbracketOpen();
    antlr4::tree::TerminalNode *SbracketClose();
    std::vector<TensorLiteralContext *> tensorLiteral();
    TensorLiteralContext* tensorLiteral(size_t i);
    std::vector<antlr4::tree::TerminalNode *> Comma();
    antlr4::tree::TerminalNode* Comma(size_t i);
    antlr4::tree::TerminalNode *Number();


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  TensorLiteralContext* tensorLiteral();

  class  VarDeclContext : public antlr4::ParserRuleContext {
  public:
    VarDeclContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Var();
    antlr4::tree::TerminalNode *Identifier();
    TypeContext *type();
    antlr4::tree::TerminalNode *Equal();
    ExpressionContext *expression();


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  VarDeclContext* varDecl();

  class  TypeContext : public antlr4::ParserRuleContext {
  public:
    TypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *AngleBracketOpen();
    std::vector<antlr4::tree::TerminalNode *> Number();
    antlr4::tree::TerminalNode* Number(size_t i);
    antlr4::tree::TerminalNode *AngleBracketClose();
    std::vector<antlr4::tree::TerminalNode *> Comma();
    antlr4::tree::TerminalNode* Comma(size_t i);


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  TypeContext* type();

  class  FunDefineContext : public antlr4::ParserRuleContext {
  public:
    FunDefineContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PrototypeContext *prototype();
    BlockContext *block();


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  FunDefineContext* funDefine();

  class  PrototypeContext : public antlr4::ParserRuleContext {
  public:
    PrototypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Def();
    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *ParentheseOpen();
    antlr4::tree::TerminalNode *ParentheseClose();
    DeclListContext *declList();


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  PrototypeContext* prototype();

  class  DeclListContext : public antlr4::ParserRuleContext {
  public:
    DeclListContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *Comma();
    DeclListContext *declList();


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  DeclListContext* declList();

  class  BlockContext : public antlr4::ParserRuleContext {
  public:
    BlockContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *BracketOpen();
    antlr4::tree::TerminalNode *BracketClose();
    std::vector<BlockExprContext *> blockExpr();
    BlockExprContext* blockExpr(size_t i);
    std::vector<antlr4::tree::TerminalNode *> Semi();
    antlr4::tree::TerminalNode* Semi(size_t i);


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  BlockContext* block();

  class  BlockExprContext : public antlr4::ParserRuleContext {
  public:
    BlockExprContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    VarDeclContext *varDecl();
    ReturnExprContext *returnExpr();
    ExpressionContext *expression();


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  BlockExprContext* blockExpr();


  // By default the static state used to implement the parser is lazily initialized during the first
  // call to the constructor. You can call this function if you wish to initialize the static state
  // ahead of time.
  static void initialize();

private:
};

