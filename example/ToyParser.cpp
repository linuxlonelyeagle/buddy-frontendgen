
// Generated from Toy.g4 by ANTLR 4.10.1


#include "ToyVisitor.h"

#include "ToyParser.h"


using namespace antlrcpp;

using namespace antlr4;

namespace {

struct ToyParserStaticData final {
  ToyParserStaticData(std::vector<std::string> ruleNames,
                        std::vector<std::string> literalNames,
                        std::vector<std::string> symbolicNames)
      : ruleNames(std::move(ruleNames)), literalNames(std::move(literalNames)),
        symbolicNames(std::move(symbolicNames)),
        vocabulary(this->literalNames, this->symbolicNames) {}

  ToyParserStaticData(const ToyParserStaticData&) = delete;
  ToyParserStaticData(ToyParserStaticData&&) = delete;
  ToyParserStaticData& operator=(const ToyParserStaticData&) = delete;
  ToyParserStaticData& operator=(ToyParserStaticData&&) = delete;

  std::vector<antlr4::dfa::DFA> decisionToDFA;
  antlr4::atn::PredictionContextCache sharedContextCache;
  const std::vector<std::string> ruleNames;
  const std::vector<std::string> literalNames;
  const std::vector<std::string> symbolicNames;
  const antlr4::dfa::Vocabulary vocabulary;
  antlr4::atn::SerializedATNView serializedATN;
  std::unique_ptr<antlr4::atn::ATN> atn;
};

std::once_flag toyParserOnceFlag;
ToyParserStaticData *toyParserStaticData = nullptr;

void toyParserInitialize() {
  assert(toyParserStaticData == nullptr);
  auto staticData = std::make_unique<ToyParserStaticData>(
    std::vector<std::string>{
      "module", "expression", "returnExpr", "identifierExpr", "tensorLiteral", 
      "varDecl", "type", "funDefine", "prototype", "declList", "block", 
      "blockExpr"
    },
    std::vector<std::string>{
      "", "'return'", "')'", "'['", "", "'('", "']'", "'var'", "'add'", 
      "'sub'", "", "','", "';'", "'{'", "'def'", "'}'", "'<'", "'='", "'>'"
    },
    std::vector<std::string>{
      "", "Return", "ParentheseOpen", "SbracketOpen", "Identifier", "ParentheseClose", 
      "SbracketClose", "Var", "Add", "Sub", "Number", "Comma", "Semi", "BracketOpen", 
      "Def", "BracketClose", "AngleBracketOpen", "Equal", "AngleBracketClose", 
      "WS", "Comment"
    }
  );
  static const int32_t serializedATNSegment[] = {
  	4,1,20,123,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,6,2,
  	7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,1,0,4,0,26,8,0,11,0,12,0,27,
  	1,1,1,1,1,1,3,1,33,8,1,1,2,1,2,3,2,37,8,2,1,3,1,3,1,3,1,3,1,3,1,3,5,3,
  	45,8,3,10,3,12,3,48,9,3,3,3,50,8,3,1,3,3,3,53,8,3,1,4,1,4,1,4,1,4,5,4,
  	59,8,4,10,4,12,4,62,9,4,3,4,64,8,4,1,4,1,4,3,4,68,8,4,1,5,1,5,1,5,3,5,
  	73,8,5,1,5,1,5,3,5,77,8,5,1,6,1,6,1,6,1,6,5,6,83,8,6,10,6,12,6,86,9,6,
  	1,6,1,6,1,7,1,7,1,7,1,8,1,8,1,8,1,8,3,8,97,8,8,1,8,1,8,1,9,1,9,1,9,1,
  	9,3,9,105,8,9,1,10,1,10,1,10,1,10,5,10,111,8,10,10,10,12,10,114,9,10,
  	1,10,1,10,1,11,1,11,1,11,3,11,121,8,11,1,11,0,0,12,0,2,4,6,8,10,12,14,
  	16,18,20,22,0,0,128,0,25,1,0,0,0,2,32,1,0,0,0,4,34,1,0,0,0,6,52,1,0,0,
  	0,8,67,1,0,0,0,10,69,1,0,0,0,12,78,1,0,0,0,14,89,1,0,0,0,16,92,1,0,0,
  	0,18,104,1,0,0,0,20,106,1,0,0,0,22,120,1,0,0,0,24,26,3,14,7,0,25,24,1,
  	0,0,0,26,27,1,0,0,0,27,25,1,0,0,0,27,28,1,0,0,0,28,1,1,0,0,0,29,33,5,
  	10,0,0,30,33,3,8,4,0,31,33,3,6,3,0,32,29,1,0,0,0,32,30,1,0,0,0,32,31,
  	1,0,0,0,33,3,1,0,0,0,34,36,5,1,0,0,35,37,3,2,1,0,36,35,1,0,0,0,36,37,
  	1,0,0,0,37,5,1,0,0,0,38,53,5,4,0,0,39,40,5,4,0,0,40,49,5,2,0,0,41,46,
  	3,2,1,0,42,43,5,11,0,0,43,45,3,2,1,0,44,42,1,0,0,0,45,48,1,0,0,0,46,44,
  	1,0,0,0,46,47,1,0,0,0,47,50,1,0,0,0,48,46,1,0,0,0,49,41,1,0,0,0,49,50,
  	1,0,0,0,50,51,1,0,0,0,51,53,5,5,0,0,52,38,1,0,0,0,52,39,1,0,0,0,53,7,
  	1,0,0,0,54,63,5,3,0,0,55,60,3,8,4,0,56,57,5,11,0,0,57,59,3,8,4,0,58,56,
  	1,0,0,0,59,62,1,0,0,0,60,58,1,0,0,0,60,61,1,0,0,0,61,64,1,0,0,0,62,60,
  	1,0,0,0,63,55,1,0,0,0,63,64,1,0,0,0,64,65,1,0,0,0,65,68,5,6,0,0,66,68,
  	5,10,0,0,67,54,1,0,0,0,67,66,1,0,0,0,68,9,1,0,0,0,69,70,5,7,0,0,70,72,
  	5,4,0,0,71,73,3,12,6,0,72,71,1,0,0,0,72,73,1,0,0,0,73,76,1,0,0,0,74,75,
  	5,17,0,0,75,77,3,2,1,0,76,74,1,0,0,0,76,77,1,0,0,0,77,11,1,0,0,0,78,79,
  	5,16,0,0,79,84,5,10,0,0,80,81,5,11,0,0,81,83,5,10,0,0,82,80,1,0,0,0,83,
  	86,1,0,0,0,84,82,1,0,0,0,84,85,1,0,0,0,85,87,1,0,0,0,86,84,1,0,0,0,87,
  	88,5,18,0,0,88,13,1,0,0,0,89,90,3,16,8,0,90,91,3,20,10,0,91,15,1,0,0,
  	0,92,93,5,14,0,0,93,94,5,4,0,0,94,96,5,2,0,0,95,97,3,18,9,0,96,95,1,0,
  	0,0,96,97,1,0,0,0,97,98,1,0,0,0,98,99,5,5,0,0,99,17,1,0,0,0,100,105,5,
  	4,0,0,101,102,5,4,0,0,102,103,5,11,0,0,103,105,3,18,9,0,104,100,1,0,0,
  	0,104,101,1,0,0,0,105,19,1,0,0,0,106,112,5,13,0,0,107,108,3,22,11,0,108,
  	109,5,12,0,0,109,111,1,0,0,0,110,107,1,0,0,0,111,114,1,0,0,0,112,110,
  	1,0,0,0,112,113,1,0,0,0,113,115,1,0,0,0,114,112,1,0,0,0,115,116,5,15,
  	0,0,116,21,1,0,0,0,117,121,3,10,5,0,118,121,3,4,2,0,119,121,3,2,1,0,120,
  	117,1,0,0,0,120,118,1,0,0,0,120,119,1,0,0,0,121,23,1,0,0,0,16,27,32,36,
  	46,49,52,60,63,67,72,76,84,96,104,112,120
  };
  staticData->serializedATN = antlr4::atn::SerializedATNView(serializedATNSegment, sizeof(serializedATNSegment) / sizeof(serializedATNSegment[0]));

  antlr4::atn::ATNDeserializer deserializer;
  staticData->atn = deserializer.deserialize(staticData->serializedATN);

  const size_t count = staticData->atn->getNumberOfDecisions();
  staticData->decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    staticData->decisionToDFA.emplace_back(staticData->atn->getDecisionState(i), i);
  }
  toyParserStaticData = staticData.release();
}

}

ToyParser::ToyParser(TokenStream *input) : ToyParser(input, antlr4::atn::ParserATNSimulatorOptions()) {}

ToyParser::ToyParser(TokenStream *input, const antlr4::atn::ParserATNSimulatorOptions &options) : Parser(input) {
  ToyParser::initialize();
  _interpreter = new atn::ParserATNSimulator(this, *toyParserStaticData->atn, toyParserStaticData->decisionToDFA, toyParserStaticData->sharedContextCache, options);
}

ToyParser::~ToyParser() {
  delete _interpreter;
}

const atn::ATN& ToyParser::getATN() const {
  return *toyParserStaticData->atn;
}

std::string ToyParser::getGrammarFileName() const {
  return "Toy.g4";
}

const std::vector<std::string>& ToyParser::getRuleNames() const {
  return toyParserStaticData->ruleNames;
}

const dfa::Vocabulary& ToyParser::getVocabulary() const {
  return toyParserStaticData->vocabulary;
}

antlr4::atn::SerializedATNView ToyParser::getSerializedATN() const {
  return toyParserStaticData->serializedATN;
}


//----------------- ModuleContext ------------------------------------------------------------------

ToyParser::ModuleContext::ModuleContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<ToyParser::FunDefineContext *> ToyParser::ModuleContext::funDefine() {
  return getRuleContexts<ToyParser::FunDefineContext>();
}

ToyParser::FunDefineContext* ToyParser::ModuleContext::funDefine(size_t i) {
  return getRuleContext<ToyParser::FunDefineContext>(i);
}


size_t ToyParser::ModuleContext::getRuleIndex() const {
  return ToyParser::RuleModule;
}


std::any ToyParser::ModuleContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ToyVisitor*>(visitor))
    return parserVisitor->visitModule(this);
  else
    return visitor->visitChildren(this);
}

ToyParser::ModuleContext* ToyParser::module() {
  ModuleContext *_localctx = _tracker.createInstance<ModuleContext>(_ctx, getState());
  enterRule(_localctx, 0, ToyParser::RuleModule);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(25); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(24);
      funDefine();
      setState(27); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while (_la == ToyParser::Def);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExpressionContext ------------------------------------------------------------------

ToyParser::ExpressionContext::ExpressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ToyParser::ExpressionContext::Number() {
  return getToken(ToyParser::Number, 0);
}

ToyParser::TensorLiteralContext* ToyParser::ExpressionContext::tensorLiteral() {
  return getRuleContext<ToyParser::TensorLiteralContext>(0);
}

ToyParser::IdentifierExprContext* ToyParser::ExpressionContext::identifierExpr() {
  return getRuleContext<ToyParser::IdentifierExprContext>(0);
}


size_t ToyParser::ExpressionContext::getRuleIndex() const {
  return ToyParser::RuleExpression;
}


std::any ToyParser::ExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ToyVisitor*>(visitor))
    return parserVisitor->visitExpression(this);
  else
    return visitor->visitChildren(this);
}

ToyParser::ExpressionContext* ToyParser::expression() {
  ExpressionContext *_localctx = _tracker.createInstance<ExpressionContext>(_ctx, getState());
  enterRule(_localctx, 2, ToyParser::RuleExpression);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(32);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 1, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(29);
      match(ToyParser::Number);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(30);
      tensorLiteral();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(31);
      identifierExpr();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ReturnExprContext ------------------------------------------------------------------

ToyParser::ReturnExprContext::ReturnExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ToyParser::ReturnExprContext::Return() {
  return getToken(ToyParser::Return, 0);
}

ToyParser::ExpressionContext* ToyParser::ReturnExprContext::expression() {
  return getRuleContext<ToyParser::ExpressionContext>(0);
}


size_t ToyParser::ReturnExprContext::getRuleIndex() const {
  return ToyParser::RuleReturnExpr;
}


std::any ToyParser::ReturnExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ToyVisitor*>(visitor))
    return parserVisitor->visitReturnExpr(this);
  else
    return visitor->visitChildren(this);
}

ToyParser::ReturnExprContext* ToyParser::returnExpr() {
  ReturnExprContext *_localctx = _tracker.createInstance<ReturnExprContext>(_ctx, getState());
  enterRule(_localctx, 4, ToyParser::RuleReturnExpr);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(34);
    match(ToyParser::Return);
    setState(36);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << ToyParser::SbracketOpen)
      | (1ULL << ToyParser::Identifier)
      | (1ULL << ToyParser::Number))) != 0)) {
      setState(35);
      expression();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IdentifierExprContext ------------------------------------------------------------------

ToyParser::IdentifierExprContext::IdentifierExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ToyParser::IdentifierExprContext::Identifier() {
  return getToken(ToyParser::Identifier, 0);
}

tree::TerminalNode* ToyParser::IdentifierExprContext::ParentheseOpen() {
  return getToken(ToyParser::ParentheseOpen, 0);
}

tree::TerminalNode* ToyParser::IdentifierExprContext::ParentheseClose() {
  return getToken(ToyParser::ParentheseClose, 0);
}

std::vector<ToyParser::ExpressionContext *> ToyParser::IdentifierExprContext::expression() {
  return getRuleContexts<ToyParser::ExpressionContext>();
}

ToyParser::ExpressionContext* ToyParser::IdentifierExprContext::expression(size_t i) {
  return getRuleContext<ToyParser::ExpressionContext>(i);
}

std::vector<tree::TerminalNode *> ToyParser::IdentifierExprContext::Comma() {
  return getTokens(ToyParser::Comma);
}

tree::TerminalNode* ToyParser::IdentifierExprContext::Comma(size_t i) {
  return getToken(ToyParser::Comma, i);
}


size_t ToyParser::IdentifierExprContext::getRuleIndex() const {
  return ToyParser::RuleIdentifierExpr;
}


std::any ToyParser::IdentifierExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ToyVisitor*>(visitor))
    return parserVisitor->visitIdentifierExpr(this);
  else
    return visitor->visitChildren(this);
}

ToyParser::IdentifierExprContext* ToyParser::identifierExpr() {
  IdentifierExprContext *_localctx = _tracker.createInstance<IdentifierExprContext>(_ctx, getState());
  enterRule(_localctx, 6, ToyParser::RuleIdentifierExpr);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(52);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 5, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(38);
      match(ToyParser::Identifier);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(39);
      match(ToyParser::Identifier);
      setState(40);
      match(ToyParser::ParentheseOpen);
      setState(49);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << ToyParser::SbracketOpen)
        | (1ULL << ToyParser::Identifier)
        | (1ULL << ToyParser::Number))) != 0)) {
        setState(41);
        expression();
        setState(46);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == ToyParser::Comma) {
          setState(42);
          match(ToyParser::Comma);
          setState(43);
          expression();
          setState(48);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
      }
      setState(51);
      match(ToyParser::ParentheseClose);
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TensorLiteralContext ------------------------------------------------------------------

ToyParser::TensorLiteralContext::TensorLiteralContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ToyParser::TensorLiteralContext::SbracketOpen() {
  return getToken(ToyParser::SbracketOpen, 0);
}

tree::TerminalNode* ToyParser::TensorLiteralContext::SbracketClose() {
  return getToken(ToyParser::SbracketClose, 0);
}

std::vector<ToyParser::TensorLiteralContext *> ToyParser::TensorLiteralContext::tensorLiteral() {
  return getRuleContexts<ToyParser::TensorLiteralContext>();
}

ToyParser::TensorLiteralContext* ToyParser::TensorLiteralContext::tensorLiteral(size_t i) {
  return getRuleContext<ToyParser::TensorLiteralContext>(i);
}

std::vector<tree::TerminalNode *> ToyParser::TensorLiteralContext::Comma() {
  return getTokens(ToyParser::Comma);
}

tree::TerminalNode* ToyParser::TensorLiteralContext::Comma(size_t i) {
  return getToken(ToyParser::Comma, i);
}

tree::TerminalNode* ToyParser::TensorLiteralContext::Number() {
  return getToken(ToyParser::Number, 0);
}


size_t ToyParser::TensorLiteralContext::getRuleIndex() const {
  return ToyParser::RuleTensorLiteral;
}


std::any ToyParser::TensorLiteralContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ToyVisitor*>(visitor))
    return parserVisitor->visitTensorLiteral(this);
  else
    return visitor->visitChildren(this);
}

ToyParser::TensorLiteralContext* ToyParser::tensorLiteral() {
  TensorLiteralContext *_localctx = _tracker.createInstance<TensorLiteralContext>(_ctx, getState());
  enterRule(_localctx, 8, ToyParser::RuleTensorLiteral);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(67);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case ToyParser::SbracketOpen: {
        enterOuterAlt(_localctx, 1);
        setState(54);
        match(ToyParser::SbracketOpen);
        setState(63);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == ToyParser::SbracketOpen

        || _la == ToyParser::Number) {
          setState(55);
          tensorLiteral();
          setState(60);
          _errHandler->sync(this);
          _la = _input->LA(1);
          while (_la == ToyParser::Comma) {
            setState(56);
            match(ToyParser::Comma);
            setState(57);
            tensorLiteral();
            setState(62);
            _errHandler->sync(this);
            _la = _input->LA(1);
          }
        }
        setState(65);
        match(ToyParser::SbracketClose);
        break;
      }

      case ToyParser::Number: {
        enterOuterAlt(_localctx, 2);
        setState(66);
        match(ToyParser::Number);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- VarDeclContext ------------------------------------------------------------------

ToyParser::VarDeclContext::VarDeclContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ToyParser::VarDeclContext::Var() {
  return getToken(ToyParser::Var, 0);
}

tree::TerminalNode* ToyParser::VarDeclContext::Identifier() {
  return getToken(ToyParser::Identifier, 0);
}

ToyParser::TypeContext* ToyParser::VarDeclContext::type() {
  return getRuleContext<ToyParser::TypeContext>(0);
}

tree::TerminalNode* ToyParser::VarDeclContext::Equal() {
  return getToken(ToyParser::Equal, 0);
}

ToyParser::ExpressionContext* ToyParser::VarDeclContext::expression() {
  return getRuleContext<ToyParser::ExpressionContext>(0);
}


size_t ToyParser::VarDeclContext::getRuleIndex() const {
  return ToyParser::RuleVarDecl;
}


std::any ToyParser::VarDeclContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ToyVisitor*>(visitor))
    return parserVisitor->visitVarDecl(this);
  else
    return visitor->visitChildren(this);
}

ToyParser::VarDeclContext* ToyParser::varDecl() {
  VarDeclContext *_localctx = _tracker.createInstance<VarDeclContext>(_ctx, getState());
  enterRule(_localctx, 10, ToyParser::RuleVarDecl);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(69);
    match(ToyParser::Var);
    setState(70);
    match(ToyParser::Identifier);
    setState(72);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == ToyParser::AngleBracketOpen) {
      setState(71);
      type();
    }
    setState(76);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == ToyParser::Equal) {
      setState(74);
      match(ToyParser::Equal);
      setState(75);
      expression();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TypeContext ------------------------------------------------------------------

ToyParser::TypeContext::TypeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ToyParser::TypeContext::AngleBracketOpen() {
  return getToken(ToyParser::AngleBracketOpen, 0);
}

std::vector<tree::TerminalNode *> ToyParser::TypeContext::Number() {
  return getTokens(ToyParser::Number);
}

tree::TerminalNode* ToyParser::TypeContext::Number(size_t i) {
  return getToken(ToyParser::Number, i);
}

tree::TerminalNode* ToyParser::TypeContext::AngleBracketClose() {
  return getToken(ToyParser::AngleBracketClose, 0);
}

std::vector<tree::TerminalNode *> ToyParser::TypeContext::Comma() {
  return getTokens(ToyParser::Comma);
}

tree::TerminalNode* ToyParser::TypeContext::Comma(size_t i) {
  return getToken(ToyParser::Comma, i);
}


size_t ToyParser::TypeContext::getRuleIndex() const {
  return ToyParser::RuleType;
}


std::any ToyParser::TypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ToyVisitor*>(visitor))
    return parserVisitor->visitType(this);
  else
    return visitor->visitChildren(this);
}

ToyParser::TypeContext* ToyParser::type() {
  TypeContext *_localctx = _tracker.createInstance<TypeContext>(_ctx, getState());
  enterRule(_localctx, 12, ToyParser::RuleType);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(78);
    match(ToyParser::AngleBracketOpen);
    setState(79);
    match(ToyParser::Number);
    setState(84);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == ToyParser::Comma) {
      setState(80);
      match(ToyParser::Comma);
      setState(81);
      match(ToyParser::Number);
      setState(86);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(87);
    match(ToyParser::AngleBracketClose);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FunDefineContext ------------------------------------------------------------------

ToyParser::FunDefineContext::FunDefineContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

ToyParser::PrototypeContext* ToyParser::FunDefineContext::prototype() {
  return getRuleContext<ToyParser::PrototypeContext>(0);
}

ToyParser::BlockContext* ToyParser::FunDefineContext::block() {
  return getRuleContext<ToyParser::BlockContext>(0);
}


size_t ToyParser::FunDefineContext::getRuleIndex() const {
  return ToyParser::RuleFunDefine;
}


std::any ToyParser::FunDefineContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ToyVisitor*>(visitor))
    return parserVisitor->visitFunDefine(this);
  else
    return visitor->visitChildren(this);
}

ToyParser::FunDefineContext* ToyParser::funDefine() {
  FunDefineContext *_localctx = _tracker.createInstance<FunDefineContext>(_ctx, getState());
  enterRule(_localctx, 14, ToyParser::RuleFunDefine);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(89);
    prototype();
    setState(90);
    block();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PrototypeContext ------------------------------------------------------------------

ToyParser::PrototypeContext::PrototypeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ToyParser::PrototypeContext::Def() {
  return getToken(ToyParser::Def, 0);
}

tree::TerminalNode* ToyParser::PrototypeContext::Identifier() {
  return getToken(ToyParser::Identifier, 0);
}

tree::TerminalNode* ToyParser::PrototypeContext::ParentheseOpen() {
  return getToken(ToyParser::ParentheseOpen, 0);
}

tree::TerminalNode* ToyParser::PrototypeContext::ParentheseClose() {
  return getToken(ToyParser::ParentheseClose, 0);
}

ToyParser::DeclListContext* ToyParser::PrototypeContext::declList() {
  return getRuleContext<ToyParser::DeclListContext>(0);
}


size_t ToyParser::PrototypeContext::getRuleIndex() const {
  return ToyParser::RulePrototype;
}


std::any ToyParser::PrototypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ToyVisitor*>(visitor))
    return parserVisitor->visitPrototype(this);
  else
    return visitor->visitChildren(this);
}

ToyParser::PrototypeContext* ToyParser::prototype() {
  PrototypeContext *_localctx = _tracker.createInstance<PrototypeContext>(_ctx, getState());
  enterRule(_localctx, 16, ToyParser::RulePrototype);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(92);
    match(ToyParser::Def);
    setState(93);
    match(ToyParser::Identifier);
    setState(94);
    match(ToyParser::ParentheseOpen);
    setState(96);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == ToyParser::Identifier) {
      setState(95);
      declList();
    }
    setState(98);
    match(ToyParser::ParentheseClose);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DeclListContext ------------------------------------------------------------------

ToyParser::DeclListContext::DeclListContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ToyParser::DeclListContext::Identifier() {
  return getToken(ToyParser::Identifier, 0);
}

tree::TerminalNode* ToyParser::DeclListContext::Comma() {
  return getToken(ToyParser::Comma, 0);
}

ToyParser::DeclListContext* ToyParser::DeclListContext::declList() {
  return getRuleContext<ToyParser::DeclListContext>(0);
}


size_t ToyParser::DeclListContext::getRuleIndex() const {
  return ToyParser::RuleDeclList;
}


std::any ToyParser::DeclListContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ToyVisitor*>(visitor))
    return parserVisitor->visitDeclList(this);
  else
    return visitor->visitChildren(this);
}

ToyParser::DeclListContext* ToyParser::declList() {
  DeclListContext *_localctx = _tracker.createInstance<DeclListContext>(_ctx, getState());
  enterRule(_localctx, 18, ToyParser::RuleDeclList);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(104);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 13, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(100);
      match(ToyParser::Identifier);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(101);
      match(ToyParser::Identifier);
      setState(102);
      match(ToyParser::Comma);
      setState(103);
      declList();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BlockContext ------------------------------------------------------------------

ToyParser::BlockContext::BlockContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ToyParser::BlockContext::BracketOpen() {
  return getToken(ToyParser::BracketOpen, 0);
}

tree::TerminalNode* ToyParser::BlockContext::BracketClose() {
  return getToken(ToyParser::BracketClose, 0);
}

std::vector<ToyParser::BlockExprContext *> ToyParser::BlockContext::blockExpr() {
  return getRuleContexts<ToyParser::BlockExprContext>();
}

ToyParser::BlockExprContext* ToyParser::BlockContext::blockExpr(size_t i) {
  return getRuleContext<ToyParser::BlockExprContext>(i);
}

std::vector<tree::TerminalNode *> ToyParser::BlockContext::Semi() {
  return getTokens(ToyParser::Semi);
}

tree::TerminalNode* ToyParser::BlockContext::Semi(size_t i) {
  return getToken(ToyParser::Semi, i);
}


size_t ToyParser::BlockContext::getRuleIndex() const {
  return ToyParser::RuleBlock;
}


std::any ToyParser::BlockContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ToyVisitor*>(visitor))
    return parserVisitor->visitBlock(this);
  else
    return visitor->visitChildren(this);
}

ToyParser::BlockContext* ToyParser::block() {
  BlockContext *_localctx = _tracker.createInstance<BlockContext>(_ctx, getState());
  enterRule(_localctx, 20, ToyParser::RuleBlock);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(106);
    match(ToyParser::BracketOpen);
    setState(112);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << ToyParser::Return)
      | (1ULL << ToyParser::SbracketOpen)
      | (1ULL << ToyParser::Identifier)
      | (1ULL << ToyParser::Var)
      | (1ULL << ToyParser::Number))) != 0)) {
      setState(107);
      blockExpr();
      setState(108);
      match(ToyParser::Semi);
      setState(114);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(115);
    match(ToyParser::BracketClose);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BlockExprContext ------------------------------------------------------------------

ToyParser::BlockExprContext::BlockExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

ToyParser::VarDeclContext* ToyParser::BlockExprContext::varDecl() {
  return getRuleContext<ToyParser::VarDeclContext>(0);
}

ToyParser::ReturnExprContext* ToyParser::BlockExprContext::returnExpr() {
  return getRuleContext<ToyParser::ReturnExprContext>(0);
}

ToyParser::ExpressionContext* ToyParser::BlockExprContext::expression() {
  return getRuleContext<ToyParser::ExpressionContext>(0);
}


size_t ToyParser::BlockExprContext::getRuleIndex() const {
  return ToyParser::RuleBlockExpr;
}


std::any ToyParser::BlockExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ToyVisitor*>(visitor))
    return parserVisitor->visitBlockExpr(this);
  else
    return visitor->visitChildren(this);
}

ToyParser::BlockExprContext* ToyParser::blockExpr() {
  BlockExprContext *_localctx = _tracker.createInstance<BlockExprContext>(_ctx, getState());
  enterRule(_localctx, 22, ToyParser::RuleBlockExpr);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(120);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case ToyParser::Var: {
        enterOuterAlt(_localctx, 1);
        setState(117);
        varDecl();
        break;
      }

      case ToyParser::Return: {
        enterOuterAlt(_localctx, 2);
        setState(118);
        returnExpr();
        break;
      }

      case ToyParser::SbracketOpen:
      case ToyParser::Identifier:
      case ToyParser::Number: {
        enterOuterAlt(_localctx, 3);
        setState(119);
        expression();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

void ToyParser::initialize() {
  std::call_once(toyParserOnceFlag, toyParserInitialize);
}
