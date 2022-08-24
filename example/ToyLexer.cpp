
// Generated from Toy.g4 by ANTLR 4.10.1


#include "ToyLexer.h"


using namespace antlr4;



using namespace antlr4;

namespace {

struct ToyLexerStaticData final {
  ToyLexerStaticData(std::vector<std::string> ruleNames,
                          std::vector<std::string> channelNames,
                          std::vector<std::string> modeNames,
                          std::vector<std::string> literalNames,
                          std::vector<std::string> symbolicNames)
      : ruleNames(std::move(ruleNames)), channelNames(std::move(channelNames)),
        modeNames(std::move(modeNames)), literalNames(std::move(literalNames)),
        symbolicNames(std::move(symbolicNames)),
        vocabulary(this->literalNames, this->symbolicNames) {}

  ToyLexerStaticData(const ToyLexerStaticData&) = delete;
  ToyLexerStaticData(ToyLexerStaticData&&) = delete;
  ToyLexerStaticData& operator=(const ToyLexerStaticData&) = delete;
  ToyLexerStaticData& operator=(ToyLexerStaticData&&) = delete;

  std::vector<antlr4::dfa::DFA> decisionToDFA;
  antlr4::atn::PredictionContextCache sharedContextCache;
  const std::vector<std::string> ruleNames;
  const std::vector<std::string> channelNames;
  const std::vector<std::string> modeNames;
  const std::vector<std::string> literalNames;
  const std::vector<std::string> symbolicNames;
  const antlr4::dfa::Vocabulary vocabulary;
  antlr4::atn::SerializedATNView serializedATN;
  std::unique_ptr<antlr4::atn::ATN> atn;
};

std::once_flag toylexerLexerOnceFlag;
ToyLexerStaticData *toylexerLexerStaticData = nullptr;

void toylexerLexerInitialize() {
  assert(toylexerLexerStaticData == nullptr);
  auto staticData = std::make_unique<ToyLexerStaticData>(
    std::vector<std::string>{
      "Return", "ParentheseOpen", "SbracketOpen", "Identifier", "ParentheseClose", 
      "SbracketClose", "Var", "Add", "Sub", "Number", "Comma", "Semi", "BracketOpen", 
      "Def", "BracketClose", "AngleBracketOpen", "Equal", "AngleBracketClose", 
      "WS", "Comment"
    },
    std::vector<std::string>{
      "DEFAULT_TOKEN_CHANNEL", "HIDDEN"
    },
    std::vector<std::string>{
      "DEFAULT_MODE"
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
  	4,0,20,113,6,-1,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
  	6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,2,14,
  	7,14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,1,0,1,0,1,0,1,
  	0,1,0,1,0,1,0,1,1,1,1,1,2,1,2,1,3,1,3,5,3,55,8,3,10,3,12,3,58,9,3,1,4,
  	1,4,1,5,1,5,1,6,1,6,1,6,1,6,1,7,1,7,1,7,1,7,1,8,1,8,1,8,1,8,1,9,4,9,77,
  	8,9,11,9,12,9,78,1,10,1,10,1,11,1,11,1,12,1,12,1,13,1,13,1,13,1,13,1,
  	14,1,14,1,15,1,15,1,16,1,16,1,17,1,17,1,18,1,18,1,18,1,18,1,19,1,19,5,
  	19,105,8,19,10,19,12,19,108,9,19,1,19,1,19,1,19,1,19,1,106,0,20,1,1,3,
  	2,5,3,7,4,9,5,11,6,13,7,15,8,17,9,19,10,21,11,23,12,25,13,27,14,29,15,
  	31,16,33,17,35,18,37,19,39,20,1,0,4,2,0,65,90,97,122,4,0,48,57,65,90,
  	95,95,97,122,1,0,48,57,3,0,9,10,13,13,32,32,115,0,1,1,0,0,0,0,3,1,0,0,
  	0,0,5,1,0,0,0,0,7,1,0,0,0,0,9,1,0,0,0,0,11,1,0,0,0,0,13,1,0,0,0,0,15,
  	1,0,0,0,0,17,1,0,0,0,0,19,1,0,0,0,0,21,1,0,0,0,0,23,1,0,0,0,0,25,1,0,
  	0,0,0,27,1,0,0,0,0,29,1,0,0,0,0,31,1,0,0,0,0,33,1,0,0,0,0,35,1,0,0,0,
  	0,37,1,0,0,0,0,39,1,0,0,0,1,41,1,0,0,0,3,48,1,0,0,0,5,50,1,0,0,0,7,52,
  	1,0,0,0,9,59,1,0,0,0,11,61,1,0,0,0,13,63,1,0,0,0,15,67,1,0,0,0,17,71,
  	1,0,0,0,19,76,1,0,0,0,21,80,1,0,0,0,23,82,1,0,0,0,25,84,1,0,0,0,27,86,
  	1,0,0,0,29,90,1,0,0,0,31,92,1,0,0,0,33,94,1,0,0,0,35,96,1,0,0,0,37,98,
  	1,0,0,0,39,102,1,0,0,0,41,42,5,114,0,0,42,43,5,101,0,0,43,44,5,116,0,
  	0,44,45,5,117,0,0,45,46,5,114,0,0,46,47,5,110,0,0,47,2,1,0,0,0,48,49,
  	5,41,0,0,49,4,1,0,0,0,50,51,5,91,0,0,51,6,1,0,0,0,52,56,7,0,0,0,53,55,
  	7,1,0,0,54,53,1,0,0,0,55,58,1,0,0,0,56,54,1,0,0,0,56,57,1,0,0,0,57,8,
  	1,0,0,0,58,56,1,0,0,0,59,60,5,40,0,0,60,10,1,0,0,0,61,62,5,93,0,0,62,
  	12,1,0,0,0,63,64,5,118,0,0,64,65,5,97,0,0,65,66,5,114,0,0,66,14,1,0,0,
  	0,67,68,5,97,0,0,68,69,5,100,0,0,69,70,5,100,0,0,70,16,1,0,0,0,71,72,
  	5,115,0,0,72,73,5,117,0,0,73,74,5,98,0,0,74,18,1,0,0,0,75,77,7,2,0,0,
  	76,75,1,0,0,0,77,78,1,0,0,0,78,76,1,0,0,0,78,79,1,0,0,0,79,20,1,0,0,0,
  	80,81,5,44,0,0,81,22,1,0,0,0,82,83,5,59,0,0,83,24,1,0,0,0,84,85,5,123,
  	0,0,85,26,1,0,0,0,86,87,5,100,0,0,87,88,5,101,0,0,88,89,5,102,0,0,89,
  	28,1,0,0,0,90,91,5,125,0,0,91,30,1,0,0,0,92,93,5,60,0,0,93,32,1,0,0,0,
  	94,95,5,61,0,0,95,34,1,0,0,0,96,97,5,62,0,0,97,36,1,0,0,0,98,99,7,3,0,
  	0,99,100,1,0,0,0,100,101,6,18,0,0,101,38,1,0,0,0,102,106,5,35,0,0,103,
  	105,9,0,0,0,104,103,1,0,0,0,105,108,1,0,0,0,106,107,1,0,0,0,106,104,1,
  	0,0,0,107,109,1,0,0,0,108,106,1,0,0,0,109,110,5,10,0,0,110,111,1,0,0,
  	0,111,112,6,19,0,0,112,40,1,0,0,0,4,0,56,78,106,1,6,0,0
  };
  staticData->serializedATN = antlr4::atn::SerializedATNView(serializedATNSegment, sizeof(serializedATNSegment) / sizeof(serializedATNSegment[0]));

  antlr4::atn::ATNDeserializer deserializer;
  staticData->atn = deserializer.deserialize(staticData->serializedATN);

  const size_t count = staticData->atn->getNumberOfDecisions();
  staticData->decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    staticData->decisionToDFA.emplace_back(staticData->atn->getDecisionState(i), i);
  }
  toylexerLexerStaticData = staticData.release();
}

}

ToyLexer::ToyLexer(CharStream *input) : Lexer(input) {
  ToyLexer::initialize();
  _interpreter = new atn::LexerATNSimulator(this, *toylexerLexerStaticData->atn, toylexerLexerStaticData->decisionToDFA, toylexerLexerStaticData->sharedContextCache);
}

ToyLexer::~ToyLexer() {
  delete _interpreter;
}

std::string ToyLexer::getGrammarFileName() const {
  return "Toy.g4";
}

const std::vector<std::string>& ToyLexer::getRuleNames() const {
  return toylexerLexerStaticData->ruleNames;
}

const std::vector<std::string>& ToyLexer::getChannelNames() const {
  return toylexerLexerStaticData->channelNames;
}

const std::vector<std::string>& ToyLexer::getModeNames() const {
  return toylexerLexerStaticData->modeNames;
}

const dfa::Vocabulary& ToyLexer::getVocabulary() const {
  return toylexerLexerStaticData->vocabulary;
}

antlr4::atn::SerializedATNView ToyLexer::getSerializedATN() const {
  return toylexerLexerStaticData->serializedATN;
}

const atn::ATN& ToyLexer::getATN() const {
  return *toylexerLexerStaticData->atn;
}




void ToyLexer::initialize() {
  std::call_once(toylexerLexerOnceFlag, toylexerLexerInitialize);
}
