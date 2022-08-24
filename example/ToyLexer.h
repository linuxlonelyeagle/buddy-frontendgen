
// Generated from Toy.g4 by ANTLR 4.10.1

#pragma once


#include "antlr4-runtime.h"




class  ToyLexer : public antlr4::Lexer {
public:
  enum {
    Return = 1, ParentheseOpen = 2, SbracketOpen = 3, Identifier = 4, ParentheseClose = 5, 
    SbracketClose = 6, Var = 7, Add = 8, Sub = 9, Number = 10, Comma = 11, 
    Semi = 12, BracketOpen = 13, Def = 14, BracketClose = 15, AngleBracketOpen = 16, 
    Equal = 17, AngleBracketClose = 18, WS = 19, Comment = 20
  };

  explicit ToyLexer(antlr4::CharStream *input);

  ~ToyLexer() override;


  std::string getGrammarFileName() const override;

  const std::vector<std::string>& getRuleNames() const override;

  const std::vector<std::string>& getChannelNames() const override;

  const std::vector<std::string>& getModeNames() const override;

  const antlr4::dfa::Vocabulary& getVocabulary() const override;

  antlr4::atn::SerializedATNView getSerializedATN() const override;

  const antlr4::atn::ATN& getATN() const override;

  // By default the static state used to implement the lexer is lazily initialized during the first
  // call to the constructor. You can call this function if you wish to initialize the static state
  // ahead of time.
  static void initialize();

private:

  // Individual action functions triggered by action() above.

  // Individual semantic predicate functions triggered by sempred() above.

};

