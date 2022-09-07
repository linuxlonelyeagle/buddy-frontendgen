grammar Toy;

module
  : funDefine 
  ;

expression
  : Number 
  | tensorLiteral 
  | identifierExpr 
  | expression Add expression 
  ;

returnExpr
  : Return expression ?
  ;

identifierExpr
  : Identifier 
  | Identifier ParentheseOpen (expression (Comma expression )*)?ParentheseClose 
  ;

tensorLiteral
  : SbracketOpen (tensorLiteral (Comma tensorLiteral )*)?SbracketClose 
  | Number 
  ;

varDecl
  : Var Identifier (type )?(Equal expression )?
  ;

type
  : AngleBracketOpen Number (Comma Number )*AngleBracketClose 
  ;

funDefine
  : prototype block 
  ;

prototype
  : Def Identifier ParentheseOpen declList ?ParentheseClose 
  ;

declList
  : Identifier 
  | Identifier Comma declList 
  ;

block
  : BracketOpen (blockExpr Semi )*BracketClose 
  ;

blockExpr
  : varDecl 
  | returnExpr 
  | expression 
  ;

Return
  : 'return'
  ;

ParentheseOpen
  : '('
  ;

SbracketOpen
  : '['
  ;

Identifier
  : [a-zA-Z][a-zA-Z0-9_]*
  ;

ParentheseClose
  : ')'
  ;

SbracketClose
  : ']'
  ;

Var
  : 'var'
  ;

Add
  : 'add'
  ;

Sub
  : 'sub'
  ;

Number
  : [0-9]+
  ;

Comma
  : ','
  ;

Semi
  : ';'
  ;

BracketOpen
  : '{'
  ;

Def
  : 'def'
  ;

BracketClose
  : '}'
  ;

AngleBracketOpen
  : '<'
  ;

Equal
  : '='
  ;

AngleBracketClose
  : '>'
  ;

WS
  : [ \r\n\t] -> skip
  ;

Comment
  : '#' .*? '\n' ->skip
  ;
