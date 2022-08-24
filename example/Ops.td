include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
def ShapeInferenceOpInterface : OpInterface<"ShapeInference">{
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.
  }];
  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "void", "inferShapes">
  ];
}

def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "::mlir::toy";
  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

class Toy_Op<string mnemonic, list<Trait> traits = []> :
  Op<Toy_Dialect, mnemonic, traits>;

def ConsantOp : Toy_Op<"consant", [NoSideEffect]> { 
  let summary = "consant";
  let arguments = (ins F64ElementsAttr : $value);
  let results = (outs F64Tensor);
  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "DenseElementsAttr" : $value),
    [{ build($_builder, $_state, value.getType(), value); }]>,
    OpBuilder<(ins "double" : $value)>];
}

def AddOp : Toy_Op<"add", [ NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface> ]> { 
  let arguments = (ins F64Tensor : $lhs, F64Tensor: $rhs);
  let results = (outs F64Tensor);
  let hasCustomAssemblyFormat = 1;
  let builders = [OpBuilder<(ins "Value" : $lhs, "Value" : $rhs)>];
}

def CastOp : Toy_Op<"cast", [
    DeclareOpInterfaceMethods<CastOpInterface>,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    NoSideEffect,
    SameOperandsAndResultShape]> { 
  let summary = "shape cast operation";
  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}

def FuncOp : Toy_Op<"func", [
    DeclareOpInterfaceMethods<CallableOpInterface>, FunctionOpInterface,
    IsolatedFromAbove, Symbol
  ]> { 
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type
  );
  let hasCustomAssemblyFormat = 1;
  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>];
  let extraClassDeclaration = [{
    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }}];
  let  skipDefaultBuilders = 1;
  let regions = (region AnyRegion:$body);
}

def MulOp : Toy_Op<"mul", [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> { 
  let summary = "element-wise multiplication operation";
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def PrintOp : Toy_Op<"print"> { 
  let arguments = (ins F64Tensor:$input);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def Reshape : Toy_Op<"reshape", [NoSideEffect]> { 
  let arguments = (ins F64Tensor : $input);
  let results = (outs StaticShapeTensorOf<[F64]>);
  let hasCanonicalizer = 1;
  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
    }];
}

def ReturnOp : Toy_Op<"return", [NoSideEffect, HasParent<"FuncOp">, Terminator]> { 
  let arguments = (ins Variadic<F64Tensor>:$input);
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, llvm::None); }]>
  ];
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";
}

