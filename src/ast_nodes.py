# ast_nodes.py
# Defines the Abstract Syntax Tree (AST) node types for NotScheme.

from dataclasses import dataclass, field
from typing import List, Union, Any, Optional  # Any for value in literals for now

# --- Base Node (Optional, for common functionality if needed later) ---
# class ASTNode:
#     pass


# --- Literal Nodes ---
@dataclass
class NumberNode:
    value: Union[int, float]


@dataclass
class StringNode:
    value: str


@dataclass
class BooleanNode:
    value: bool


@dataclass
class NilNode:
    pass  # Represents the 'nil' value


@dataclass
class SymbolNode:  # Represents identifiers (variables, function names not yet resolved)
    name: str


@dataclass
class QuoteNode:  # Represents a quoted expression, e.g., '(1 2) or 'symbol
    expression: Any  # Can be any AST node representing the quoted data


# --- Expression Nodes ---
Expression = Union[
    NumberNode,
    StringNode,
    BooleanNode,
    NilNode,
    SymbolNode,
    QuoteNode,
    "CallNode",
    "IfNode",
    "LetNode",
    "LambdaNode",
    "GetNode",
    "SetNode",
    "WhileNode",
    "BeginNode",
]


@dataclass
class CallNode:
    callable_expr: Expression  # Could be a SymbolNode (function name) or a LambdaNode
    arguments: List[Expression]


@dataclass
class IfNode:
    condition: Expression
    then_branch: Expression
    else_branch: Expression  # NotScheme 'if' requires an else branch


@dataclass
class LetBinding:
    name: SymbolNode
    value: Expression


@dataclass
class LetNode:  # Handles both (let symbol expr) and (let ((s1 e1) (s2 e2)) body...)
    bindings: List[LetBinding]  # List of (name, value_expr) pairs
    body: List[Expression]  # Sequence of expressions in the let body


@dataclass
class LambdaNode:
    params: List[SymbolNode]  # Parameter names
    body: List[Expression]  # Sequence of expressions in the function body


@dataclass
class GetNode:  # For (get instance field)
    instance: Expression
    field_name: SymbolNode


@dataclass
class SetNode:  # For (set instance field value)
    instance: Expression
    field_name: SymbolNode
    value: Expression


@dataclass
class WhileNode:
    condition: Expression
    body: List[Expression]


@dataclass
class BeginNode:  # For (begin expr1 expr2 ...)
    expressions: List[Expression]


# --- Definition Nodes (Top-level or within certain scopes) ---
@dataclass
class StaticNode:  # (static name value_expr)
    name: SymbolNode
    value: Expression


@dataclass
class FnNode:  # (fn name (params...) body...)
    name: SymbolNode
    params: List[SymbolNode]
    body: List[Expression]


@dataclass
class StructDefNode:  # (struct Name (field1 field2 ...))
    name: SymbolNode  # Struct name (e.g., Point)
    fields: List[SymbolNode]  # Field names (e.g., x_coord, y_coord)


# --- Module-related Nodes ---
@dataclass
class UseNode:  # (use module_name (item1 item2 ...)) or (use module_name *)
    module_name: SymbolNode
    items: Union[
        List[SymbolNode], SymbolNode
    ]  # List of symbols or a single SymbolNode for '*'


# --- Program Node ---
# A program is a list of top-level forms (definitions or expressions)
TopLevelForm = Union[StaticNode, FnNode, StructDefNode, UseNode, Expression]


@dataclass
class ProgramNode:
    forms: List[TopLevelForm]
