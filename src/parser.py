# parser.py
# Parser for the NotScheme language.
# Converts a stream of tokens from the lexer into an Abstract Syntax Tree (AST).

from lexer import (
    LexerError,  # User's import change
    TokenType,
    Token,
)
from ast_nodes import (
    NumberNode,
    StringNode,
    BooleanNode,
    NilNode,
    SymbolNode,
    QuoteNode,
    CallNode,
    IfNode,
    LetBinding,
    LetNode,
    LambdaNode,
    GetNode,
    SetNode,
    WhileNode,
    BeginNode,
    StaticNode,
    FnNode,
    StructDefNode,
    UseNode,
    ProgramNode,
    Expression,
    TopLevelForm,
)
from typing import Any, List, Union, Optional  # User's import change


class ParserError(Exception):
    """Custom exception for parsing errors."""

    def __init__(self, message, token: Optional[Token] = None):
        if token and token.type != TokenType.EOF:
            super().__init__(
                f"{message} (at line {token.line}, col {token.col}, token: {token.value!r})"
            )
        elif token:
            super().__init__(f"{message} (at EOF)")
        else:
            super().__init__(message)
        self.token = token


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.position = 0

    def _current_token(self) -> Token:
        """Returns the current token without consuming it."""
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        raise ParserError("Unexpected end of input, expected EOF token.")

    def _peek_token(self) -> Token:
        """Returns the next token without consuming it, or EOF if at the end."""
        if self.position + 1 < len(self.tokens):
            return self.tokens[self.position + 1]
        return self._current_token()

    def _consume_token(self, expected_type: Optional[TokenType] = None) -> Token:
        """Consumes and returns the current token.
        Optionally checks if it matches the expected type."""
        token = self._current_token()
        if expected_type and token.type != expected_type:
            if token.type == TokenType.EOF:
                raise ParserError(
                    f"Unexpected end of input. Expected {expected_type.name}", token
                )
            raise ParserError(
                f"Expected token type {expected_type.name} but got {token.type.name}",
                token,
            )

        self.position += 1
        return token

    def parse_program(self) -> ProgramNode:
        """Parses the entire token stream into a ProgramNode."""
        forms: List[TopLevelForm] = []
        while self._current_token().type != TokenType.EOF:
            forms.append(self._parse_top_level_form())
        return ProgramNode(forms=forms)

    def _parse_top_level_form(self) -> TopLevelForm:
        """Parses a single top-level form (definition or expression)."""
        return self._parse_expression()

    def _parse_atom(self) -> Expression:
        """Parses an atomic expression (number, string, boolean, nil, symbol)."""
        token = self._current_token()
        if token.type == TokenType.NUMBER:
            self._consume_token()
            return NumberNode(value=token.value)
        elif token.type == TokenType.STRING:
            self._consume_token()
            return StringNode(value=token.value)
        elif token.type == TokenType.BOOLEAN:
            self._consume_token()
            return BooleanNode(value=token.value)
        elif token.type == TokenType.NIL:
            self._consume_token()
            return NilNode()
        elif token.type == TokenType.SYMBOL:
            self._consume_token()
            return SymbolNode(name=token.value)
        else:
            raise ParserError(
                f"Unexpected token type for atom: {token.type.name}", token
            )

    def _parse_expression(self) -> Expression:
        """Parses a single NotScheme expression."""
        token = self._current_token()
        if token.type == TokenType.LPAREN:
            return self._parse_list_or_call()
        elif token.type == TokenType.QUOTE:
            return self._parse_quote()  # This handles the outermost quote
        else:
            return self._parse_atom()

    def _parse_quote(self) -> QuoteNode:
        """Parses a quoted expression: 'expr. This is called when a QUOTE token is encountered."""
        self._consume_token(TokenType.QUOTE)  # Consume the initial '
        # _parse_quoted_s_expression will parse the content that is being quoted.
        # If that content itself starts with a quote (e.g., ' 'foo),
        # _parse_quoted_s_expression needs to handle that.
        quoted_data = self._parse_quoted_s_expression()
        return QuoteNode(expression=quoted_data)

    def _parse_quoted_s_expression(self) -> Any:
        """
        Parses a literal S-expression that might appear after a quote.
        This is the content of a QuoteNode.
        If this content is itself a quote (e.g. for ''foo), it should return a QuoteNode.
        """
        token = self._current_token()
        if token.type == TokenType.LPAREN:
            self._consume_token(TokenType.LPAREN)
            elements = []
            while self._current_token().type != TokenType.RPAREN:
                if self._current_token().type == TokenType.EOF:
                    raise ParserError(
                        "Unexpected EOF inside quoted list", self._current_token()
                    )
                elements.append(
                    self._parse_quoted_s_expression()
                )  # Recursively parse elements
            self._consume_token(TokenType.RPAREN)
            return elements  # Represents a quoted list as a Python list of its parsed quoted contents
        elif token.type == TokenType.SYMBOL:
            self._consume_token()
            return SymbolNode(
                name=token.value
            )  # Quoted symbols are represented as SymbolNodes
        elif token.type == TokenType.QUOTE:  # **FIXED HERE** Handle nested quotes
            self._consume_token(TokenType.QUOTE)  # Consume the inner quote
            expression_inside_inner_quote = (
                self._parse_quoted_s_expression()
            )  # Parse what it quotes
            return QuoteNode(
                expression=expression_inside_inner_quote
            )  # Return a QuoteNode for the inner quote
        elif token.type in [
            TokenType.NUMBER,
            TokenType.STRING,
            TokenType.BOOLEAN,
            TokenType.NIL,
        ]:
            # For other atoms, return their direct Python value when quoted
            atom_node = self._parse_atom()  # Consumes the token
            if isinstance(atom_node, (NumberNode, StringNode, BooleanNode)):
                return atom_node.value
            elif isinstance(atom_node, NilNode):
                return None  # Represent nil as Python None in quoted data
            # This path should ideally not be complex if _parse_atom is robust
            return atom_node  # Fallback, though specific cases above are preferred
        else:
            raise ParserError(
                f"Unexpected token in quoted S-expression: {token.type.name}", token
            )

    def _parse_list_or_call(self) -> Union[Expression, TopLevelForm]:
        """Parses a list, determining if it's a special form or a call."""
        start_token = self._consume_token(TokenType.LPAREN)

        if self._current_token().type == TokenType.RPAREN:
            self._consume_token(TokenType.RPAREN)
            raise ParserError(
                "Empty list () is not a valid expression or definition directly.",
                start_token,
            )

        first_element_token = self._current_token()

        if first_element_token.type == TokenType.SYMBOL:
            keyword = first_element_token.value
            if keyword == "static":
                return self._parse_static_definition(start_token)
            if keyword == "fn":
                return self._parse_fn_definition(start_token)
            if keyword == "struct":
                return self._parse_struct_definition(start_token)
            if keyword == "use":
                return self._parse_use_statement(start_token)
            if keyword == "if":
                return self._parse_if_expression(start_token)
            if keyword == "let":
                return self._parse_let_expression(start_token)
            if keyword == "lambda":
                return self._parse_lambda_expression(start_token)
            if keyword == "get":
                return self._parse_get_expression(start_token)
            if keyword == "set":
                return self._parse_set_expression(start_token)
            if keyword == "while":
                return self._parse_while_expression(start_token)
            if keyword == "begin":
                return self._parse_begin_expression(start_token)

        callable_expr = self._parse_expression()
        arguments: List[Expression] = []
        while self._current_token().type != TokenType.RPAREN:
            if self._current_token().type == TokenType.EOF:
                raise ParserError(
                    "Unexpected EOF inside list/call", self._current_token()
                )
            arguments.append(self._parse_expression())

        self._consume_token(TokenType.RPAREN)
        return CallNode(callable_expr=callable_expr, arguments=arguments)

    def _parse_static_definition(self, start_token: Token) -> StaticNode:
        self._consume_token(TokenType.SYMBOL)  # Consume 'static'
        name_token = self._consume_token(TokenType.SYMBOL)
        name_node = SymbolNode(name=name_token.value)
        value_expr = self._parse_expression()
        self._consume_token(TokenType.RPAREN)
        return StaticNode(name=name_node, value=value_expr)

    def _parse_fn_definition(self, start_token: Token) -> FnNode:
        self._consume_token(TokenType.SYMBOL)  # Consume 'fn'
        name_token = self._consume_token(TokenType.SYMBOL)
        name_node = SymbolNode(name=name_token.value)

        self._consume_token(TokenType.LPAREN)
        params: List[SymbolNode] = []
        while self._current_token().type != TokenType.RPAREN:
            if self._current_token().type == TokenType.EOF:
                raise ParserError(
                    "Unexpected EOF in function parameter list", self._current_token()
                )
            param_token = self._consume_token(TokenType.SYMBOL)
            params.append(SymbolNode(name=param_token.value))
        self._consume_token(TokenType.RPAREN)

        body: List[Expression] = []
        while self._current_token().type != TokenType.RPAREN:
            if self._current_token().type == TokenType.EOF:
                raise ParserError(
                    "Unexpected EOF in function body", self._current_token()
                )
            body.append(self._parse_expression())

        if not body:
            raise ParserError("Function body cannot be empty", self._current_token())

        self._consume_token(TokenType.RPAREN)
        return FnNode(name=name_node, params=params, body=body)

    def _parse_struct_definition(self, start_token: Token) -> StructDefNode:
        self._consume_token(TokenType.SYMBOL)  # Consume 'struct'
        name_token = self._consume_token(TokenType.SYMBOL)
        name_node = SymbolNode(name=name_token.value)

        self._consume_token(TokenType.LPAREN)
        fields: List[SymbolNode] = []
        while self._current_token().type != TokenType.RPAREN:
            if self._current_token().type == TokenType.EOF:
                raise ParserError(
                    "Unexpected EOF in struct field list", self._current_token()
                )
            field_token = self._consume_token(TokenType.SYMBOL)
            fields.append(SymbolNode(name=field_token.value))
        self._consume_token(TokenType.RPAREN)

        self._consume_token(TokenType.RPAREN)
        return StructDefNode(name=name_node, fields=fields)

    def _parse_use_statement(self, start_token: Token) -> UseNode:
        self._consume_token(TokenType.SYMBOL)  # Consume 'use'
        module_name_token = self._consume_token(TokenType.SYMBOL)
        module_name_node = SymbolNode(name=module_name_token.value)

        items: Union[List[SymbolNode], SymbolNode]
        if self._current_token().type == TokenType.LPAREN:
            self._consume_token(TokenType.LPAREN)
            imported_items: List[SymbolNode] = []
            while self._current_token().type != TokenType.RPAREN:
                if self._current_token().type == TokenType.EOF:
                    raise ParserError(
                        "Unexpected EOF in use item list", self._current_token()
                    )
                item_token = self._consume_token(TokenType.SYMBOL)
                imported_items.append(SymbolNode(name=item_token.value))
            self._consume_token(TokenType.RPAREN)
            items = imported_items
        elif (
            self._current_token().type == TokenType.SYMBOL
            and self._current_token().value == "*"
        ):
            star_token = self._consume_token(TokenType.SYMBOL)
            items = SymbolNode(name=star_token.value)
        else:
            raise ParserError(
                "Expected item list or '*' in use statement", self._current_token()
            )

        self._consume_token(TokenType.RPAREN)
        return UseNode(module_name=module_name_node, items=items)

    def _parse_if_expression(self, start_token: Token) -> IfNode:
        self._consume_token(TokenType.SYMBOL)  # Consume 'if'
        condition = self._parse_expression()
        then_branch = self._parse_expression()

        if self._current_token().type == TokenType.RPAREN:
            raise ParserError("'if' expression requires an else branch", start_token)

        else_branch = self._parse_expression()
        self._consume_token(TokenType.RPAREN)
        return IfNode(
            condition=condition, then_branch=then_branch, else_branch=else_branch
        )

    def _parse_let_expression(self, start_token: Token) -> LetNode:
        self._consume_token(TokenType.SYMBOL)  # Consume 'let'
        bindings: List[LetBinding] = []

        if (
            self._current_token().type == TokenType.SYMBOL
        ):  # Single binding form: (let var expr)
            var_token = self._consume_token(TokenType.SYMBOL)
            var_node = SymbolNode(name=var_token.value)
            val_expr = self._parse_expression()
            bindings.append(LetBinding(name=var_node, value=val_expr))
        elif (
            self._current_token().type == TokenType.LPAREN
        ):  # Multi-binding form: (let ((v1 e1)...) body)
            self._consume_token(TokenType.LPAREN)  # Consume outer '(' of bindings list
            while self._current_token().type != TokenType.RPAREN:
                if self._current_token().type == TokenType.EOF:
                    raise ParserError(
                        "Unexpected EOF in let bindings list", self._current_token()
                    )
                self._consume_token(
                    TokenType.LPAREN
                )  # Consume '(' for individual binding
                var_token = self._consume_token(TokenType.SYMBOL)
                var_node = SymbolNode(name=var_token.value)
                val_expr = self._parse_expression()
                bindings.append(LetBinding(name=var_node, value=val_expr))
                self._consume_token(
                    TokenType.RPAREN
                )  # Consume ')' for individual binding
            self._consume_token(TokenType.RPAREN)  # Consume outer ')' of bindings list
        else:
            raise ParserError("Invalid 'let' binding structure", self._current_token())

        if not bindings:
            raise ParserError("'let' must have at least one binding", start_token)

        body_expressions: List[Expression] = []
        while self._current_token().type != TokenType.RPAREN:
            if self._current_token().type == TokenType.EOF:
                raise ParserError("Unexpected EOF in let body", self._current_token())
            body_expressions.append(self._parse_expression())

        self._consume_token(TokenType.RPAREN)
        return LetNode(bindings=bindings, body=body_expressions)

    def _parse_lambda_expression(self, start_token: Token) -> LambdaNode:
        self._consume_token(TokenType.SYMBOL)  # Consume 'lambda'

        self._consume_token(TokenType.LPAREN)
        params: List[SymbolNode] = []
        while self._current_token().type != TokenType.RPAREN:
            if self._current_token().type == TokenType.EOF:
                raise ParserError(
                    "Unexpected EOF in lambda parameter list", self._current_token()
                )
            param_token = self._consume_token(TokenType.SYMBOL)
            params.append(SymbolNode(name=param_token.value))
        self._consume_token(TokenType.RPAREN)

        body: List[Expression] = []
        while self._current_token().type != TokenType.RPAREN:
            if self._current_token().type == TokenType.EOF:
                raise ParserError(
                    "Unexpected EOF in lambda body", self._current_token()
                )
            body.append(self._parse_expression())

        if not body:
            raise ParserError("Lambda body cannot be empty", self._current_token())

        self._consume_token(TokenType.RPAREN)
        return LambdaNode(params=params, body=body)

    def _parse_get_expression(self, start_token: Token) -> GetNode:
        self._consume_token(TokenType.SYMBOL)  # Consume 'get'
        instance_expr = self._parse_expression()
        field_name_token = self._consume_token(TokenType.SYMBOL)
        field_name_node = SymbolNode(name=field_name_token.value)
        self._consume_token(TokenType.RPAREN)
        return GetNode(instance=instance_expr, field_name=field_name_node)

    def _parse_set_expression(self, start_token: Token) -> SetNode:
        self._consume_token(TokenType.SYMBOL)  # Consume 'set'
        instance_expr = self._parse_expression()
        field_name_token = self._consume_token(TokenType.SYMBOL)
        field_name_node = SymbolNode(name=field_name_token.value)
        value_expr = self._parse_expression()
        self._consume_token(TokenType.RPAREN)
        return SetNode(
            instance=instance_expr, field_name=field_name_node, value=value_expr
        )

    def _parse_while_expression(self, start_token: Token) -> WhileNode:
        self._consume_token(TokenType.SYMBOL)  # Consume 'while'
        condition = self._parse_expression()

        body: List[Expression] = []
        while self._current_token().type != TokenType.RPAREN:
            if self._current_token().type == TokenType.EOF:
                raise ParserError("Unexpected EOF in while body", self._current_token())
            body.append(self._parse_expression())
        self._consume_token(TokenType.RPAREN)
        return WhileNode(condition=condition, body=body)

    def _parse_begin_expression(self, start_token: Token) -> BeginNode:
        self._consume_token(TokenType.SYMBOL)  # Consume 'begin'
        expressions: List[Expression] = []
        while self._current_token().type != TokenType.RPAREN:
            if self._current_token().type == TokenType.EOF:
                raise ParserError("Unexpected EOF in begin body", self._current_token())
            expressions.append(self._parse_expression())
        self._consume_token(TokenType.RPAREN)
        return BeginNode(expressions=expressions)


if __name__ == "__main__":
    from lexer import tokenize

    print("--- NotScheme Parser ---")

    test_code_1 = """
    // Top-level static definition
    (static pi 3.14)

    // Function definition
    (fn greet (name)
      (print "Hello, " name "!"))

    // Struct definition
    (struct Point (x_coord y_coord))

    // Top-level expression (call)
    (greet "World") 
    
    // Let expression (multi-binding with body)
    (let ((a 10) (b 20)) 
        (print (+ a b)))

    // Simpler let (single binding, now should parse with empty body in AST)
    (let message "A simple let") 

    // Quoted list (single binding, now should parse with empty body in AST)
    (let data '(1 foo #t))
    
    // While loop
    (begin
        (let count 0) 
        (let counter_struct (Point 0 0)) 
        (while (< (get counter_struct x_coord) 3)
            (print (get counter_struct x_coord))
            (set counter_struct x_coord (+ (get counter_struct x_coord) 1)))
        (print "Loop finished. Count was a local binding, not directly settable with our 'set'."))
    """
    print(f"\nParsing code:\n{test_code_1}")
    try:
        tokens = tokenize(test_code_1)
        parser = Parser(tokens)
        ast = parser.parse_program()
        print("\nAST:")

        import sys
        from ast_nodes import ProgramNode

        def print_ast(node, indent=0):
            indent_str = "  " * indent
            if isinstance(node, ProgramNode):
                print(f"{indent_str}ProgramNode:")
                for form in node.forms:
                    print_ast(form, indent + 1)
            elif isinstance(node, list):
                # Check if it's a list of AST nodes or a list of primitive quoted data
                if node and hasattr(node[0], "__dataclass_fields__"):
                    print(f"{indent_str}[")
                    for item in node:
                        print_ast(item, indent + 1)
                    print(f"{indent_str}]")
                else:  # Likely a list of primitive data from a quote
                    print(f"{indent_str}{node!r}")

            elif hasattr(node, "__dataclass_fields__"):
                print(f"{indent_str}{node.__class__.__name__}:")
                for field_name in node.__dataclass_fields__:
                    field_value = getattr(node, field_name)
                    if (  # Check if list of AST nodes
                        isinstance(field_value, list)
                        and field_value
                        and hasattr(field_value[0], "__dataclass_fields__")
                    ):
                        print(f"{indent_str}  {field_name}:")
                        print_ast(field_value, indent + 2)  # Pass list to print_ast
                    elif hasattr(
                        field_value, "__dataclass_fields__"
                    ):  # Single AST node
                        print(f"{indent_str}  {field_name}:")
                        print_ast(field_value, indent + 2)
                    else:  # Primitive value or list of primitives from quote
                        print(f"{indent_str}  {field_name}: {field_value!r}")
            else:
                print(f"{indent_str}{node!r}")  # Primitive from quote

        print_ast(ast)

    except (LexerError, ParserError) as e:
        print(f"Error: {e}")

    test_code_if_error = "(if #t 1)"
    print(f"\nParsing code with error:\n{test_code_if_error}")
    try:
        tokens = tokenize(test_code_if_error)
        parser = Parser(tokens)
        ast = parser.parse_program()
    except ParserError as e:
        print(f"Caught expected error: {e}")

    test_code_empty_list = "()"
    print(f"\nParsing code with error:\n{test_code_empty_list}")
    try:
        tokens = tokenize(test_code_empty_list)
        parser = Parser(tokens)
        ast = parser.parse_program()
    except ParserError as e:
        print(f"Caught expected error: {e}")

    test_code_quote = """
    (let x 'foo) 
    (let y '(bar (baz 10) #f))
    (let z ''(a b)) 
    """
    print(f"\nParsing quoted code:\n{test_code_quote}")
    try:
        tokens = tokenize(test_code_quote)
        parser = Parser(tokens)
        ast = parser.parse_program()
        print("\nAST (Quote Test):")
        print_ast(ast)
    except (LexerError, ParserError) as e:
        print(f"Error: {e}")
