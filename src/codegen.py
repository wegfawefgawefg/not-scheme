# codegen.py
# Code Generator for the NotScheme language.
# Converts an AST into bytecode for the NotScheme VM.

from ast_nodes import (
    ProgramNode,
    StaticNode,
    FnNode,
    StructDefNode,
    UseNode,
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
    Expression,
    TopLevelForm,
)
from vm import OpCode, QuotedSymbol  # Import QuotedSymbol from vm
from typing import List, Dict, Any, Tuple, Union, get_args

EXPRESSION_NODE_TYPES = (
    NumberNode,
    StringNode,
    BooleanNode,
    NilNode,
    SymbolNode,
    QuoteNode,
    CallNode,
    IfNode,
    LetNode,
    LambdaNode,
    GetNode,
    SetNode,
    WhileNode,
    BeginNode,
)


class CodeGenerationError(Exception):
    pass


PRIMITIVE_OPERATIONS = {
    "+": (OpCode.ADD, None),
    "-": (OpCode.SUB, None),
    "*": (OpCode.MUL, None),
    "/": (OpCode.DIV, None),
    "=": (OpCode.EQ, None),
    ">": (OpCode.GT, None),
    "<": (OpCode.LT, None),
    "not": (OpCode.NOT, None),
    "print": (OpCode.PRINT, -1),
    "is_nil": (OpCode.IS_NIL, None),
    "cons": (OpCode.CONS, None),
    "first": (OpCode.FIRST, None),
    "rest": (OpCode.REST, None),
    "list": (OpCode.MAKE_LIST, -1),
}


class CodeGenerator:
    def __init__(self):
        self.bytecode: List[Union[tuple, str]] = []
        self.label_count = 0
        self.global_env: Dict[str, Any] = {}
        self.struct_definitions: Dict[str, List[str]] = {}
        self.compile_time_env_chain: List[Dict[str, str]] = [{"type": "global"}]

    def _new_label(self, prefix="L") -> str:
        self.label_count += 1
        return f"{prefix}{self.label_count}"

    def _emit(self, *args: Any):
        if len(args) == 1 and isinstance(args[0], str) and args[0].endswith(":"):
            self.bytecode.append(args[0])
        else:
            self.bytecode.append(tuple(args))

    def _enter_scope(self):
        self.compile_time_env_chain.append({"type": "local"})

    def _exit_scope(self):
        if len(self.compile_time_env_chain) > 1:
            self.compile_time_env_chain.pop()
        else:
            print("Warning: Attempted to pop global scope from compile_time_env_chain.")

    def _add_local_to_current_scope(self, name: str):
        if (
            self.compile_time_env_chain
            and self.compile_time_env_chain[-1]["type"] == "local"
        ):
            self.compile_time_env_chain[-1][name] = "local_variable"
        else:
            print(
                f"Warning: Could not add '{name}' to current compile-time local scope."
            )

    def generate_program(self, program_node: ProgramNode) -> List[Union[tuple, str]]:
        self.bytecode = []
        self.global_env.clear()
        self.struct_definitions.clear()
        self.compile_time_env_chain = [{"type": "global"}]
        for i, form in enumerate(program_node.forms):
            self._generate_top_level_form(
                form, is_last_form_in_program=(i == len(program_node.forms) - 1)
            )
        if not self.bytecode or not (
            isinstance(self.bytecode[-1], tuple)
            and self.bytecode[-1][0] in (OpCode.HALT, OpCode.RETURN, OpCode.JUMP)
        ):
            self._emit(OpCode.HALT)
        return self.bytecode

    def _generate_top_level_form(
        self, form: TopLevelForm, is_last_form_in_program: bool = False
    ):
        start_len = len(self.bytecode)
        if isinstance(form, StaticNode):
            self._generate_static_node(form)
        elif isinstance(form, FnNode):
            self._generate_fn_node(form)
        elif isinstance(form, StructDefNode):
            self._generate_struct_def_node(form)
        elif isinstance(form, UseNode):
            self._generate_use_node(form)
        elif isinstance(form, EXPRESSION_NODE_TYPES):
            self._generate_expression(form)
            if not is_last_form_in_program and len(self.bytecode) > start_len:
                last_instr = self.bytecode[-1]
                if isinstance(last_instr, tuple) and last_instr[0] not in (
                    OpCode.STORE,
                    OpCode.JUMP,
                    OpCode.RETURN,
                    OpCode.HALT,
                    OpCode.PRINT,
                ):
                    self._emit(OpCode.POP)
        else:
            raise CodeGenerationError(f"Unsupported top-level form: {type(form)}")

    def _generate_expression(self, expr_node: Expression):
        if isinstance(expr_node, NumberNode):
            self._emit(OpCode.PUSH, expr_node.value)
        elif isinstance(expr_node, StringNode):
            self._emit(OpCode.PUSH, expr_node.value)
        elif isinstance(expr_node, BooleanNode):
            self._emit(OpCode.PUSH, expr_node.value)
        elif isinstance(expr_node, NilNode):
            self._emit(OpCode.PUSH, None)
        elif isinstance(expr_node, SymbolNode):
            self._emit(OpCode.LOAD, expr_node.name)
        elif isinstance(expr_node, QuoteNode):
            self._generate_quote_node(
                expr_node
            )  # Changed from _generate_quoted_expression
        elif isinstance(expr_node, CallNode):
            self._generate_call_node(expr_node)
        elif isinstance(expr_node, IfNode):
            self._generate_if_node(expr_node)
        elif isinstance(expr_node, LetNode):
            self._generate_let_node(expr_node)
        elif isinstance(expr_node, LambdaNode):
            self._generate_lambda_node(expr_node)
        elif isinstance(expr_node, GetNode):
            self._generate_get_node(expr_node)
        elif isinstance(expr_node, SetNode):
            self._generate_set_node(expr_node)
        elif isinstance(expr_node, WhileNode):
            self._generate_while_node(expr_node)
        elif isinstance(expr_node, BeginNode):
            self._generate_begin_node(expr_node)
        elif isinstance(expr_node, (StaticNode, FnNode, StructDefNode, UseNode)):
            raise CodeGenerationError(
                f"Definition node {type(expr_node)} found in expression context."
            )
        else:
            raise CodeGenerationError(f"Unsupported expression type: {type(expr_node)}")

    def _generate_runtime_value_for_quoted_item(self, item_data: Any):
        """
        Generates bytecode to PUSH the runtime representation of a single item
        found within a quoted structure.
        - Quoted symbols become QuotedSymbol objects.
        - Quoted lists become runtime lists of such items.
        - Literals (numbers, strings, booleans, nil) are pushed as their Python equivalents.
        - Nested QuoteNodes mean we're creating a list like (quote actual_item).
        """
        if isinstance(
            item_data, SymbolNode
        ):  # e.g. 'foo from parser's QuoteNode(SymbolNode('foo'))
            self._emit(OpCode.PUSH, QuotedSymbol(name=item_data.name))
        elif isinstance(
            item_data, list
        ):  # This is a parsed quoted list e.g. from '(a b) -> [SymbolNode('a'), SymbolNode('b')]
            for sub_item in item_data:
                self._generate_runtime_value_for_quoted_item(
                    sub_item
                )  # Push each element's runtime form
            self._emit(OpCode.MAKE_LIST, len(item_data))  # Construct the list
        elif isinstance(
            item_data, QuoteNode
        ):  # e.g. ' 'a  or '(quote a) from parser -> QuoteNode(QuoteNode(SymbolNode('a')))
            # This means we want to create the structure (list 'quote <inner_quoted_value>)
            self._emit(
                OpCode.PUSH, QuotedSymbol(name="quote")
            )  # Push the symbol 'quote'
            self._generate_runtime_value_for_quoted_item(
                item_data.expression
            )  # Push the runtime value of what's inside the inner quote
            self._emit(OpCode.MAKE_LIST, 2)  # Create (list 'quote <inner_value>)
        elif isinstance(item_data, (int, float, str, bool)) or item_data is None:
            # These are Python primitives that the parser's _parse_quoted_s_expression
            # returns for literal numbers, strings, booleans, nil inside a quote.
            self._emit(OpCode.PUSH, item_data)
        else:
            # This case might be hit if _parse_quoted_s_expression returns other AST nodes directly
            # for atoms, which it currently doesn't (it returns Python values).
            raise CodeGenerationError(
                f"Cannot generate runtime value for quoted item of type: {type(item_data)}, value: {item_data!r}"
            )

    def _generate_quote_node(self, node: QuoteNode):
        """
        Generates code for a QuoteNode. The node.expression is what's being quoted.
        Example: 'foo -> QuoteNode(SymbolNode('foo'))
                 '(a b) -> QuoteNode([SymbolNode('a'), SymbolNode('b')])
                 ''foo -> QuoteNode(QuoteNode(SymbolNode('foo')))
        """
        self._generate_runtime_value_for_quoted_item(node.expression)

    def _generate_static_node(self, node: StaticNode):
        self._generate_expression(node.value)
        self._emit(OpCode.STORE, node.name.name)
        self.global_env[node.name.name] = "static_variable"

    def _generate_fn_or_lambda_body(
        self,
        name_for_label: str,
        params: List[SymbolNode],
        body: List[Expression],
        is_named_fn: bool,
    ):
        entry_label = self._new_label(
            f"{'fn' if is_named_fn else 'lambda'}_{name_for_label}"
        )
        if is_named_fn:
            self._emit(OpCode.MAKE_CLOSURE, entry_label)
            self._emit(OpCode.STORE, name_for_label)
            self.global_env[name_for_label] = {
                "type": "function",
                "label": entry_label,
                "params": [p.name for p in params],
            }
        else:
            self._emit(OpCode.MAKE_CLOSURE, entry_label)
        end_body_label = self._new_label(
            f"end_{'fn' if is_named_fn else 'lambda'}_{name_for_label}"
        )
        self._emit(OpCode.JUMP, end_body_label)
        self._emit(entry_label + ":")
        self._enter_scope()
        for param_node in reversed(params):
            self._emit(OpCode.STORE, param_node.name)
            self._add_local_to_current_scope(param_node.name)
        if not body:
            self._emit(OpCode.PUSH, None)
        else:
            for i, expr in enumerate(body):
                self._generate_expression(expr)
                if i < len(body) - 1:
                    self._emit(OpCode.POP)
        self._emit(OpCode.RETURN)
        self._exit_scope()
        self._emit(end_body_label + ":")

    def _generate_fn_node(self, node: FnNode):
        self._generate_fn_or_lambda_body(
            node.name.name, node.params, node.body, is_named_fn=True
        )

    def _generate_lambda_node(self, node: LambdaNode):
        self._generate_fn_or_lambda_body(
            "anon", node.params, node.body, is_named_fn=False
        )

    def _generate_struct_def_node(self, node: StructDefNode):
        field_names = [field.name for field in node.fields]
        if node.name.name in self.struct_definitions:
            raise CodeGenerationError(f"Struct '{node.name.name}' already defined.")
        self.struct_definitions[node.name.name] = field_names
        self.global_env[node.name.name] = {"type": "struct_type", "fields": field_names}

    def _generate_call_node(self, node: CallNode):
        if isinstance(node.callable_expr, SymbolNode):
            op_name = node.callable_expr.name
            if op_name in PRIMITIVE_OPERATIONS:
                opcode, arity_info = PRIMITIVE_OPERATIONS[op_name]
                if op_name == "print":
                    if not node.arguments:
                        self._emit(OpCode.PUSH, "")
                        self._emit(OpCode.PRINT)
                    else:
                        for arg_expr in node.arguments:
                            self._generate_expression(arg_expr)
                            self._emit(OpCode.PRINT)
                    self._emit(OpCode.PUSH, None)
                    return
                elif op_name == "list":
                    for arg_expr in node.arguments:
                        self._generate_expression(arg_expr)
                    self._emit(OpCode.MAKE_LIST, len(node.arguments))
                    return
                if arity_info is None:
                    expected_arity = 0
                    is_cons_op = op_name == "cons"
                    if op_name in ["+", "-", "*", "/", "=", ">", "<", "cons"]:
                        expected_arity = 2
                    elif op_name in ["not", "is_nil", "first", "rest"]:
                        expected_arity = 1
                    if len(node.arguments) != expected_arity:
                        raise CodeGenerationError(
                            f"Primitive '{op_name}' expects {expected_arity} args, got {len(node.arguments)}"
                        )
                    if is_cons_op:
                        self._generate_expression(node.arguments[1])
                        self._generate_expression(node.arguments[0])
                    else:
                        for arg_expr in node.arguments:
                            self._generate_expression(arg_expr)
                    self._emit(opcode)
                    return
            elif op_name in self.struct_definitions:
                struct_name, field_names = op_name, self.struct_definitions[op_name]
                if len(node.arguments) != len(field_names):
                    raise CodeGenerationError(
                        f"Struct '{struct_name}': expected {len(field_names)} args, got {len(node.arguments)}."
                    )
                for arg_expr in node.arguments:
                    self._generate_expression(arg_expr)
                self._emit(OpCode.MAKE_STRUCT, struct_name, tuple(field_names))
                return
        for arg_expr in node.arguments:
            self._generate_expression(arg_expr)
        self._generate_expression(node.callable_expr)
        self._emit(OpCode.CALL, len(node.arguments))

    def _generate_if_node(self, node: IfNode):
        else_label, end_if_label = self._new_label("else"), self._new_label("end_if")
        self._generate_expression(node.condition)
        self._emit(OpCode.JUMP_IF_FALSE, else_label)
        self._generate_expression(node.then_branch)
        self._emit(OpCode.JUMP, end_if_label)
        self._emit(else_label + ":")
        self._generate_expression(node.else_branch)
        self._emit(end_if_label + ":")

    def _generate_let_node(self, node: LetNode):
        self._enter_scope()
        for binding in node.bindings:
            self._generate_expression(binding.value)
            self._emit(OpCode.STORE, binding.name.name)
            self._add_local_to_current_scope(binding.name.name)
        if not node.body:
            self._emit(OpCode.PUSH, None)
        else:
            for i, expr in enumerate(node.body):
                self._generate_expression(expr)
                if i < len(node.body) - 1:
                    self._emit(OpCode.POP)
        self._exit_scope()

    def _generate_get_node(self, node: GetNode):
        self._generate_expression(node.instance)
        self._emit(OpCode.GET_FIELD, node.field_name.name)

    def _generate_set_node(self, node: SetNode):
        self._generate_expression(node.instance)
        self._generate_expression(node.value)
        self._emit(OpCode.SET_FIELD, node.field_name.name)

    def _generate_while_node(self, node: WhileNode):
        start_label, end_label = (
            self._new_label("while_start"),
            self._new_label("while_end"),
        )
        self._emit(start_label + ":")
        self._generate_expression(node.condition)
        self._emit(OpCode.JUMP_IF_FALSE, end_label)
        if node.body:
            for expr in node.body:
                self._generate_expression(expr)
                self._emit(OpCode.POP)
        self._emit(OpCode.JUMP, start_label)
        self._emit(end_label + ":")
        self._emit(OpCode.PUSH, None)

    def _generate_begin_node(self, node: BeginNode):
        if not node.expressions:
            self._emit(OpCode.PUSH, None)
            return
        for i, expr in enumerate(node.expressions):
            self._generate_expression(expr)
            if i < len(node.expressions) - 1:
                self._emit(OpCode.POP)

    def _generate_use_node(self, node: UseNode):
        print(f"Warning: 'use' for '{node.module_name.name}' not implemented.")
        pass


if __name__ == "__main__":
    from lexer import tokenize, LexerError
    from parser import Parser, ParserError

    print("--- NotScheme Code Generator (Quote Test Focus) ---")

    quote_tests = {
        "Simple Symbol Quote": "(print 'my_symbol)",
        "Simple List Quote": "(print '(item1 10 true nil))",
        "Nested Symbol Quote": "(print ''another_symbol)",
        "List with Nested Quote": "(print '(a 'b c))",
        "Quote of Empty List": "(print '())",
    }

    for name, code in quote_tests.items():
        print(f"\n--- Generating code for: {name} ---")
        print(f"Source: {code}")
        try:
            tokens = tokenize(code)
            parser = Parser(tokens)
            ast = parser.parse_program()

            codegen = CodeGenerator()
            bytecode = codegen.generate_program(ast)

            print("\nGenerated Bytecode:")
            for i, instruction in enumerate(bytecode):
                print(f"{i:03d}: {instruction}")

        except (LexerError, ParserError, CodeGenerationError) as e:
            print(f"Error for '{name}': {e}")
            import traceback

            traceback.print_exc()

    print("\n--- Codegen tests from previous run (condensed) ---")
    previous_tests = {
        "Static Vars": """(static a 10) (static b (+ 5 5))""",
        "List Operations": """
            (static my_list (list 1 (+ 1 1) "three")) 
            (print (first my_list))                     
            (print (rest my_list))                      
            (static my_list2 (cons 0 my_list))          
            (print my_list2)
            (print (is_nil nil))                        
            (print (is_nil my_list2))                   
        """,
    }
    for name, code in previous_tests.items():
        print(f"\n--- Generating code for: {name} ---")
        try:
            tokens = tokenize(code)
            parser = Parser(tokens)
            ast = parser.parse_program()
            codegen = CodeGenerator()
            bytecode = codegen.generate_program(ast)
            print("\nGenerated Bytecode:")
            for i, instruction in enumerate(bytecode):
                print(f"{i:03d}: {instruction}")
        except Exception as e:
            print(f"Error for '{name}': {e}")
