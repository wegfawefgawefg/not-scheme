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
from vm import OpCode  # Assuming vm.py is accessible
from typing import List, Dict, Any, Tuple, Union, get_args

# Define a tuple of all concrete AST node types that are considered Expressions
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


# Mapping of NotScheme primitive symbols to VM OpCodes and expected arity
# Arity of -1 means variable number of arguments (e.g. print) - special handling needed
# Arity of None means the opcode itself handles stack (like ADD, SUB etc. pop 2)
PRIMITIVE_OPERATIONS = {
    "+": (OpCode.ADD, None),
    "-": (OpCode.SUB, None),
    "*": (OpCode.MUL, None),
    "/": (OpCode.DIV, None),
    # "%": (OpCode.MOD, None), # Assuming MOD opcode exists or will be added
    "=": (OpCode.EQ, None),
    ">": (OpCode.GT, None),
    "<": (OpCode.LT, None),
    # ">=": (OpCode.GTE, None),
    # "<=": (OpCode.LTE, None),
    # "!=": (OpCode.NEQ, None), # Or (not (= ..))
    "not": (OpCode.NOT, None),  # Expects 1 arg on stack
    "print": (
        OpCode.PRINT,
        -1,
    ),  # Special: print pops 1, but can be called with multiple args
    # For now, let's assume (print expr) -> PUSH expr, PRINT
    # If (print e1 e2), we'd need multiple PRINT opcodes or a loop.
    # Let's simplify: (print e) -> gen(e), PRINT
    # If multiple args to print, parser makes CallNode with multiple args.
    # Codegen will handle this by emitting multiple PRINTs.
    # List primitives might be more complex if they aren't direct opcodes
    # "list": (OpCode.MAKE_LIST_OP, -1) # Example, if we had such an opcode
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
            if "vars" not in self.compile_time_env_chain[-1]:
                self.compile_time_env_chain[-1]["vars"] = set()
            self.compile_time_env_chain[-1][name] = "local_variable"
        else:
            print(
                f"Warning: Could not add '{name}' to current compile-time local scope. Current scope type: {self.compile_time_env_chain[-1]['type'] if self.compile_time_env_chain else 'None'}"
            )

    def generate_program(self, program_node: ProgramNode) -> List[Union[tuple, str]]:
        self.bytecode = []
        self.global_env.clear()
        self.struct_definitions.clear()
        self.compile_time_env_chain = [{"type": "global"}]

        # Process top-level forms. We might need multiple passes for `use` or forward declarations.
        # For now, a single pass.
        for i, form in enumerate(program_node.forms):
            is_last_form = i == len(program_node.forms) - 1
            self._generate_top_level_form(form, is_last_form_in_program=is_last_form)

        # Ensure HALT is the very last instruction if not already terminated by other means.
        if not self.bytecode or not (
            isinstance(self.bytecode[-1], tuple) and self.bytecode[-1][0] == OpCode.HALT
        ):
            # If the last generated instruction was RETURN (e.g. from a top-level function call that's the whole program)
            # then we don't need an explicit HALT.
            if not (
                self.bytecode
                and isinstance(self.bytecode[-1], tuple)
                and self.bytecode[-1][0] == OpCode.RETURN
            ):
                self._emit(OpCode.HALT)
        return self.bytecode

    def _generate_top_level_form(
        self, form: TopLevelForm, is_last_form_in_program: bool = False
    ):
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
            # If a top-level expression's result is not used and it's not the last form, POP it.
            if not is_last_form_in_program:
                # This is a simplification. If the expression has no side effects and its value
                # isn't stored or printed, it should be popped.
                # Example: (fn foo () 10) (foo) (static x 5) -> (foo) result should be popped.
                # If the expression is the *very last thing* in the program, its value is the program's result.
                self._emit(OpCode.POP)
        else:
            raise CodeGenerationError(f"Unsupported top-level form: {type(form)}")

    # --- Expression Generators ---
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
            self._generate_quoted_expression(expr_node.expression)
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
            raise CodeGenerationError(
                f"Unsupported expression type in _generate_expression: {type(expr_node)}"
            )

    def _generate_quoted_expression(self, quoted_data: Any):
        if isinstance(quoted_data, SymbolNode):
            self._emit(OpCode.PUSH, f"'{quoted_data.name}")
        elif isinstance(quoted_data, list):
            temp_list_repr = []
            for item in quoted_data:
                if isinstance(item, SymbolNode):
                    temp_list_repr.append(f"'{item.name}")
                elif isinstance(item, list):
                    temp_list_repr.append(self._generate_quoted_expression_value(item))
                elif isinstance(item, QuoteNode):
                    temp_list_repr.append(self._generate_quoted_expression_value(item))
                else:
                    temp_list_repr.append(item)
            self._emit(OpCode.PUSH, tuple(temp_list_repr))
        elif isinstance(quoted_data, QuoteNode):
            self._emit(OpCode.PUSH, self._generate_quoted_expression_value(quoted_data))
        else:
            self._emit(OpCode.PUSH, quoted_data)

    def _generate_quoted_expression_value(self, data: Any) -> Any:
        if isinstance(data, SymbolNode):
            return f"'{data.name}"
        elif isinstance(data, list):
            return tuple(self._generate_quoted_expression_value(item) for item in data)
        elif isinstance(data, QuoteNode):
            return (
                SymbolNode(name="quote").name,
                self._generate_quoted_expression_value(data.expression),
            )
        return data

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
        # Check for primitive operations first
        if isinstance(node.callable_expr, SymbolNode):
            op_name = node.callable_expr.name
            if op_name in PRIMITIVE_OPERATIONS:
                opcode, arity = PRIMITIVE_OPERATIONS[op_name]

                # Handle arguments for primitives that take them from stack (like ADD, SUB)
                # These opcodes typically expect args to be pushed in a specific order.
                # For binary ops like (+ a b), 'a' then 'b' is pushed. Opcode pops b then a.
                if (
                    arity is None
                ):  # Opcode handles its own args from stack (e.g. ADD, GT)
                    if len(node.arguments) != 2 and op_name not in [
                        "not"
                    ]:  # most binary ops
                        if op_name == "not" and len(node.arguments) != 1:
                            raise CodeGenerationError(
                                f"Primitive '{op_name}' expects 1 argument, got {len(node.arguments)}"
                            )
                        elif op_name != "not":
                            raise CodeGenerationError(
                                f"Primitive '{op_name}' expects 2 arguments, got {len(node.arguments)}"
                            )

                    # Arguments are pushed left-to-right by the loop below
                    for arg_expr in node.arguments:
                        self._generate_expression(arg_expr)
                    self._emit(opcode)
                    return

                # Handle 'print' which can take multiple arguments
                elif op_name == "print":
                    if not node.arguments:  # (print) with no args
                        # Could print newline or be an error. Let's say it prints newline (by pushing None then PRINT)
                        # Or better, make (print) without args an error or no-op.
                        # For now, let's assume (print) is an error, caught by arity check if we set arity > 0.
                        # Let's assume (print expr) is the main form.
                        # If (print e1 e2), codegen will emit gen(e1), PRINT, gen(e2), PRINT
                        pass  # Will be handled by loop below
                    for arg_expr in node.arguments:
                        self._generate_expression(arg_expr)
                        self._emit(opcode)  # Emit PRINT for each argument
                    # 'print' itself doesn't leave a value on stack, VM's PRINT opcode handles that.
                    # If (print) is the last expression, its "value" could be nil.
                    # We need to push something if it's not the last expression in a sequence.
                    # For now, let's assume print is like a statement.
                    # To make (print a b) work as a single conceptual call leaving one value (e.g. nil):
                    # This is complex. For now: multiple PRINT opcodes.
                    # The 'print' function should probably return nil or its last argument.
                    # Let's assume print returns nil.
                    self._emit(
                        OpCode.PUSH, None
                    )  # Result of a (print ...) expression is nil
                    return
                # Add other arity checks as needed

            # Check for struct instantiation
            elif op_name in self.struct_definitions:
                struct_name = op_name
                field_names = self.struct_definitions[struct_name]
                if len(node.arguments) != len(field_names):
                    raise CodeGenerationError(
                        f"Struct '{struct_name}' constructor called with {len(node.arguments)} arguments, "
                        f"but expected {len(field_names)}."
                    )
                for arg_expr in node.arguments:
                    self._generate_expression(arg_expr)
                self._emit(OpCode.MAKE_STRUCT, struct_name, tuple(field_names))
                return

        # Default: Regular user-defined function call or lambda call
        for arg_expr in node.arguments:
            self._generate_expression(arg_expr)
        self._generate_expression(
            node.callable_expr
        )  # Closure should be on top of args
        self._emit(OpCode.CALL, len(node.arguments))

    def _generate_if_node(self, node: IfNode):
        else_label = self._new_label("else")
        end_if_label = self._new_label("end_if")
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
        # SET_FIELD in VM pushes the modified instance.
        # Per common Lisp conventions, set! often returns the value assigned, or unspecified.
        # If we want it to return the value: POP instance, PUSH value (already on stack before instance).
        # For now, it returns the modified instance (as per VM). Let's POP it and PUSH nil
        # to make (set ...) evaluate to nil, or the value.
        # Let's decide (set ...) evaluates to the value that was set.
        # Stack after SET_FIELD: [modified_instance]
        # Value was on stack before instance. We need to preserve it or re-push.
        # Simpler: (set ...) returns the modified instance. If value is needed, do (get ...) after.
        # Current: SET_FIELD pushes instance. If (set p f v) is not last expr in seq, it should be POPped.

    def _generate_while_node(self, node: WhileNode):
        loop_start_label = self._new_label("while_start")
        loop_end_label = self._new_label("while_end")

        self._emit(loop_start_label + ":")
        self._generate_expression(node.condition)
        self._emit(OpCode.JUMP_IF_FALSE, loop_end_label)

        if not node.body:
            pass
        else:
            for i, expr in enumerate(node.body):
                self._generate_expression(expr)
                self._emit(
                    OpCode.POP
                )  # Results of while body expressions are discarded

        self._emit(OpCode.JUMP, loop_start_label)
        self._emit(loop_end_label + ":")
        self._emit(OpCode.PUSH, None)  # While loop evaluates to nil

    def _generate_begin_node(self, node: BeginNode):
        if not node.expressions:
            self._emit(OpCode.PUSH, None)
            return

        for i, expr in enumerate(node.expressions):
            self._generate_expression(expr)
            if i < len(node.expressions) - 1:
                self._emit(OpCode.POP)

    def _generate_use_node(self, node: UseNode):
        print(
            f"Warning: 'use' statement for module '{node.module_name.name}' encountered. Module processing not yet implemented."
        )
        pass


if __name__ == "__main__":
    from lexer import tokenize, LexerError
    from parser import Parser, ParserError

    print("--- NotScheme Code Generator ---")

    # Test cases (ensure they are valid NotScheme according to the spec)
    test_code_static = """(static a 10) (static b (+ 5 5))"""
    test_code_fn_def = """(fn add (x y) (+ x y)) (static result (add 10 20))"""
    test_code_struct_def = (
        """(struct Point (x_coord y_coord)) (static p1 (Point 1 2))"""
    )
    test_code_if_expr = """(static x 10) (static if_result (if (> x 5) 100 200))"""
    test_code_let_expr = """
    (fn test_let ()
        (let ((a 10) (b (+ a 5))) 
            (+ a b)))
    (test_let) 
    """
    test_code_lambda_expr = """
    (static my_adder ((lambda (val_to_add) (lambda (x) (+ x val_to_add))) 5))
    (my_adder 10) 
    """

    test_code_get_set_in_fn = """
    (struct Pair (first second)) 
    (fn test_get_set ()
      (let p (Pair 10 20))
      (set p second 30) // set returns the modified struct, which might be popped if not last
      (get p second)    // This is the actual return value of the function
    )
    (test_get_set)
    """

    test_code_while_correct_set = """
    (struct Counter (value current_sum))
    (fn test_while ()
      (let c (Counter 0 0))
      (while (< (get c value) 3)
        (set c current_sum (+ (get c current_sum) (get c value)))
        (set c value (+ (get c value) 1))
      )
      (get c current_sum) 
    )
    (test_while)
    """

    test_code_begin_expr = """
    (begin 
        (let temp 5) 
        (+ temp 10)) 
    """

    test_code_print = """
    (print "Hello") 
    (print (+ 10 20))
    """

    tests = {
        "Static Vars": test_code_static,
        "Function Def & Call": test_code_fn_def,
        "Struct Def & Instance": test_code_struct_def,
        "If Expression": test_code_if_expr,
        "Let Expression": test_code_let_expr,
        "Lambda Expression": test_code_lambda_expr,
        "Get/Set Struct Fields (in Fn)": test_code_get_set_in_fn,
        "While Loop (Corrected Set)": test_code_while_correct_set,
        "Begin Expression": test_code_begin_expr,
        "Print Expressions": test_code_print,
    }

    for name, code in tests.items():
        print(f"\n--- Generating code for: {name} ---")
        # print(code)
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

    test_code_unsupported_expr = """
    (use my_module (something)) 
    """
    print(f"\n--- Generating code for: Unsupported (UseNode) ---")
    try:
        tokens = tokenize(test_code_unsupported_expr)
        parser = Parser(tokens)
        ast = parser.parse_program()
        codegen = CodeGenerator()
        bytecode = codegen.generate_program(ast)
    except (LexerError, ParserError, CodeGenerationError) as e:
        print(
            f"Caught expected warning/behavior for 'UseNode' (or error if strict): {e}"
        )
