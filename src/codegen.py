"""
code generation
    converts the AST into bytecode targetting the vm
"""

from typing import List, Dict, Any, Tuple, Union, Set, Optional

from src.ast_nodes import (
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
    LetNode,
    LambdaNode,
    GetNode,
    SetNode,
    WhileNode,
    BeginNode,
    Expression,
    TopLevelForm,
)
from src.vm import OpCode, QuotedSymbol

# EXPRESSION_NODE_TYPES was unused, replaced with inline tuple below
# EXPRESSION_NODE_TYPES = (
#     NumberNode,
#     StringNode,
#     BooleanNode,
# (NilNode,)
# (SymbolNode,)
# (QuoteNode,)
# (CallNode,)
# (IfNode,)
# (LetNode,)
# (LambdaNode,)
# (GetNode,)
# (SetNode,)
# (WhileNode,)
# (BeginNode,)
# )


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
    "is_boolean": (OpCode.IS_BOOLEAN, None),
    "is_number": (OpCode.IS_NUMBER, None),
    "is_string": (OpCode.IS_STRING, None),
    "is_list": (OpCode.IS_LIST, None),
    "is_struct": (OpCode.IS_STRUCT, None),
    "is_function": (OpCode.IS_FUNCTION, None),
}


class CodeGenerator:
    def __init__(self, processed_modules_cache: Optional[Set[str]] = None):
        self.bytecode: List[Union[tuple, str]] = []
        self.label_count = 0
        self.global_env: Dict[str, Any] = {}
        self.struct_definitions: Dict[str, List[str]] = {}
        self.compile_time_env_chain: List[Dict[str, str]] = [{"type": "global"}]
        self.processed_modules_for_defs: Set[str] = (
            processed_modules_cache if processed_modules_cache is not None else set()
        )
        self.current_module_name_for_labels: str = "__main__"
        self.discovered_dependencies: Set[str] = set()

    def _new_label(self, prefix="L") -> str:
        self.label_count += 1
        return f"{self.current_module_name_for_labels}_{prefix}{self.label_count}"

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

    def generate_program(
        self, program_node: ProgramNode, module_name: str = "__main__"
    ) -> Tuple[List[Union[tuple, str]], Set[str]]:
        self.bytecode = []
        self.discovered_dependencies = set()
        self.current_module_name_for_labels = (
            module_name.replace(".ns", "").replace("/", "_").replace("\\", "_")
        )

        use_forms = [form for form in program_node.forms if isinstance(form, UseNode)]
        for form in use_forms:
            self._generate_use_node(form)

        other_forms = [
            form for form in program_node.forms if not isinstance(form, UseNode)
        ]
        for i, form in enumerate(other_forms):
            is_last_exec_form = i == len(other_forms) - 1
            self._generate_top_level_form(
                form, is_last_form_in_program=is_last_exec_form
            )

        return self.bytecode, self.discovered_dependencies

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
        elif isinstance(
            form,
            (
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
            ),
        ):
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
            self._generate_quote_node(expr_node)
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

    def _generate_quote_node(self, node: QuoteNode):
        self._generate_runtime_value_for_quoted_item(node.expression)

    def _generate_runtime_value_for_quoted_item(self, item_data: Any):
        if isinstance(item_data, SymbolNode):
            self._emit(OpCode.PUSH, QuotedSymbol(name=item_data.name))
        elif isinstance(item_data, list):
            for sub_item in item_data:
                self._generate_runtime_value_for_quoted_item(sub_item)
            self._emit(OpCode.MAKE_LIST, len(item_data))
        elif isinstance(item_data, QuoteNode):
            self._emit(OpCode.PUSH, QuotedSymbol(name="quote"))
            self._generate_runtime_value_for_quoted_item(item_data.expression)
            self._emit(OpCode.MAKE_LIST, 2)
        elif isinstance(item_data, (int, float, str, bool)) or item_data is None:
            self._emit(OpCode.PUSH, item_data)
        else:
            raise CodeGenerationError(
                f"Cannot generate runtime value for quoted item of type: {type(item_data)}, value: {item_data!r}"
            )

    def _generate_static_node(self, node: StaticNode):
        self._generate_expression(node.value)
        self._emit(OpCode.STORE, node.name.name)
        self.global_env[node.name.name] = {
            "type": "static_variable",
            "value_type": type(node.value).__name__,
        }

    def _generate_fn_or_lambda_body(
        self,
        name_for_label: str,
        params: List[SymbolNode],
        body: List[Expression],
        is_named_fn: bool,
    ):
        label_prefix = f"{self.current_module_name_for_labels}_{'fn' if is_named_fn else 'lambda'}_{name_for_label}"
        entry_label = self._new_label(label_prefix)
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
        end_body_label = self._new_label(f"end_{label_prefix}")
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
            if self.struct_definitions[node.name.name] != field_names:
                raise CodeGenerationError(
                    f"Struct '{node.name.name}' already defined with different fields."
                )
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
                    if op_name in ["+", "-", "*", "/", "=", ">", "<", "cons"]:
                        expected_arity = 2
                    elif op_name in [
                        "not",
                        "is_nil",
                        "first",
                        "rest",
                        "is_boolean",
                        "is_number",
                        "is_string",
                        "is_list",
                        "is_struct",
                        "is_function",
                    ]:
                        expected_arity = 1
                    if len(node.arguments) != expected_arity:
                        raise CodeGenerationError(
                            f"Primitive '{op_name}' expects {expected_arity} args, got {len(node.arguments)}"
                        )
                    if op_name == "cons":
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
            elif (
                op_name in self.global_env
                and isinstance(self.global_env[op_name], dict)
                and self.global_env[op_name].get("type") == "struct_type"
            ):
                struct_info = self.global_env[op_name]
                struct_name, field_names = op_name, struct_info["fields"]
                if len(node.arguments) != len(field_names):
                    raise CodeGenerationError(
                        f"Imported struct '{struct_name}': expected {len(field_names)} args, got {len(node.arguments)}."
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
        module_name_to_load = node.module_name.name
        self.discovered_dependencies.add(module_name_to_load)

        if module_name_to_load in self.processed_modules_for_defs:
            return
        self.processed_modules_for_defs.add(module_name_to_load)

        try:
            module_file_path = f"{module_name_to_load}.ns"
            with open(module_file_path, "r") as f:
                source_code = f.read()
        except FileNotFoundError:
            raise CodeGenerationError(f"Module file not found: {module_file_path}")
        except Exception as e:
            raise CodeGenerationError(
                f"Error reading module file {module_file_path}: {e}"
            )

        from src.lexer import tokenize as dep_tokenize, LexerError as DepLexerError
        from src.parser import Parser as DepParser, ParserError as DepParserError

        try:
            dep_tokens = dep_tokenize(source_code)
            dep_parser = DepParser(dep_tokens)
            dep_ast = dep_parser.parse_program()
        except (DepLexerError, DepParserError) as e:
            raise CodeGenerationError(
                f"Error compiling used module '{module_name_to_load}' for definitions: {e}"
            )

        temp_dep_codegen = CodeGenerator(
            processed_modules_cache=self.processed_modules_for_defs
        )
        _, further_deps = temp_dep_codegen.generate_program(
            dep_ast, module_name=module_name_to_load
        )
        for dep in further_deps:
            self.discovered_dependencies.add(dep)

        items_to_import_names: List[str] = []
        import_all = isinstance(node.items, SymbolNode) and node.items.name == "*"

        if import_all:
            items_to_import_names.extend(temp_dep_codegen.global_env.keys())
            items_to_import_names.extend(temp_dep_codegen.struct_definitions.keys())
            items_to_import_names = list(set(items_to_import_names))
        elif isinstance(node.items, list):
            items_to_import_names = [item.name for item in node.items]

        for item_name in items_to_import_names:
            imported_something = False
            if item_name in temp_dep_codegen.global_env:
                self.global_env[item_name] = temp_dep_codegen.global_env[item_name]
                imported_something = True
            if item_name in temp_dep_codegen.struct_definitions:
                self.struct_definitions[item_name] = (
                    temp_dep_codegen.struct_definitions[item_name]
                )
                if item_name not in self.global_env:
                    self.global_env[item_name] = {
                        "type": "struct_type",
                        "fields": temp_dep_codegen.struct_definitions[item_name],
                    }
                imported_something = True
            if not imported_something and not import_all:
                print(
                    f"Warning: Item '{item_name}' in '(use {module_name_to_load} ...)' not found in module '{module_name_to_load}'."
                )
