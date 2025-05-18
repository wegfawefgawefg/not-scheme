# Contains test cases for the Parser.

from lexer import tokenize, LexerError, TokenType, Token
from parser import Parser, ParserError # Removed print_ast from here
from ast_nodes import (
    ProgramNode,
)  # Import other AST nodes as needed by print_ast or specific tests
# import sys # sys will be imported in __main__ if needed


# Helper function to print AST (moved from parser.py)
def print_ast(node, indent=0):
    indent_str = "  " * indent
    if isinstance(node, ProgramNode):
        print(f"{indent_str}ProgramNode:")
        for form in node.forms:
            print_ast(form, indent + 1)
    elif isinstance(node, list):
        # Check if it's a list of AST nodes or a list of primitive quoted data
        if node and hasattr(
            node[0], "__dataclass_fields__"
        ):  # Simple check if elements are dataclasses
            print(f"{indent_str}[")
            for item in node:
                print_ast(item, indent + 1)
            print(f"{indent_str}]")
        else:  # Likely a list of primitive data from a quote
            print(f"{indent_str}{node!r}")

    elif hasattr(
        node, "__dataclass_fields__"
    ):  # Check if it's a dataclass (our AST nodes are)
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
            elif hasattr(field_value, "__dataclass_fields__"):  # Single AST node
                print(f"{indent_str}  {field_name}:")
                print_ast(field_value, indent + 2)
            else:  # Primitive value or list of primitives from quote
                print(f"{indent_str}  {field_name}: {field_value!r}")
    else:
        print(f"{indent_str}{node!r}")  # Primitive from quote or unhandled type


def run_parser_tests():
    """Runs all parser tests."""
    print("--- Running NotScheme Parser Tests ---")

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
    (let data '(1 foo true))
    
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
        print_ast(ast)

    except (LexerError, ParserError) as e:
        print(f"Error: {e}")

    test_code_if_error = "(if true 1)"
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
    (let y '(bar (baz 10) false))
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

    print("\n--- Parser tests completed ---")


if __name__ == "__main__":
    import sys
    import os
    # Add src directory to sys.path to allow imports of lexer, parser etc.
    # when running this test script directly.
    # Project root is two levels up from src/tests/
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    run_parser_tests()
