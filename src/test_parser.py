# Contains test cases for the Parser.

from lexer import tokenize, LexerError, TokenType, Token
from parser import Parser, ParserError, print_ast
from ast_nodes import (
    ProgramNode,
)  # Import other AST nodes as needed by print_ast or specific tests
import sys  # For print_ast if it uses sys


# Helper function to print AST (copied from parser.py)


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
    run_parser_tests()
