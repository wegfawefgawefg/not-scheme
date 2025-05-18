# Contains test cases for the Lexer.

from src.lexer import tokenize, LexerError


def run_lexer_tests():
    """Runs all lexer tests."""
    print("--- Running NotScheme Lexer Tests ---")

    test_code_1 = """
    // This is a NotScheme program
    (fn my_func (arg1 arg2)
      (let result (+ arg1 arg2)) // sum them up
      (if (> result 0)
          (print "Positive:" result)
          (print "Not positive:" result))
      result)

    (static PI 3.14)
    (struct Point (x_coord y_coord))
    (let p1 (Point 10 20))
    (set p1 x_coord (+ (get p1 x_coord) 5)) // p1.x = 15
    (print (get p1 x_coord))

    (let items '(1 "two" true nil (nested list)))
    (while false (print "looping"))
    """

    print(f"\nSource Code 1:\n{test_code_1}")
    try:
        tokens1 = tokenize(test_code_1)
        print("\nTokens 1:")
        for token in tokens1:
            print(token)
    except LexerError as e:
        print(f"Lexer Error: {e}")

    test_code_2 = """
    (let a -5)
    (let b -10.5)
    (let c "a string with \\"quotes\\" and newline\\n")
    '() // quoted empty list
    """
    print(f"\nSource Code 2:\n{test_code_2}")
    try:
        tokens2 = tokenize(test_code_2)
        print("\nTokens 2:")
        for token in tokens2:
            print(token)
    except LexerError as e:
        print(f"Lexer Error: {e}")

    test_code_3_error = "(let x &y)"  # Invalid character '&'
    print(f"\nSource Code 3 (Error Test):\n{test_code_3_error}")
    try:
        tokens3 = tokenize(test_code_3_error)
        print("\nTokens 3:")
        for token in tokens3:
            print(token)
    except LexerError as e:
        print(f"Lexer Error (Expected): {e}")

    test_code_4_multiline_string = """
    (let s "this is a
    multi-line string literal in source,
    but will be one line in token value unless \\n is used.")
    (print s)
    """
    print(f"\nSource Code 4 (Multiline String):\n{test_code_4_multiline_string}")
    try:
        tokens4 = tokenize(test_code_4_multiline_string)
        print("\nTokens 4:")
        for token in tokens4:
            print(token)
    except LexerError as e:
        print(f"Lexer Error: {e}")

    print("\n--- Lexer tests completed ---")


if __name__ == "__main__":
    # To run this file directly, ensure the project root is in PYTHONPATH
    # or run as a module: python -m src.tests.test_lexer
    run_lexer_tests()
