import re
from enum import Enum  # Removed auto
from collections import namedtuple


class TokenType(Enum):
    LPAREN = 1
    RPAREN = 2
    QUOTE = 3
    SYMBOL = 4
    NUMBER = 5
    STRING = 6
    BOOLEAN = 7
    NIL = 8
    COMMENT = 9
    EOF = 10


# Token named tuple to store type, value, and optionally line/column numbers
Token = namedtuple("Token", ["type", "value", "line", "col"])

# Regular expressions for token patterns
# Order matters: keywords like 'nil' should be checked before general SYMBOL
# to avoid 'nil' being tokenized as SYMBOL("nil") if we were to distinguish.
# However, for S-expressions, keywords are often just symbols until parsing.
# Let's keep it simple: 'nil', 'true', 'false' will be specific token types.
# All other non-numeric/non-string sequences are SYMBOLS.
TOKEN_REGEX_PATTERNS = [
    (r"//[^\n]*", TokenType.COMMENT),  # Comments (ignore)
    (r"\(", TokenType.LPAREN),
    (r"\)", TokenType.RPAREN),
    (r"'", TokenType.QUOTE),  # For '() lists
    (r"\btrue\b|\bfalse\b", TokenType.BOOLEAN),
    (r"nil", TokenType.NIL),
    (r'"(?:\\.|[^"\\])*"', TokenType.STRING),  # Strings with escape sequences
    # Numbers: integers, floats, negative numbers
    # This regex handles integers and floats, including negative ones.
    # It prioritizes floats if a '.' is present.
    (r"-?\d+\.\d+", TokenType.NUMBER),  # Floats (e.g., 3.14, -0.5)
    (r"-?\d+", TokenType.NUMBER),  # Integers (e.g., 10, -5)
    # Symbols: must start with a letter or specific non-numeric chars,
    # can contain letters, numbers, underscores, hyphens, ?, !, etc.
    # For NotScheme, we're aiming for underscore_case.
    # This regex is quite permissive for symbols; strict keyword matching happens later if needed.
    # It also includes operators like +, *, >= etc.
    (r"[a-zA-Z_+\-*/%<>=!?][a-zA-Z0-9_+\-*/%<>=!?]*", TokenType.SYMBOL),
]


class LexerError(Exception):
    pass


def tokenize(source_code: str) -> list[Token]:
    """
    Converts NotScheme source code into a list of tokens.
    """
    tokens = []
    line_num = 1
    col_num = 1
    position = 0
    source_len = len(source_code)

    while position < source_len:
        # Update column for current position before matching
        # This finds the start of the current line to calculate col_num correctly
        current_line_start = source_code.rfind("\n", 0, position)
        if current_line_start == -1:  # First line
            current_line_start = 0
            col_num = position + 1
        else:  # Subsequent lines
            # rfind gives index of '\n', so start of current line is one after
            col_num = position - current_line_start

        match = None
        # Try to match each regex pattern
        for pattern, token_type in TOKEN_REGEX_PATTERNS:
            regex = re.compile(pattern)
            m = regex.match(source_code, position)
            if m:
                value = m.group(0)
                match_end = m.end()

                if token_type == TokenType.COMMENT:
                    # Skip comments, advance position
                    # Handle newlines within or after comment
                    newline_count_in_comment = value.count("\n")
                    if newline_count_in_comment > 0:
                        line_num += newline_count_in_comment
                        # col_num will be reset at the start of the loop
                    position = match_end
                    match = True  # Found a comment, break from pattern loop
                    break

                if token_type == TokenType.STRING:
                    # Store string content without the surrounding quotes
                    # Also, handle basic escape sequences (more can be added)
                    processed_value = (
                        value[1:-1]
                        .replace('\\"', '"')
                        .replace("\\n", "\n")
                        .replace("\\t", "\t")
                        .replace("\\\\", "\\")
                    )
                    tokens.append(Token(token_type, processed_value, line_num, col_num))
                elif token_type == TokenType.NUMBER:
                    # Convert to int or float
                    try:
                        if "." in value:
                            num_val = float(value)
                        else:
                            num_val = int(value)
                        tokens.append(Token(token_type, num_val, line_num, col_num))
                    except ValueError:
                        raise LexerError(
                            f"Invalid number format: {value} at line {line_num}, col {col_num}"
                        )
                elif token_type == TokenType.BOOLEAN:
                    tokens.append(
                        Token(token_type, value == "true", line_num, col_num)
                    )  # Store as Python bool
                elif token_type == TokenType.NIL:
                    tokens.append(
                        Token(token_type, None, line_num, col_num)
                    )  # Store as Python None
                else:  # LPAREN, RPAREN, QUOTE, SYMBOL
                    tokens.append(Token(token_type, value, line_num, col_num))

                # Update position and line/col for next token
                newline_count = value.count("\n")
                if newline_count > 0:
                    line_num += newline_count
                    # col_num will be reset correctly at the start of the loop

                position = match_end
                match = True  # Found a token, break from pattern loop
                break  # Found a match, go to next position in source

        if not match:
            # If it's whitespace, skip it
            if source_code[position].isspace():
                if source_code[position] == "\n":
                    line_num += 1
                position += 1
            else:
                # No pattern matched and not whitespace: unknown character
                raise LexerError(
                    f"Unexpected character: '{source_code[position]}' at line {line_num}, col {col_num}"
                )

    tokens.append(Token(TokenType.EOF, None, line_num, 1))  # Add End Of File token
    return tokens
