# Our Language NotScheme: Syntax and Examples (v15 - Loops, Lists, Std Lib Ideas)
# File Extension: .ns

## I. Syntax Overview (S-expressions)

Our language will use S-expressions. All user-defined identifiers (variables, functions, struct names, field names, module names) should use underscore_case (e.g., `my_variable`, `calculate_sum`).

1.  **Basic Structure:** (Unchanged)
2.  **Atoms:**
    * **Numbers:** (Unchanged)
    * **Strings:** (Unchanged)
    * **Symbols:** Used for variable names, function names, keywords, etc. Follow underscore_case for user definitions.
        * Examples: `x`, `my_variable`, `calculate_sum`, `Point`, `x_coord`
    * **Booleans:** `#t` for true, `#f` for false.
    * **Nil/Empty List:** `nil` (represents an empty list, also often used as a general null value).

3.  **Comments:**
    * Single-line comments start with `//`. Everything from `//` to the end of the line is ignored.

4.  **Data Structures (Literals & Constructors):**
    * **Lists:**
        * S-expressions naturally form lists.
        * **Quoted Lists (Literal Data):** `'(<item1> <item2> ...)` creates a literal list. The items are not evaluated.
            * Example: `'(1 "hello" #t)`
        * **List Constructor (Built-in/Std Lib):** `(list <item1-expr> <item2-expr> ...)` creates a list by evaluating each item expression.
            * Example: `(list (+ 1 2) "world")` results in `(3 "world")`
    * **Structs:** (Defined by `struct` special form)

5.  **Special Forms (Keywords with specific evaluation rules):**

    * **`static`**:
        * For defining module-scoped variables initialized at runtime: `(static <symbol> <expression>)`
    * **`fn`**:
        * For defining named functions (typically module-scoped): `(fn <symbol> (<param1> <param2> ...) <body_expression1> ... <body_expressionN>)`
    * **`struct`**:
        * Defines a new struct type with named fields. Structs are mutable.
        * `(struct <StructNameSymbol> (<field1_name> <field2_name> ...))`
        * Example: `(struct Point (x_coord y_coord))`
        * This implicitly defines:
            * The struct type `<StructNameSymbol>`.
            * A constructor function also named `<StructNameSymbol>` (e.g., `Point` itself becomes the constructor).
        * Field access and mutation are done using `get` and `set` special forms.
    * **`get` (for field access)**:
        * Accesses a field of a struct instance.
        * `(get <instance_expression> <field_name_symbol>)`
    * **`set` (for field mutation)**:
        * Modifies a field of a struct instance.
        * `(set <instance_expression> <field_name_symbol> <new_value_expression>)`
    * **`lambda`**:
        * For creating anonymous functions (closures): `(lambda (<param1> <param2> ...) <body_expression1> ... <body_expressionN>)`
    * **`let`**:
        * For defining lexically-scoped local variables: `(let <symbol> <expression>)`
        * A more general form for multiple local bindings (like Scheme's `let*`):
            `(let ((<sym1> <expr1>) (<sym2> <expr2>) ...) <body_expr1> ... <body_exprN>)`
            Bindings are sequential; `expr2` can use `sym1`.
    * **`if`**:
        * Conditional expression: `(if <condition_expression> <then_expression> <else_expression>)`
    * **`begin`**: (Useful for sequencing expressions)
        * `(begin <expression1> <expression2> ... <expressionN>)`
    * **`while` (Looping):**
        * Executes body expressions as long as the condition is true.
        * `(while <condition_expression> <body_expression1> ... <body_expressionN>)`
        * Returns `nil` (or the result of the last body expression on the final iteration, TBD). Let's assume `nil` for now for simplicity.
    * **`use`**: (Module-related)
        * Imports definitions from another module. The module is identified by a symbol (underscore_case), which corresponds to a filename (e.g., `<module_name_symbol>.ns`).
        * `(use <module_name_symbol> (<item_name1> <item_name2> ...))`
        * `(use <module_name_symbol> *)`

6.  **Function Calls & Struct Instantiation:**
    * General function calls: `( <function_expression> <argument1_expression> <argument2_expression> ... )`
    * Struct instantiation: `( <StructNameSymbol> <arg_for_field1> <arg_for_field2> ...)`

## II. Modules and Imports

1.  **File-based Modules:** Each `.ns` file is considered a module. The filename (without extension, underscore_case) serves as its default module name. (e.g., `my_module.ns`)
2.  **Public by Default:**
    * All top-level definitions (`static`, `fn`, `struct` type and its constructor) in a module are public by default.
3.  **Imports with `use`:**
    * `(use <module_name_symbol> (<item_name1> ...))` imports specified items.
    * `(use <module_name_symbol> *)` imports all items.
    * The compiler resolves `<module_name_symbol>` (e.g., `math_utils`) to a filename (e.g., `math_utils.ns`).

## III. Standard Library (Conceptual - To be provided by VM or core modules)

NotScheme would benefit from a core set of built-in functions and possibly standard library modules. These functions would typically be implemented in the host language (Python, for the VM) or in NotScheme itself.

**A. Core Built-in Functions/Operations (Examples):**
   (These might translate to specific VM opcodes or be primitive functions)

* **Arithmetic:** `+`, `-`, `*`, `/`, `%` (modulo)
* **Comparison:** `=`, `>`, `<`, `>=`, `<=`, `!=` (or use `(not (= a b))`)
* **Logical:** `and`, `or`, `not` (these might be special forms for short-circuiting)
* **Type Predicates:** `is_nil`, `is_boolean?`, `is_number?`, `is_string?`, `is_list?`, `is_struct?`, `is_function?`
* **List Primitives:**
    * `(cons item list)`: Constructs a new list by adding `item` to the front of `list`.
    * `(first list)`: Returns the first item of a list (Head).
    * `(rest list)`: Returns the rest of the list after the first item (Tail).
    * `(list item1 item2 ...)`: Creates a new list containing the items.
* **Basic I/O:**
    * `(print <expr1> <expr2> ...)`: Prints expressions to standard output.
    * `(read_line)`: Reads a line from standard input.
* **Error Handling:**
    * `(error <message_string>)`: Halts execution and reports an error.

**B. Potential Standard Library Modules (Example: `list_utils.ns`, `string_utils.ns`, `file_io.ns`)**

* **`list_utils.ns`:**
    * `(length list)`
    * `(append list1 list2 ...)`
    * `(map fn list)`
    * `(filter predicate_fn list)`
    * `(reduce fn initial_value list)`
    * `(nth index list)` or `(list_ref list index)`
    * `(reverse list)`
* **`string_utils.ns`:**
    * `(string_length str)`
    * `(string_append str1 str2 ...)`
    * `(substring str start_index end_index)`
    * `(string_to_chars str)`
    * `(chars_to_string char_list)`
* **`file_io.ns`:**
    * `(open_file path_string mode_string)`: Returns a file handle. `mode_string` e.g., `"r"`, `"w"`, `"a"`.
    * `(close_file file_handle)`
    * `(read_from_file file_handle num_bytes_or_line_mode)`
    * `(write_to_file file_handle string_data)`
    * `(file_exists? path_string)`
* **`math_ext.ns` (Extended Math):**
    * `(sqrt num)`
    * `(pow base exponent)`
    * `(sin angle)`, `(cos angle)`, `(tan angle)`
    * `(random)`: Returns a random float between 0 and 1.

## IV. Example Code Snippets (Incorporating New Features)

**File: `list_processing.ns`**
```ns
// list_processing.ns
(use core_primitives (print is_nil first rest cons let if + list)) // Assuming these are core

(fn sum_list (lst)
  (let sum 0)
  (let current_list lst)
  (while (not (is_nil current_list))
    (set sum (+ sum (first current_list)))
    (set current_list (rest current_list)))
  sum)

(fn map_list (func input_list)
  (if (is_nil input_list)
      nil
      (cons (func (first input_list)) (map_list func (rest input_list))))) // Recursive map

(let my_numbers (list 1 2 3 4 5))
(print "Sum:" (sum_list my_numbers)) // Expected: Sum: 15

(let doubled_numbers (map_list (lambda (x) (* x 2)) my_numbers))
(print "Doubled:" doubled_numbers) // Expected: Doubled: (2 4 6 8 10)

// Example of iterating with an index (more complex without a dedicated 'for' loop)
(fn print_list_with_index (lst)
  (let current lst)
  (let index 0)
  (while (not (is_nil current))
    (print "Index:" index "Value:" (first current))
    (set current (rest current))
    (set index (+ index 1)))
  nil)

(print_list_with_index (list "a" "b" "c"))
// Expected:
// Index: 0 Value: "a"
// Index: 1 Value: "b"
// Index: 2 Value: "c"
File: geometry.ns (Unchanged from v14, but underscore_case is consistent)// geometry.ns
(struct Point (x_coord y_coord))
(fn distance_to_origin (p)
  (let current_x (get p x_coord))
  (let current_y (get p y_coord))
  (+ (* current_x current_x) (* current_y current_y)))
(fn move_point (p dx dy)
  (set p x_coord (+ (get p x_coord) dx))
  (set p y_coord (+ (get p y_coord) dy))
  p)
File: main.ns (Updated to use underscore_case consistently)// main.ns
(use math_constants (PI))
(use geometry (Point distance_to_origin move_point))
(use list_processing (sum_list)) // Example of using another local module

(fn describe_point (p)
  (+ (get p x_coord) (get p y_coord)))

(let p1 (Point 10 20))
(let p2 (Point 3 4))
(describe_point p1)
(get p2 y_coord)
(distance_to_origin p2)
(set p1 x_coord 100)
(describe_point p1)
(move_point p2 1 1)
(describe_point p2)
(static origin_point (Point 0 0))
(describe_point origin_point)

(let numbers_to_sum (list 10 20 (get origin_point x_coord))) // Example: (list 10 20 0)
(print "Sum from main:" (sum_list numbers_to_sum)) //
