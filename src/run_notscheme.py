# run_notscheme.py
# End-to-end pipeline for lexing, parsing, compiling, and running NotScheme code.

from lexer import tokenize, LexerError
from parser import Parser, ParserError
from ast_nodes import ProgramNode
from codegen import CodeGenerator, CodeGenerationError
from vm import VirtualMachine, OpCode, QuotedSymbol

import io
import sys
import os
from typing import Any, Optional, List, Dict, Set, Tuple


class NotSchemeError(Exception):
    """Generic error for issues during the NotScheme pipeline."""

    pass


# Global caches for the compilation process
# Stores the *own* bytecode for each module once compiled.
module_own_bytecode_cache: Dict[str, List[Any]] = {}
# Stores the ordered list of all unique module names encountered in a compilation run,
# in an order that should be suitable for concatenation (dependencies first).
ordered_modules_for_linking: List[str] = []
# Cache for CodeGenerator instances to know which modules' *definitions*
# have already been extracted to avoid re-parsing files for symbol resolution.
shared_definitions_cache_for_codegen: Set[str] = set()
# Stack to detect circular dependencies during the module compilation process.
compilation_in_progress_stack: List[str] = []


def compile_all_modules_recursively(
    module_name: str,
    base_path: str,  # Base path to resolve this module_name.ns
):
    """
    Recursively ensures a module and all its dependencies are compiled.
    Populates `module_own_bytecode_cache` and `ordered_modules_for_linking`.
    Does not return bytecode directly; results are stored in global caches.
    """
    # print(f"Ensuring module compiled: {module_name} from base: {base_path}")

    if (
        module_name in module_own_bytecode_cache
    ):  # Already compiled and its deps processed
        # print(f"Module {module_name} already in bytecode cache. Skipping.")
        return

    if module_name in compilation_in_progress_stack:
        # print(f"Circular dependency detected for {module_name}. Breaking recursion for this path.")
        return

    compilation_in_progress_stack.append(module_name)

    module_file_to_open = os.path.join(base_path, f"{module_name}.ns")
    try:
        # print(f"Reading source for {module_name} from {module_file_to_open}")
        with open(module_file_to_open, "r") as f:
            source_code = f.read()
    except FileNotFoundError:
        compilation_in_progress_stack.pop()
        raise NotSchemeError(f"Module file not found: {module_file_to_open}")
    except Exception as e:
        compilation_in_progress_stack.pop()
        raise NotSchemeError(f"Error reading module file {module_file_to_open}: {e}")

    original_cwd = os.getcwd()
    try:
        os.chdir(base_path)

        tokens = tokenize(source_code)
        parser = Parser(tokens)
        ast = parser.parse_program()

        # The shared_definitions_cache helps CodeGenerator's 'use' processing
        # avoid re-parsing files just for definition extraction.
        codegen = CodeGenerator(
            processed_modules_cache=shared_definitions_cache_for_codegen
        )

        # generate_program returns the module's *own* bytecode and its direct dependencies
        own_bytecode, direct_dependencies = codegen.generate_program(
            ast, module_name=module_name
        )

        # Recursively compile dependencies first
        for dep_name in direct_dependencies:
            # For now, assume dependencies are found relative to the current module's path
            compile_all_modules_recursively(dep_name, base_path)

        # After all dependencies are processed (and thus in ordered_modules_for_linking),
        # add the current module.
        if module_name not in module_own_bytecode_cache:  # Should be true here
            module_own_bytecode_cache[module_name] = own_bytecode
            if module_name not in ordered_modules_for_linking:
                ordered_modules_for_linking.append(module_name)

    except Exception as e:
        os.chdir(original_cwd)
        if module_name in compilation_in_progress_stack:
            compilation_in_progress_stack.pop()
        raise NotSchemeError(
            f"Error during recursive compilation of module '{module_name}': {e}"
        )
    finally:
        os.chdir(original_cwd)

    if (
        module_name in compilation_in_progress_stack
    ):  # Should be true unless error before pop
        compilation_in_progress_stack.pop()


def compile_program_with_dependencies(main_file_path: str) -> List[Any]:
    """
    Compiles the main NotScheme file and all its dependencies.
    Returns a single list of aggregated bytecode.
    """
    main_module_name = os.path.splitext(os.path.basename(main_file_path))[0]
    entry_base_path = os.path.abspath(os.path.dirname(main_file_path))
    if not entry_base_path or entry_base_path == os.getcwd():
        entry_base_path = "."

    # Clear global caches for a fresh compilation run
    module_own_bytecode_cache.clear()
    ordered_modules_for_linking.clear()
    shared_definitions_cache_for_codegen.clear()
    compilation_in_progress_stack.clear()

    compile_all_modules_recursively(main_module_name, entry_base_path)

    final_bytecode: List[Any] = []
    # print(f"Final compilation order for bytecode aggregation: {ordered_modules_for_linking}")

    for module_name in ordered_modules_for_linking:
        if module_name in module_own_bytecode_cache:
            module_bc = module_own_bytecode_cache[module_name]
            # Remove HALT from dependency modules, only the main program's stream should end with one explicit HALT.
            if (
                module_name != main_module_name
                and module_bc
                and isinstance(module_bc[-1], tuple)
                and module_bc[-1] == (OpCode.HALT,)
            ):
                # print(f"Appending bytecode for {module_name} (without its HALT)")
                final_bytecode.extend(module_bc[:-1])
            else:
                # print(f"Appending bytecode for {module_name}")
                final_bytecode.extend(module_bc)
        else:
            # This should ideally not happen if logic is correct
            print(
                f"Warning: Module {module_name} was in ordered_modules_for_linking but not in module_own_bytecode_cache."
            )

    if not final_bytecode or not (
        isinstance(final_bytecode[-1], tuple)
        and final_bytecode[-1][0] in (OpCode.HALT, OpCode.RETURN, OpCode.JUMP)
    ):
        final_bytecode.append((OpCode.HALT,))
    return final_bytecode


def execute_bytecode(bytecode: list, capture_prints=False):
    vm = VirtualMachine(bytecode)
    if capture_prints:
        old_stdout, sys.stdout = sys.stdout, io.StringIO()
        try:
            result, printed_text = vm.run(), sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        return result, printed_text
    else:
        return vm.run()


def run_notscheme_test(
    test_name: str,
    source_code: str,
    main_file_name: Optional[str] = None,
    aux_files: Optional[Dict[str, str]] = None,
    expected_value: Any = None,
    expect_error: bool = False,
    expected_prints: Optional[List[Any]] = None,
):
    print(f"\n--- Test: {test_name} ---")

    created_files = []
    test_base_path = os.getcwd()

    if aux_files:
        for fname, fcontent in aux_files.items():
            full_aux_path = os.path.join(test_base_path, fname)
            aux_dir = os.path.dirname(full_aux_path)
            if aux_dir and not os.path.exists(aux_dir):
                os.makedirs(aux_dir)
            with open(full_aux_path, "w") as f:
                f.write(fcontent)
            created_files.append(full_aux_path)

    entry_point_filename_for_test = (
        main_file_name
        if main_file_name
        else f"{test_name.lower().replace(' ', '_')}_main.ns"
    )
    full_entry_point_path = os.path.join(test_base_path, entry_point_filename_for_test)

    entry_dir = os.path.dirname(full_entry_point_path)
    if entry_dir and not os.path.exists(entry_dir):
        os.makedirs(entry_dir)

    with open(full_entry_point_path, "w") as f:
        f.write(source_code)
    created_files.append(full_entry_point_path)

    # print(f"Main file for test: {full_entry_point_path}")
    # print("Source (for main file):\n" + "-" * 10 + f"\n{source_code}\n" + "-" * 10)
    # if aux_files:
    #     for fname, fcontent in aux_files.items():
    #         print(f"Auxiliary file: {fname}\n" + "-"*10 + f"\n{fcontent}\n" + "-"*10)

    actual_prints_text = ""
    try:
        final_bytecode = compile_program_with_dependencies(full_entry_point_path)

        print("Final Aggregated Bytecode:")
        for i, instruction in enumerate(final_bytecode):
            print(f"  {i:03d}: {instruction}")

        if expected_prints is not None:
            result, actual_prints_text = execute_bytecode(
                final_bytecode, capture_prints=True
            )
            print("Captured Output:")
            if actual_prints_text:
                print(actual_prints_text, end="")
            else:
                print("<no output>")
        else:
            result = execute_bytecode(final_bytecode)

        if expect_error:
            print(
                f"FAIL: Expected an error, but execution succeeded with result: {result}"
            )
        else:
            if expected_prints is not None:
                actual_print_lines = [
                    l
                    for l in actual_prints_text.split("\n")
                    if l.strip() and l.strip() != "Execution halted."
                ]
                formatted_expected_prints = []
                for p_val in expected_prints:
                    if isinstance(p_val, QuotedSymbol):
                        formatted_expected_prints.append(f"Output: {p_val!r}")
                    elif isinstance(p_val, list):
                        list_content_str = ", ".join(repr(item) for item in p_val)
                        formatted_expected_prints.append(
                            f"Output: [{list_content_str}]"
                        )
                    elif p_val is True:
                        formatted_expected_prints.append("Output: True")
                    elif p_val is False:
                        formatted_expected_prints.append("Output: False")
                    elif p_val is None:
                        formatted_expected_prints.append("Output: None")
                    else:
                        formatted_expected_prints.append(f"Output: {p_val}")
                if actual_print_lines == formatted_expected_prints:
                    print(f"Prints: PASS")
                else:
                    print(f"Prints: FAIL")
                    print(f"  Expected: {formatted_expected_prints}")
                    print(f"  Actual  : {actual_print_lines}")
            if expected_value is not None or not expected_prints:
                if result == expected_value:
                    print(f"Result: PASS (Expected: {expected_value}, Got: {result})")
                else:
                    print(f"Result: FAIL (Expected: {expected_value}, Got: {result})")
            elif expected_prints and expected_value is None:
                if result is None:
                    print(f"Result: PASS (Expected: None, Got: {result})")
                else:
                    print(
                        f"Result: UNEXPECTED (Expected None after HALT, Got: {result})"
                    )
    except (NotSchemeError, Exception) as e:
        if expect_error:
            print(f"PASS: Caught expected error: {e}")
        else:
            print(f"FAIL: Unexpected error: {e}")
            import traceback

            traceback.print_exc()
    finally:
        for fname in created_files:
            if os.path.exists(fname):
                try:
                    os.remove(fname)
                except OSError as e_os:
                    print(f"Warning: could not remove test file {fname}: {e_os}")
        if aux_files:
            for fname in aux_files.keys():
                aux_dir = os.path.dirname(os.path.join(test_base_path, fname))
                if (
                    aux_dir
                    and aux_dir != test_base_path
                    and os.path.exists(aux_dir)
                    and not os.listdir(aux_dir)
                ):
                    try:
                        os.rmdir(aux_dir)
                    except OSError as e_os:
                        print(f"Warning: could not remove test dir {aux_dir}: {e_os}")
        entry_dir = os.path.dirname(full_entry_point_path)
        if (
            entry_dir
            and entry_dir != test_base_path
            and os.path.exists(entry_dir)
            and not os.listdir(entry_dir)
        ):
            try:
                os.rmdir(entry_dir)
            except OSError as e_os:
                print(f"Warning: could not remove test dir {entry_dir}: {e_os}")
    print("-" * 20)


if __name__ == "__main__":
    # --- Single File Tests ---
    run_notscheme_test(
        "Static Vars", "(static a 10)(static b (+ a 5)) b", expected_value=15
    )
    run_notscheme_test(
        "Function Def & Call",
        "(fn add (x y) (+ x y))(static r (add 10 20)) r",
        expected_value=30,
    )
    run_notscheme_test(
        "Print Test",
        """(print "Hello")(print 123)(print true)(print nil)(+ 1 1)""",
        expected_value=2,
        expected_prints=["Hello", 123, True, None],
    )
    run_notscheme_test(
        "List Operations Test",
        """
        (static my_list (list 1 (+ 1 1) "three")) 
        (print (first my_list))                     
        (print (rest my_list))                      
        (static my_list2 (cons 0 my_list))          
        (print my_list2)
        (print (is_nil nil))                        
        (print (is_nil my_list2))                   
        (first (list "final"))                      
        """,
        expected_value="final",
        expected_prints=[1, [2, "three"], [0, 1, 2, "three"], True, False],
    )
    run_notscheme_test(
        "Quote: Simple Symbol",
        "(print 'my_symbol)",
        expected_value=None,
        expected_prints=[QuotedSymbol(name="my_symbol")],
    )
    run_notscheme_test(
        "Quote: Simple List",
        "(print '(item1 10 true nil))",
        expected_value=None,
        expected_prints=[[QuotedSymbol(name="item1"), 10, True, None]],
    )

    # --- Multi-File Module System Tests ---
    math_utils_content = """
    // math_utils.ns
    (struct Vec2 (x y))
    (static gravity 9.8)
    (fn square (val) (* val val))
    (fn add_vec (v1 v2) 
        (Vec2 (+ (get v1 x) (get v2 x)) 
              (+ (get v1 y) (get v2 y))))
    """

    string_ext_content_safe = """
    // string_ext.ns
    (static greeting "Hello from string_ext module")
    (fn get_greeting () greeting) 
    """

    main_specific_import_content = """
    // main_specific.ns
    (use math_utils (gravity square Vec2 add_vec))
    (static my_g gravity)
    (static nine (square 3))
    (static v1 (Vec2 1 2))
    (static v2 (Vec2 3 4))
    (static v_sum (add_vec v1 v2))
    (get v_sum x) 
    """
    run_notscheme_test(
        "Module: Use Specific Items",
        source_code=main_specific_import_content,
        main_file_name="main_specific.ns",
        aux_files={"math_utils.ns": math_utils_content},
        expected_value=4,
    )

    main_all_import_content = """
    // main_all.ns
    (use string_ext *)
    (use math_utils (square Vec2)) 
    (print greeting)
    (print (square (get (Vec2 3 0) x))) 
    (get_greeting) 
    """
    run_notscheme_test(
        "Module: Use All Items (*)",
        source_code=main_all_import_content,
        main_file_name="main_all.ns",
        aux_files={
            "string_ext.ns": string_ext_content_safe,
            "math_utils.ns": math_utils_content,
        },
        expected_value="Hello from string_ext module",
        expected_prints=["Hello from string_ext module", 9],
    )

    module_a_content = """
    // module_a.ns
    (use module_b (b_val get_b_internal)) 
    (static a_val 10)
    (fn get_a () a_val)
    (fn call_b_from_a () (get_b_internal)) 
    """
    module_b_content = """
    // module_b.ns
    (use module_a (a_val)) 
    (static b_val (+ a_val 20)) 
    (fn get_b_internal () b_val)
    (fn get_b_direct () b_val) 
    """
    main_circular_content = """
    // main_circular.ns
    (use module_a (call_b_from_a get_a)) // get_a is not used, but tests import
    (use module_b (get_b_direct))     // get_b_direct is not used, but tests import
    (call_b_from_a)                   // This is the actual call we test the result of
    """
    run_notscheme_test(
        "Module: Circular Use for Definitions",
        source_code=main_circular_content,
        main_file_name="main_circular.ns",
        aux_files={"module_a.ns": module_a_content, "module_b.ns": module_b_content},
        expected_value=30,
    )

    print("\n--- All NotScheme end-to-end tests completed ---")
