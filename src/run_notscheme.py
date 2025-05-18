# run_notscheme.py
# End-to-end pipeline for lexing, parsing, compiling, and running NotScheme code.
# Can be used as a CLI to run .ns files or to run internal tests.

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
module_own_bytecode_cache: Dict[str, List[Any]] = {}
ordered_modules_for_linking: List[str] = []
shared_definitions_cache_for_codegen: Set[str] = set()
compilation_in_progress_stack: List[str] = []


def compile_all_modules_recursively(
    module_name: str,
    base_path: str,
    main_module_name_for_halt_logic: str,  # To know which module's HALT to keep
    # This cache is for CodeGenerator instances to know which modules' *definitions*
    # have already been extracted during the current overall compilation run.
    _shared_definitions_cache: Set[str],  # Renamed to avoid conflict with global
):
    """
    Recursively ensures a module and all its dependencies are compiled.
    Populates `module_own_bytecode_cache` and `ordered_modules_for_linking`.
    """
    # print(f"Ensuring module compiled: {module_name} from base: {base_path}")

    if module_name in module_own_bytecode_cache:
        return

    if module_name in compilation_in_progress_stack:
        return

    compilation_in_progress_stack.append(module_name)

    module_file_to_open = os.path.join(base_path, f"{module_name}.ns")
    try:
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

        codegen = CodeGenerator(processed_modules_cache=_shared_definitions_cache)
        own_bytecode, direct_dependencies = codegen.generate_program(
            ast, module_name=module_name
        )

        for dep_name in direct_dependencies:
            compile_all_modules_recursively(
                dep_name,
                base_path,
                main_module_name_for_halt_logic,
                _shared_definitions_cache,
            )

        if module_name not in module_own_bytecode_cache:
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

    if module_name in compilation_in_progress_stack:
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
    shared_definitions_cache_for_codegen.clear()  # This is the one passed to CodeGenerator
    compilation_in_progress_stack.clear()

    compile_all_modules_recursively(
        main_module_name,
        entry_base_path,
        main_module_name,
        shared_definitions_cache_for_codegen,
    )

    final_bytecode: List[Any] = []

    # The `ordered_modules_for_linking` should ideally be topologically sorted
    # or at least have dependencies before the modules that use them.
    # The current recursive approach aims for this by adding to `ordered_modules_for_linking`
    # *after* its dependencies have been processed.
    for module_name in ordered_modules_for_linking:
        if module_name in module_own_bytecode_cache:
            module_bc = module_own_bytecode_cache[module_name]
            if (
                module_name != main_module_name
                and module_bc
                and isinstance(module_bc[-1], tuple)
                and module_bc[-1] == (OpCode.HALT,)
            ):
                final_bytecode.extend(module_bc[:-1])
            else:
                final_bytecode.extend(module_bc)
        else:
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

    # print(f"Main file for test: {full_entry_point_path}") # Optional debug
    # print("Source (for main file):\n" + "-" * 10 + f"\n{source_code}\n" + "-" * 10) # Optional debug
    # if aux_files: # Optional debug
    #     for fname, fcontent in aux_files.items():
    #         print(f"Auxiliary file: {fname}\n" + "-"*10 + f"\n{fcontent}\n" + "-"*10)

    actual_prints_text = ""
    try:
        final_bytecode = compile_program_with_dependencies(full_entry_point_path)

        # print("Final Aggregated Bytecode:") # Optional debug
        # for i, instruction in enumerate(final_bytecode):
        #     print(f"  {i:03d}: {instruction}")

        if expected_prints is not None:
            result, actual_prints_text = execute_bytecode(
                final_bytecode, capture_prints=True
            )
            # print("Captured Output:") # Optional debug
            # if actual_prints_text: print(actual_prints_text, end="")
            # else: print("<no output>")
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
            for fname_key in (
                aux_files.keys()
            ):  # Iterate over keys to construct paths as they were made
                aux_path = os.path.join(test_base_path, fname_key)
                aux_dir = os.path.dirname(aux_path)
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


def run_tests():
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
    (use module_a (call_b_from_a get_a)) 
    (use module_b (get_b_direct))     
    (call_b_from_a)                   
    """
    run_notscheme_test(
        "Module: Circular Use for Definitions",
        source_code=main_circular_content,
        main_file_name="main_circular.ns",
        aux_files={"module_a.ns": module_a_content, "module_b.ns": module_b_content},
        expected_value=30,
    )
    print("\n--- All NotScheme end-to-end tests completed ---")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main_file_to_run = sys.argv[1]
        if not main_file_to_run.endswith(".ns"):
            print(f"Error: File to run must be a .ns file. Got: {main_file_to_run}")
            sys.exit(1)
        if not os.path.exists(main_file_to_run):
            print(f"Error: File not found: {main_file_to_run}")
            sys.exit(1)

        print(f"Running NotScheme program: {main_file_to_run}")
        try:
            final_bytecode = compile_program_with_dependencies(main_file_to_run)
            # print("\n--- Final Bytecode for CLI Run ---")
            # for i, instruction in enumerate(final_bytecode):
            #     print(f"  {i:03d}: {instruction}")
            # print("------------------------------------")

            # Execute without capturing prints, let them go to stdout directly
            execute_bytecode(final_bytecode, capture_prints=False)
            # The VM's HALT will print "Execution halted."
            # We might not want to print the final stack value for CLI runs unless specified.
        except (NotSchemeError, Exception) as e:
            print(f"Error during execution: {e}", file=sys.stderr)
            # import traceback
            # traceback.print_exc() # For more detailed debug if needed
            sys.exit(1)
    else:
        print("No file provided to run. Running internal tests...")
        run_tests()
