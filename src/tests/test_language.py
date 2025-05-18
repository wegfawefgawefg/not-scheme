# Contains end-to-end language feature tests

import os
import traceback
from typing import Any, Optional, List, Dict

from src.vm import QuotedSymbol
from ns import (
    compile_program_with_dependencies,
    execute_bytecode,
    NotSchemeError,
)


def run_notscheme_test(
    test_name: str,
    source_code: str,
    main_file_name: Optional[str] = None,
    aux_files: Optional[Dict[str, str]] = None,
    expected_value: Any = None,
    expect_error: bool = False,
    expected_prints: Optional[List[Any]] = None,
):
    print(f"\n--- Language Test: {test_name} ---")

    created_files = []
    # Assume tests are run from the project root, so paths for aux files are relative to root.
    # If test_language.py is run directly from src/, os.getcwd() will be src/.
    # For consistency, let's try to make aux files in a temp sub-directory if possible,
    # or ensure paths are handled robustly. For now, using current os.getcwd().
    test_base_path = os.getcwd()

    if aux_files:
        for fname, fcontent in aux_files.items():
            # Ensure aux files are created relative to the test_base_path
            # This means if aux_files has "subdir/file.ns", it's created in "test_base_path/subdir/file.ns"
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
        # Create main test file in the current working directory (or a subdir if specified in name)
        else f"{test_name.lower().replace(' ', '_').replace(':', '_').replace('(', '_').replace(')', '_')}_main.ns"
    )
    full_entry_point_path = os.path.join(test_base_path, entry_point_filename_for_test)

    entry_dir = os.path.dirname(full_entry_point_path)
    if entry_dir and not os.path.exists(entry_dir):
        os.makedirs(entry_dir)

    with open(full_entry_point_path, "w") as f:
        f.write(source_code)
    created_files.append(full_entry_point_path)

    actual_prints_text = ""
    try:
        # compile_program_with_dependencies expects the full path to the main file.
        final_bytecode = compile_program_with_dependencies(full_entry_point_path)

        if expected_prints is not None:
            result, actual_prints_text = execute_bytecode(
                final_bytecode, capture_prints=True
            )
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
                    elif p_val is None:  # Python None for NotScheme's nil/none
                        formatted_expected_prints.append("Output: None")
                    else:
                        formatted_expected_prints.append(f"Output: {p_val}")
                if actual_print_lines == formatted_expected_prints:
                    print("Prints: PASS")
                else:
                    print("Prints: FAIL")
                    print(f"  Expected: {formatted_expected_prints}")
                    print(f"  Actual  : {actual_print_lines}")

            # Check result if expected_value is provided OR if there are no expected_prints (implies result matters)
            if expected_value is not None or not expected_prints:
                if result == expected_value:
                    print(f"Result: PASS (Expected: {expected_value}, Got: {result})")
                else:
                    print(f"Result: FAIL (Expected: {expected_value}, Got: {result})")
            # If only prints are expected, and expected_value is implicitly None (not provided)
            elif expected_prints and expected_value is None:
                if (
                    result is None
                ):  # After HALT, VM returns None if stack empty, or last value
                    print(
                        f"Result: PASS (Expected implicit None after prints/HALT, Got: {result})"
                    )
                else:
                    print(
                        f"Result: UNEXPECTED (Expected implicit None after prints/HALT, Got: {result})"
                    )
    except (NotSchemeError, Exception) as e:
        if expect_error:
            print(f"PASS: Caught expected error: {e}")
        else:
            print(f"FAIL: Unexpected error: {e}")
            traceback.print_exc()
    finally:
        for fname_to_remove in created_files:
            if os.path.exists(fname_to_remove):
                try:
                    os.remove(fname_to_remove)
                except OSError as e_os:
                    print(
                        f"Warning: could not remove test file {fname_to_remove}: {e_os}"
                    )

        # Attempt to remove created directories if they are empty
        # This needs to be done carefully, from child to parent, or by tracking created dirs
        # For simplicity, let's try removing the entry_dir and aux_dirs if they were created and are now empty.
        if aux_files:
            # Get unique directory paths from aux_files keys
            aux_dirs_to_check = set()
            for fname_key in aux_files.keys():
                aux_path = os.path.join(test_base_path, fname_key)
                aux_dir = os.path.dirname(aux_path)
                if (
                    aux_dir and aux_dir != test_base_path
                ):  # Only consider subdirectories
                    aux_dirs_to_check.add(aux_dir)

            for adir in sorted(
                list(aux_dirs_to_check), reverse=True
            ):  # Process deeper dirs first
                if os.path.exists(adir) and not os.listdir(adir):
                    try:
                        os.rmdir(adir)
                    except OSError as e_os:
                        print(f"Warning: could not remove test aux dir {adir}: {e_os}")

        if (
            entry_dir
            and entry_dir != test_base_path
            and os.path.exists(entry_dir)
            and not os.listdir(entry_dir)
        ):
            try:
                os.rmdir(entry_dir)
            except OSError as e_os:
                print(f"Warning: could not remove test entry dir {entry_dir}: {e_os}")

    print("-" * (20 + len(test_name) + 10))


def run_language_feature_tests():
    """Runs all end-to-end language feature tests."""
    print("--- Running NotScheme Language End-to-End Feature Tests ---")
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
        """(print "Hello")(print 123)(print true)(print nil)(+ 1 1)""",  # Assuming nil is the keyword for None
        expected_value=2,  # Result of (+ 1 1)
        expected_prints=["Hello", 123, True, None],  # Python None for NotScheme's nil
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
        expected_value=None,  # print returns None, then HALT
        expected_prints=[QuotedSymbol(name="my_symbol")],
    )
    run_notscheme_test(
        "Quote: Simple List",
        "(print '(item1 10 true nil))",
        expected_value=None,  # print returns None, then HALT
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
        main_file_name="main_specific.ns",  # This will be created in current dir
        aux_files={"math_utils.ns": math_utils_content},  # Also in current dir
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
        expected_value="Hello from string_ext module",  # Result of (get_greeting)
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
    print("\n--- All NotScheme Language End-to-End Feature Tests Completed ---")


if __name__ == "__main__":
    # To run this file directly, ensure the project root is in PYTHONPATH
    # or run as a module: python -m src.tests.test_language
    run_language_feature_tests()
