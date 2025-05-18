# Main test runner for the NotScheme language and its components.

import io
import os
import sys
from typing import Any, Optional, List, Dict

from vm import QuotedSymbol
from run_notscheme import (
    compile_program_with_dependencies,
    execute_bytecode,
    NotSchemeError,
)

# Imports for VM tests
from vm_tests import (
    test_arithmetic as vm_test_arithmetic,
    test_conditional as vm_test_conditional,
    test_function_call_closure as vm_test_function_call_closure,
    test_recursion_closure as vm_test_recursion_closure,
    test_scope_closure as vm_test_scope_closure,
    test_true_closure_make_adder as vm_test_true_closure_make_adder,
    test_struct_operations as vm_test_struct_operations,
)

# Imports for Speed tests
from speed_tests import (
    run_performance_test,
    notscheme_fib_code_template,
    py_fib,
    notscheme_sum_recursive_code_template,
    py_sum_up_to_recursive,
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


def run_language_feature_tests(): # Renamed from run_tests
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


# --- New Main Test Orchestration ---
def run_all_tests_orchestrator():
    print("=============================================")
    print("=== Running All NotScheme Test Suites ===")
    print("=============================================\n")

    print("\n--- Running NotScheme Language End-to-End Feature Tests ---")
    run_language_feature_tests()

    print("\n\n--- Running VM Tests ---")
    vm_test_arithmetic()
    vm_test_conditional()
    vm_test_function_call_closure()
    vm_test_recursion_closure()
    vm_test_scope_closure()
    vm_test_true_closure_make_adder()
    vm_test_struct_operations()
    print("\n--- All VM tests completed. ---")

    print("\n\n--- Running Performance Comparison Tests ---")
    # Replicate the calls from speed_tests.py's main block
    fib_n_value = 20
    run_performance_test(
        "Recursive Fibonacci",
        notscheme_fib_code_template.format(N=fib_n_value),
        py_fib,
        fib_n_value,
        notscheme_main_file_name=f"fib_test_{fib_n_value}.ns",
    )

    sum_n_value = 900
    run_performance_test(
        "Recursive Summation",
        notscheme_sum_recursive_code_template.format(N=sum_n_value),
        py_sum_up_to_recursive,
        sum_n_value,
        notscheme_main_file_name=f"sum_recursive_test_{sum_n_value}.ns",
    )
    print("\n--- Performance comparison finished. ---")

    print("\n\n=============================================")
    print("=== All Test Suites Completed ===")
    print("=============================================")


if __name__ == "__main__":
    run_all_tests_orchestrator()
