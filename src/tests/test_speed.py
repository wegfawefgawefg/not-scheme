# speed_tests.py
# Compares the performance of NotScheme vs Native Python for certain tasks.

import time
import os
import sys

# Assuming run_notscheme.py is in the src directory or accessible in PYTHONPATH
try:
    from ns import (
        compile_program_with_dependencies,
        execute_bytecode,
        NotSchemeError,
    )
    from src.vm import QuotedSymbol
except ImportError as e:
    print(f"Error importing from src.run_notscheme or src.vm: {e}")
    print(
        "Please ensure run_notscheme.py and vm.py are in the src/ directory or accessible via PYTHONPATH."
    )
    sys.exit(1)

# --- Test Functions in Python ---


def py_fib(n):
    if n < 2:
        return n
    return py_fib(n - 1) + py_fib(n - 2)


def py_sum_recursive_aux(n, current_sum):
    if n == 0:
        return current_sum
    return py_sum_recursive_aux(n - 1, current_sum + n)


def py_sum_up_to_recursive(max_val):
    # Python's default recursion limit is often around 1000.
    # If max_val is too high, this will cause a RecursionError.
    if max_val > 950:  # Add a small buffer
        print(
            f"Warning: max_val {max_val} for py_sum_up_to_recursive might exceed Python's recursion limit."
        )
        # return "Python recursion limit likely exceeded" # Or handle differently
    return py_sum_recursive_aux(max_val, 0)


# --- NotScheme Code for Tests ---

notscheme_fib_code_template = """
// fib_test.ns
(fn fib (n)
  (if (< n 2)
      n
      (+ (fib (- n 1)) (fib (- n 2)))))
(fib {N}) 
"""

notscheme_sum_recursive_code_template = """
// sum_recursive_test.ns
(fn sum_recursive_aux (n current_sum)
  (if (= n 0)
      current_sum
      (sum_recursive_aux (- n 1) (+ current_sum n))))

(fn sum_up_to_recursive (max_val)
  (sum_recursive_aux max_val 0))

(sum_up_to_recursive {N})
"""

# --- Test Execution ---


def run_performance_test(
    test_name: str,
    notscheme_code: str,
    python_func,
    arg,
    notscheme_main_file_name: str = "temp_perf_main.ns",
):
    print(f"\n--- Performance Test: {test_name} (Argument: {arg}) ---")

    # Python execution
    py_result = "Python execution error"  # Default in case of error
    py_duration = float("inf")
    try:
        start_time_py = time.perf_counter()
        py_result = python_func(arg)
        end_time_py = time.perf_counter()
        py_duration = end_time_py - start_time_py
        print(f"Python result: {py_result}")
        print(f"Python time:   {py_duration:.6f} seconds")
    except RecursionError:
        print(f"Python result: RecursionError (limit exceeded for n={arg})")
        print(f"Python time:   N/A (RecursionError)")
    except Exception as e_py:
        print(f"Python result: Error ({e_py})")
        print(f"Python time:   N/A (Error)")

    # NotScheme execution
    created_files = []
    ns_result = "NotScheme execution error"  # Default
    run_duration_ns = float("inf")
    try:
        with open(notscheme_main_file_name, "w") as f:
            f.write(notscheme_code)
        created_files.append(notscheme_main_file_name)

        start_time_compile_ns = time.perf_counter()
        bytecode = compile_program_with_dependencies(notscheme_main_file_name)
        end_time_compile_ns = time.perf_counter()
        compile_duration_ns = end_time_compile_ns - start_time_compile_ns
        # print(f"NotScheme compilation time: {compile_duration_ns:.6f} seconds")

        start_time_run_ns = time.perf_counter()
        ns_result = execute_bytecode(bytecode, capture_prints=False)
        end_time_run_ns = time.perf_counter()
        run_duration_ns = end_time_run_ns - start_time_run_ns

        print(f"NotScheme result: {ns_result}")
        print(f"NotScheme execution time: {run_duration_ns:.6f} seconds")

        if (
            isinstance(py_result, (int, float))
            and isinstance(ns_result, (int, float))
            and py_result == ns_result
        ):
            print("Results: MATCH")
        elif py_result == ns_result:  # For other types or error strings
            print("Results: MATCH (or both errored similarly)")
        else:
            print(f"Results: MISMATCH (Python: {py_result}, NotScheme: {ns_result})")

        if (
            py_duration > 0
            and py_duration != float("inf")
            and run_duration_ns != float("inf")
        ):
            print(
                f"NotScheme is approximately {run_duration_ns / py_duration:.2f}x slower than Python for execution."
            )
        elif py_duration == 0 or py_duration == float("inf"):
            print(
                "Python execution was too fast or errored to measure a meaningful ratio."
            )

    except NotSchemeError as e:
        print(f"NotScheme Error: {e}")
    except Exception as e:
        print(f"Unexpected Error during NotScheme test: {e}")
        import traceback

        traceback.print_exc()
    finally:
        for fname in created_files:
            if os.path.exists(fname):
                try:
                    os.remove(fname)
                except OSError:
                    pass
    print("-" * 30)


def run_all_speed_tests():
    """Runs all speed/performance comparison tests."""
    print("--- Running Performance Comparison Tests ---")

    fib_n_value = 20
    run_performance_test(
        "Recursive Fibonacci",
        notscheme_fib_code_template.format(N=fib_n_value),
        py_fib,
        fib_n_value,
        notscheme_main_file_name=f"fib_test_{fib_n_value}.ns",
    )

    sum_n_value = 900  # Reduced from a potentially higher value
    run_performance_test(
        "Recursive Summation",
        notscheme_sum_recursive_code_template.format(N=sum_n_value),
        py_sum_up_to_recursive,
        sum_n_value,
        notscheme_main_file_name=f"sum_recursive_test_{sum_n_value}.ns",
    )
    # The final "Performance comparison finished." print is now inside this function.


if __name__ == "__main__":
    # To run this file directly, ensure the project root is in PYTHONPATH
    # or run as a module: python -m src.tests.test_speed
    run_all_speed_tests()
