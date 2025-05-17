# This file contains test cases for the Virtual Machine.
# It assumes the VirtualMachine, OpCode, and Closure classes are defined
# in a file named `vm.py` in the same directory.

from vm import VirtualMachine, OpCode, Closure  # Assuming vm.py contains the VM


def run_test(
    name,
    bytecode,
    expected_result_info="",
    expected_final_stack_top_value=None,
    check_final_value=False,
):
    """Helper function to run a test and print its status."""
    print(f"--- Running Test: {name} ---")
    print(f"Expected output info: {expected_result_info}")
    if check_final_value:
        print(f"Expected final stack top value: {expected_final_stack_top_value}")

    vm = VirtualMachine(bytecode)
    result = vm.run()

    print(f"Actual Result from VM (top of stack at end): {result}")
    if check_final_value:
        if result == expected_final_stack_top_value:
            print(
                f"Final stack top value: PASS (Expected: {expected_final_stack_top_value}, Got: {result})"
            )
        else:
            print(
                f"Final stack top value: FAIL (Expected: {expected_final_stack_top_value}, Got: {result})"
            )
    print("-" * 30)
    return result


def test_arithmetic():
    """Test basic arithmetic operations: (5 + 3) * 2"""
    bytecode = [
        (OpCode.PUSH, 5),
        (OpCode.PUSH, 3),
        (OpCode.ADD,),
        (OpCode.PUSH, 2),
        (OpCode.MUL,),
        (OpCode.PRINT,),  # Should print 16
        # (OpCode.HALT,) # HALT will make result None, let value stay for check
    ]
    # If HALT is used, expected_final_stack_top_value should be None.
    # If HALT is not used, the result of MUL (16) will be on stack.
    # Let's test with HALT for now as PRINT consumes the value.
    bytecode_with_halt = bytecode + [(OpCode.HALT,)]
    run_test("Example 1: Arithmetic", bytecode_with_halt, "Output: 16", None, True)


def test_conditional():
    """Test conditional jump: if 5 > 3 then push 10 else push 20"""
    bytecode = [
        (OpCode.PUSH, 5),
        (OpCode.PUSH, 3),
        (OpCode.GT,),
        (OpCode.JUMP_IF_FALSE, "else_branch"),
        (OpCode.PUSH, 10),
        (OpCode.JUMP, "end_if"),
        "else_branch:",
        (OpCode.PUSH, 20),
        "end_if:",
        (OpCode.PRINT,),
        (OpCode.HALT,),
    ]
    run_test("Example 2: Conditional", bytecode, "Output: 10", None, True)


def test_function_call_closure():
    """Test function call using MAKE_CLOSURE: define add(a, b) and call add(7, 8)"""
    bytecode = [
        (OpCode.MAKE_CLOSURE, "add_func"),
        (OpCode.STORE, "my_add"),
        (
            OpCode.PUSH,
            8,
        ),  # Arg 1 for add (Scheme pushes right to left for (add arg1 arg2))
        (OpCode.PUSH, 7),  # Arg 2 for add
        # Stack for CALL: [closure_obj, arg1_val, arg2_val] (top is arg2_val)
        # VM CALL pops closure, then expects arg_count values on stack.
        # So bytecode for CALL should be: PUSH argN, ..., PUSH arg1, PUSH closure, CALL N
        # Corrected order for args for CALL:
        (OpCode.PUSH, 7),  # val for param 'a'
        (OpCode.PUSH, 8),  # val for param 'b'
        (OpCode.LOAD, "my_add"),
        (OpCode.CALL, 2),
        (OpCode.PRINT,),
        (OpCode.HALT,),
        "add_func:",
        (OpCode.STORE, "b"),  # param b gets value 8
        (OpCode.STORE, "a"),  # param a gets value 7
        (OpCode.LOAD, "a"),
        (OpCode.LOAD, "b"),
        (OpCode.ADD,),
        (OpCode.RETURN,),
    ]
    # The previous bytecode for CALL had args pushed before loading closure.
    # Corrected bytecode for function call:
    bytecode_corrected_call = [
        (OpCode.MAKE_CLOSURE, "add_func"),
        (OpCode.STORE, "my_add"),
        # Args are pushed first, then the closure object
        (OpCode.PUSH, 7),  # value for 'a'
        (OpCode.PUSH, 8),  # value for 'b'
        (OpCode.LOAD, "my_add"),  # Load the closure object (now on top of args)
        # Stack before CALL: [..., 7, 8, <closure_obj>]
        # VM CALL pops closure, then expects 2 args (8, then 7)
        (OpCode.CALL, 2),  # Call the closure with 2 arguments
        (OpCode.PRINT,),  # Print the result (15)
        (OpCode.HALT,),  # Stop
        "add_func:",  # Parameters are on stack: 8 (for b), then 7 (for a)
        (OpCode.STORE, "b"),  # Store arg popped first (8) into local 'b'
        (OpCode.STORE, "a"),  # Store arg popped second (7) into local 'a'
        (OpCode.LOAD, "a"),
        (OpCode.LOAD, "b"),
        (OpCode.ADD,),
        (OpCode.RETURN,),
    ]
    run_test(
        "Example 3: Function Call (Closure)",
        bytecode_corrected_call,
        "Output: 15",
        None,
        True,
    )


def test_recursion_closure():
    """Test recursion with closures: Factorial(3)"""
    bytecode = [
        (OpCode.MAKE_CLOSURE, "factorial"),
        (OpCode.STORE, "fact"),
        (OpCode.PUSH, 3),
        (OpCode.LOAD, "fact"),
        (OpCode.CALL, 1),
        (OpCode.PRINT,),
        (OpCode.HALT,),
        "factorial:",
        (OpCode.STORE, "n"),
        (OpCode.LOAD, "n"),
        (OpCode.PUSH, 1),
        (OpCode.GT,),
        (OpCode.JUMP_IF_FALSE, "fact_base_case"),
        (OpCode.LOAD, "n"),
        (OpCode.LOAD, "n"),
        (OpCode.PUSH, 1),
        (OpCode.SUB,),
        (OpCode.LOAD, "fact"),
        (OpCode.CALL, 1),
        (OpCode.MUL,),
        (OpCode.RETURN,),
        "fact_base_case:",
        (OpCode.PUSH, 1),
        (OpCode.RETURN,),
    ]
    run_test(
        "Example 4: Recursion (Factorial 3 - Closure)",
        bytecode,
        "Output: 6",
        None,
        True,
    )


def test_scope_closure():
    """Test variable scope with closures."""
    bytecode = [
        (OpCode.PUSH, 100),
        (OpCode.STORE, "x"),
        (OpCode.MAKE_CLOSURE, "scopetest_revised"),
        (OpCode.STORE, "test_func"),
        (OpCode.PUSH, 5),
        (OpCode.LOAD, "test_func"),
        (OpCode.CALL, 1),
        (OpCode.PRINT,),  # Should print 105
        (OpCode.LOAD, "x"),
        (OpCode.PRINT,),  # Should print 100 (global x unchanged)
        (OpCode.HALT,),
        "scopetest_revised:",
        (OpCode.STORE, "arg"),
        (OpCode.LOAD, "arg"),
        (OpCode.LOAD, "x"),  # Loads global 'x' (100)
        (OpCode.ADD,),
        (OpCode.STORE, "x"),  # Stores result in *local* 'x' (105), shadows global
        (OpCode.LOAD, "x"),  # Loads local 'x' (105)
        (OpCode.RETURN,),
    ]
    run_test(
        "Example 5: Scope Test (Closure)",
        bytecode,
        "Outputs: 105, then 100",
        None,
        True,
    )


def test_true_closure_make_adder():
    """Test a true closure scenario: a function returning another function (make-adder)."""
    bytecode = [
        (OpCode.MAKE_CLOSURE, "make_adder_func"),
        (OpCode.STORE, "make_adder"),
        (OpCode.PUSH, 5),
        (OpCode.LOAD, "make_adder"),
        (OpCode.CALL, 1),
        (OpCode.STORE, "add5"),
        (OpCode.PUSH, 10),
        (OpCode.LOAD, "add5"),
        (OpCode.CALL, 1),
        (OpCode.PRINT,),
        (OpCode.HALT,),
        "make_adder_func:",
        (OpCode.STORE, "x"),
        (OpCode.MAKE_CLOSURE, "inner_adder_func"),
        (OpCode.RETURN,),
        "inner_adder_func:",
        (OpCode.STORE, "y"),
        (OpCode.LOAD, "x"),
        (OpCode.LOAD, "y"),
        (OpCode.ADD,),
        (OpCode.RETURN,),
    ]
    run_test("Example 6: Closure Test (Make Adder)", bytecode, "Output: 15", None, True)


def test_struct_operations():
    """Tests MAKE_STRUCT, GET_FIELD, SET_FIELD operations."""
    bytecode = [
        # Make a Point struct instance: (Point 10 20)
        (OpCode.PUSH, 10),  # value for x_coord
        (OpCode.PUSH, 20),  # value for y_coord
        (OpCode.MAKE_STRUCT, "Point", ("x_coord", "y_coord")),
        (OpCode.STORE, "p"),  # Store instance in var p
        # (set p y_coord 25)
        # For SET_FIELD, stack should be [instance, value] (top is value)
        (OpCode.LOAD, "p"),  # Push instance p.
        (OpCode.PUSH, 25),  # Push value 25.
        (
            OpCode.SET_FIELD,
            "y_coord",
        ),  # instance.y_coord = 25. Instance is pushed back.
        (OpCode.POP,),  # Pop the modified instance from SET_FIELD.
        # (get p x_coord) -> result 10
        (OpCode.LOAD, "p"),
        (OpCode.GET_FIELD, "x_coord"),
        (OpCode.PRINT,),  # Output: 10
        # (get p y_coord) -> result 25
        (OpCode.LOAD, "p"),
        (OpCode.GET_FIELD, "y_coord"),
        # Leave the result on stack for final check if not using HALT
        # (OpCode.PRINT,), # If we print, it consumes the value
        (OpCode.HALT,),  # HALT means final stack top will be None
    ]
    # If the last PRINT is commented out and HALT is used, result is None.
    # If we want to check the value 25, we should not PRINT it last and not HALT.
    # Let's modify to test the final value on stack.

    bytecode_check_val = [
        (OpCode.PUSH, 10),
        (OpCode.PUSH, 20),
        (OpCode.MAKE_STRUCT, "Point", ("x_coord", "y_coord")),
        (OpCode.STORE, "p"),
        (OpCode.LOAD, "p"),
        (OpCode.PUSH, 25),
        (OpCode.SET_FIELD, "y_coord"),
        (OpCode.POP,),
        (OpCode.LOAD, "p"),
        (OpCode.GET_FIELD, "x_coord"),
        (OpCode.PRINT,),  # Prints 10
        (OpCode.LOAD, "p"),
        (OpCode.GET_FIELD, "y_coord"),
        # Value 25 is now on top of the stack. HALT will clear it.
        # To check it with run_test, we need it to be the return value of vm.run()
    ]
    run_test(
        "Struct Operations",
        bytecode_check_val,
        "Output: 10. Final stack top should be 25.",
        25,
        True,
    )

    # Test with HALT to ensure PRINT output is as expected and final result is None
    bytecode_with_halt_and_prints = [
        (OpCode.PUSH, 10),
        (OpCode.PUSH, 20),
        (OpCode.MAKE_STRUCT, "Point", ("x_coord", "y_coord")),
        (OpCode.STORE, "p"),
        (OpCode.LOAD, "p"),
        (OpCode.PUSH, 25),
        (OpCode.SET_FIELD, "y_coord"),
        (OpCode.POP,),
        (OpCode.LOAD, "p"),
        (OpCode.GET_FIELD, "x_coord"),
        (OpCode.PRINT,),  # Output: 10
        (OpCode.LOAD, "p"),
        (OpCode.GET_FIELD, "y_coord"),
        (OpCode.PRINT,),  # Output: 25
        (OpCode.HALT,),
    ]
    run_test(
        "Struct Operations with HALT",
        bytecode_with_halt_and_prints,
        "Outputs: 10, then 25. Final stack top None.",
        None,
        True,
    )


if __name__ == "__main__":
    test_arithmetic()
    test_conditional()
    test_function_call_closure()  # This test's bytecode was corrected
    test_recursion_closure()
    test_scope_closure()
    test_true_closure_make_adder()
    test_struct_operations()

    print("\nAll tests completed.")
