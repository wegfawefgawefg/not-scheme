import sys
from enum import Enum
import collections
import copy  # Needed for deep copying environments for closures

# --- Closure Representation ---
Closure = collections.namedtuple("Closure", ["code_label", "defining_env"])

# --- Struct Instance Representation ---
# Structs will be represented as dictionaries with a special key for their type.
# e.g., {'__type__': 'Point', 'x_coord': 10, 'y_coord': 20}


# --- Instruction Set Definition ---
class OpCode(Enum):
    # Stack Manipulation
    PUSH = 1  # Push a literal value onto the stack. Operand: value
    POP = 2  # Pop the top value from the stack.

    # Arithmetic / Logical Operations
    ADD = 10  # Pop two values, add them, push result.
    SUB = 11  # Pop two values, subtract top from second, push result.
    MUL = 12  # Pop two values, multiply them, push result.
    DIV = 13  # Pop two values, divide second by top, push result (float).
    EQ = 20  # Pop two values, push True if equal, False otherwise.
    LT = 21  # Pop two values, push True if second < top, False otherwise.
    GT = 22  # Pop two values, push True if second > top, False otherwise.
    NOT = 23  # Pop one value, push its boolean negation.

    # Variable Access
    LOAD = 30  # Load variable value onto stack. Operand: var_name
    STORE = 31  # Pop value, store in variable. Operand: var_name

    # Control Flow
    JUMP = 40  # Unconditional jump. Operand: label
    JUMP_IF_FALSE = 41  # Pop value, jump if false. Operand: label

    # Function Calls & Closures
    MAKE_CLOSURE = 45  # Create a closure. Operand: label
    CALL = 50  # Call a closure. Operand: arg_count
    RETURN = 51  # Return from function.

    # Struct Operations
    MAKE_STRUCT = (
        55  # Operands: struct_name_str, field_names_tuple (for creating dict keys)
    )
    GET_FIELD = 56  # Operand: field_name_str
    SET_FIELD = 57  # Operand: field_name_str

    # VM Control
    HALT = 60  # Stop execution.
    PRINT = 61  # Pop value and print it (for debugging/output).


# --- Virtual Machine Class ---
class VirtualMachine:
    def __init__(self, code):
        self.code = code
        self.labels = self._find_labels(code)

        self.operand_stack = []
        self.call_stack = collections.deque()
        self.ip = 0
        self.env_chain = [{"global": {}}]

    def _find_labels(self, code):
        labels = {}
        instruction_index = 0
        for instruction in code:
            if isinstance(instruction, str) and instruction.endswith(":"):
                label_name = instruction[:-1]
                if label_name in labels:
                    raise ValueError(f"Duplicate label found: {label_name}")
                labels[label_name] = instruction_index
            else:
                instruction_index += 1
        return labels

    def _get_instruction_index(self, target):
        if isinstance(target, str):
            if target not in self.labels:
                raise ValueError(f"Undefined label referenced: {target}")
            return self.labels[target]
        elif isinstance(target, int):
            return target
        else:
            raise TypeError(f"Invalid jump/call target type: {type(target)}")

    def _lookup(self, var_name):
        for scope in reversed(self.env_chain):
            if var_name in scope:
                return scope[var_name]
        raise NameError(f"Variable '{var_name}' not defined.")

    def _store(self, var_name, value):
        if not self.env_chain:
            raise RuntimeError("Environment chain is empty, cannot store.")
        self.env_chain[-1][var_name] = value

    def run(self):
        effective_code = [
            inst
            for inst in self.code
            if not (isinstance(inst, str) and inst.endswith(":"))
        ]
        code_len = len(effective_code)

        while 0 <= self.ip < code_len:
            instruction_to_execute = effective_code[self.ip]
            current_ip_for_error_reporting = self.ip

            try:
                # Ensure instruction_to_execute is a tuple before trying to subscript it
                if not isinstance(instruction_to_execute, tuple):
                    raise TypeError(
                        f"Malformed instruction: expected a tuple, got {type(instruction_to_execute)} value {instruction_to_execute!r}"
                    )

                opcode = instruction_to_execute[0]
                args = instruction_to_execute[1:]
            except TypeError as te:  # Catch the specific error if instruction_to_execute is not subscriptable
                print(f"\n--- VM Setup Error ---")
                print(f"Error: {te}")
                print(f"Instruction Pointer (IP): {current_ip_for_error_reporting}")
                print(
                    f"Malformed Instruction causing error: {instruction_to_execute!r} (type: {type(instruction_to_execute)})"
                )
                print(
                    f"Ensure all bytecode instructions are tuples, e.g., (OpCode.POP,) not OpCode.POP."
                )
                print(f"---------------------")
                self.ip = code_len  # Halt execution
                break

            self.ip += 1  # Increment IP after successfully fetching opcode and args

            try:
                if opcode == OpCode.PUSH:
                    self.operand_stack.append(args[0])
                elif opcode == OpCode.POP:
                    if not self.operand_stack:
                        raise IndexError("POP from empty stack")
                    self.operand_stack.pop()
                elif opcode == OpCode.ADD:
                    if len(self.operand_stack) < 2:
                        raise IndexError("ADD requires two operands")
                    right = self.operand_stack.pop()
                    left = self.operand_stack.pop()
                    self.operand_stack.append(left + right)
                elif opcode == OpCode.SUB:
                    if len(self.operand_stack) < 2:
                        raise IndexError("SUB requires two operands")
                    right = self.operand_stack.pop()
                    left = self.operand_stack.pop()
                    self.operand_stack.append(left - right)
                elif opcode == OpCode.MUL:
                    if len(self.operand_stack) < 2:
                        raise IndexError("MUL requires two operands")
                    right = self.operand_stack.pop()
                    left = self.operand_stack.pop()
                    self.operand_stack.append(left * right)
                elif opcode == OpCode.DIV:
                    if len(self.operand_stack) < 2:
                        raise IndexError("DIV requires two operands")
                    right = self.operand_stack.pop()
                    left = self.operand_stack.pop()
                    if right == 0:
                        raise ZeroDivisionError("Division by zero")
                    self.operand_stack.append(float(left) / right)
                elif opcode == OpCode.EQ:
                    if len(self.operand_stack) < 2:
                        raise IndexError("EQ requires two operands")
                    right = self.operand_stack.pop()
                    left = self.operand_stack.pop()
                    self.operand_stack.append(left == right)
                elif opcode == OpCode.LT:
                    if len(self.operand_stack) < 2:
                        raise IndexError("LT requires two operands")
                    right = self.operand_stack.pop()
                    left = self.operand_stack.pop()
                    self.operand_stack.append(left < right)
                elif opcode == OpCode.GT:
                    if len(self.operand_stack) < 2:
                        raise IndexError("GT requires two operands")
                    right = self.operand_stack.pop()
                    left = self.operand_stack.pop()
                    self.operand_stack.append(left > right)
                elif opcode == OpCode.NOT:
                    if not self.operand_stack:
                        raise IndexError("NOT requires one operand")
                    val = self.operand_stack.pop()
                    self.operand_stack.append(not val)
                elif opcode == OpCode.LOAD:
                    var_name = args[0]
                    value = self._lookup(var_name)
                    self.operand_stack.append(value)
                elif opcode == OpCode.STORE:
                    if not self.operand_stack:
                        raise IndexError(
                            f"STORE '{args[0]}' requires a value on the stack"
                        )
                    var_name = args[0]
                    value = self.operand_stack.pop()
                    self._store(var_name, value)
                elif opcode == OpCode.JUMP:
                    target_label = args[0]
                    self.ip = self._get_instruction_index(target_label)
                elif opcode == OpCode.JUMP_IF_FALSE:
                    if not self.operand_stack:
                        raise IndexError("JUMP_IF_FALSE requires a value on the stack")
                    condition = self.operand_stack.pop()
                    if not condition:
                        target_label = args[0]
                        self.ip = self._get_instruction_index(target_label)
                elif opcode == OpCode.MAKE_CLOSURE:
                    code_label = args[0]
                    defining_env = list(self.env_chain)
                    closure = Closure(code_label=code_label, defining_env=defining_env)
                    self.operand_stack.append(closure)
                elif opcode == OpCode.CALL:
                    arg_count = args[0]
                    if len(self.operand_stack) < arg_count + 1:  # Need args + closure
                        raise IndexError(
                            f"CALL expected {arg_count} arguments and a closure, stack too small: {len(self.operand_stack)}"
                        )
                    callee = self.operand_stack.pop()
                    if not isinstance(callee, Closure):
                        self.operand_stack.append(callee)
                        raise TypeError(
                            f"CALL expects a Closure object, got {type(callee)}"
                        )
                    if len(self.operand_stack) < arg_count:
                        self.operand_stack.append(callee)
                        raise IndexError(
                            f"CALL expected {arg_count} arguments on stack for closure '{callee.code_label}', found {len(self.operand_stack)}"
                        )
                    return_ip = self.ip
                    self.call_stack.append((return_ip, self.env_chain))
                    new_local_scope = {}
                    self.env_chain = callee.defining_env + [new_local_scope]
                    self.ip = self._get_instruction_index(callee.code_label)
                elif opcode == OpCode.RETURN:
                    if not self.call_stack:
                        print("Warning: RETURN called from top level. Halting.")
                        self.ip = code_len
                        break
                    return_ip, previous_env_chain = self.call_stack.pop()
                    self.env_chain = previous_env_chain
                    self.ip = return_ip
                elif opcode == OpCode.MAKE_STRUCT:
                    struct_name_str = args[0]
                    field_names_tuple = args[1]
                    field_count = len(field_names_tuple)
                    if len(self.operand_stack) < field_count:
                        raise IndexError(
                            f"MAKE_STRUCT '{struct_name_str}' expected {field_count} field values on stack, found {len(self.operand_stack)}"
                        )
                    struct_instance = {"__type__": struct_name_str}
                    field_values = []
                    for _ in range(field_count):
                        field_values.insert(0, self.operand_stack.pop())
                    for i in range(field_count):
                        struct_instance[field_names_tuple[i]] = field_values[i]
                    self.operand_stack.append(struct_instance)
                elif opcode == OpCode.GET_FIELD:
                    field_name_str = args[0]
                    if not self.operand_stack:
                        raise IndexError(
                            f"GET_FIELD '{field_name_str}' requires a struct instance on the stack."
                        )
                    instance = self.operand_stack.pop()
                    if not isinstance(instance, dict) or "__type__" not in instance:
                        self.operand_stack.append(instance)  # Push back for error state
                        raise TypeError(
                            f"GET_FIELD '{field_name_str}' expected a struct instance, got {type(instance)}."
                        )
                    if field_name_str not in instance:
                        self.operand_stack.append(instance)  # Push back for error state
                        raise AttributeError(
                            f"Struct type '{instance['__type__']}' has no field '{field_name_str}'."
                        )
                    self.operand_stack.append(instance[field_name_str])
                elif opcode == OpCode.SET_FIELD:
                    field_name_str = args[0]
                    if len(self.operand_stack) < 2:
                        raise IndexError(
                            f"SET_FIELD '{field_name_str}' requires a value and a struct instance on the stack."
                        )
                    new_value = self.operand_stack.pop()
                    instance = self.operand_stack.pop()
                    if not isinstance(instance, dict) or "__type__" not in instance:
                        self.operand_stack.append(instance)
                        self.operand_stack.append(new_value)
                        raise TypeError(
                            f"SET_FIELD '{field_name_str}' expected a struct instance, got {type(instance)}."
                        )
                    if field_name_str not in instance:
                        self.operand_stack.append(instance)
                        self.operand_stack.append(new_value)
                        raise AttributeError(
                            f"Struct type '{instance['__type__']}' has no field '{field_name_str}' to set."
                        )
                    instance[field_name_str] = new_value
                    self.operand_stack.append(instance)
                elif opcode == OpCode.HALT:
                    print("Execution halted.")
                    self.ip = code_len
                    break
                elif opcode == OpCode.PRINT:
                    if not self.operand_stack:
                        raise IndexError("PRINT requires a value on the stack")
                    value = self.operand_stack.pop()
                    print("Output:", value)
                else:
                    raise RuntimeError(f"Unknown opcode encountered: {opcode}")

            except (
                IndexError,
                NameError,
                ZeroDivisionError,
                ValueError,
                RuntimeError,
                TypeError,
                AttributeError,
            ) as e:
                print(f"\n--- Runtime Error ---")
                print(f"Error: {e}")
                print(f"Instruction Pointer (IP): {current_ip_for_error_reporting}")
                print(f"Instruction: {instruction_to_execute}")
                print(f"Operand Stack (Top First): {self.operand_stack[::-1]}")
                print(f"Call Stack Depth: {len(self.call_stack)}")
                print(f"---------------------")
                self.ip = code_len
                break

        if self.operand_stack:
            return self.operand_stack[-1]
        return None


if __name__ == "__main__":
    print("Virtual Machine definition loaded.")
    print("To run tests, please execute 'test.py' or run specific bytecode here.")

    print("\n--- VM Struct Operations Test ---")
    struct_test_code = [
        (OpCode.PUSH, 10),
        (OpCode.PUSH, 20),
        (OpCode.MAKE_STRUCT, "Point", ("x_coord", "y_coord")),
        (OpCode.STORE, "p"),
        (OpCode.LOAD, "p"),
        (OpCode.PUSH, 25),
        (OpCode.SET_FIELD, "y_coord"),
        (OpCode.POP,),  # This is a tuple: (OpCode.POP,)
        (OpCode.LOAD, "p"),
        (OpCode.GET_FIELD, "x_coord"),
        (OpCode.PRINT,),  # This is a tuple: (OpCode.PRINT,)
        (OpCode.LOAD, "p"),
        (OpCode.GET_FIELD, "y_coord"),
        (OpCode.PRINT,),  # This is a tuple: (OpCode.PRINT,)
        (OpCode.HALT,),  # This is a tuple: (OpCode.HALT,)
    ]
    vm_struct_test = VirtualMachine(struct_test_code)
    result = vm_struct_test.run()
    print(f"Final result from struct test: {result}")
    print("---------------------------------")
