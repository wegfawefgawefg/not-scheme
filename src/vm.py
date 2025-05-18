from enum import Enum
import collections

# --- Closure Representation ---
Closure = collections.namedtuple("Closure", ["code_label", "defining_env"])

# --- Quoted Symbol Representation ---
QuotedSymbol = collections.namedtuple("QuotedSymbol", ["name"])

# --- Struct Instance Representation ---
# Structs are dictionaries with a '__type__' key.

# --- List Representation ---
# NotScheme lists are Python lists. Nil is Python None.


class OpCode(Enum):
    """instruction set definition"""

    # Stack Manipulation
    PUSH = 1
    POP = 2

    # Arithmetic / Logical Operations
    ADD = 10
    SUB = 11
    MUL = 12
    DIV = 13
    EQ = 20
    LT = 21
    GT = 22
    NOT = 23

    # Variable Access
    LOAD = 30
    STORE = 31

    # Control Flow
    JUMP = 40
    JUMP_IF_FALSE = 41

    # Function Calls & Closures
    MAKE_CLOSURE = 45
    CALL = 50
    RETURN = 51

    # Struct Operations
    MAKE_STRUCT = 55
    GET_FIELD = 56
    SET_FIELD = 57

    # VM Control
    HALT = 60
    PRINT = 61

    # List Primitives
    IS_NIL = 70
    CONS = 71
    FIRST = 72
    REST = 73
    MAKE_LIST = 74

    # Type Predicates
    IS_BOOLEAN = 80
    IS_NUMBER = 81
    IS_STRING = 82
    IS_LIST = 83
    IS_STRUCT = 84
    IS_FUNCTION = 85
    # LENGTH, APPEND_LIST, STRING_APPEND, NTH_ELEMENT are removed


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
                if not isinstance(instruction_to_execute, tuple):
                    raise TypeError(
                        f"Malformed instruction: expected a tuple, got {type(instruction_to_execute)} value {instruction_to_execute!r}"
                    )

                opcode = instruction_to_execute[0]
                args = instruction_to_execute[1:]
            except TypeError as te:
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
                self.ip = code_len
                break

            self.ip += 1

            try:
                # --- Stack Manipulation ---
                if opcode == OpCode.PUSH:
                    self.operand_stack.append(args[0])
                elif opcode == OpCode.POP:
                    if not self.operand_stack:
                        raise IndexError("POP from empty stack")
                    self.operand_stack.pop()
                # --- Arithmetic / Logical Operations ---
                elif opcode == OpCode.ADD:
                    if len(self.operand_stack) < 2:
                        raise IndexError("ADD requires two operands")
                    right, left = self.operand_stack.pop(), self.operand_stack.pop()
                    self.operand_stack.append(left + right)
                elif opcode == OpCode.SUB:
                    if len(self.operand_stack) < 2:
                        raise IndexError("SUB requires two operands")
                    right, left = self.operand_stack.pop(), self.operand_stack.pop()
                    self.operand_stack.append(left - right)
                elif opcode == OpCode.MUL:
                    if len(self.operand_stack) < 2:
                        raise IndexError("MUL requires two operands")
                    right, left = self.operand_stack.pop(), self.operand_stack.pop()
                    self.operand_stack.append(left * right)
                elif opcode == OpCode.DIV:
                    if len(self.operand_stack) < 2:
                        raise IndexError("DIV requires two operands")
                    right, left = self.operand_stack.pop(), self.operand_stack.pop()
                    if right == 0:
                        raise ZeroDivisionError("Division by zero")
                    self.operand_stack.append(float(left) / right)
                elif opcode == OpCode.EQ:
                    if len(self.operand_stack) < 2:
                        raise IndexError("EQ requires two operands")
                    right, left = self.operand_stack.pop(), self.operand_stack.pop()
                    self.operand_stack.append(left == right)
                elif opcode == OpCode.LT:
                    if len(self.operand_stack) < 2:
                        raise IndexError("LT requires two operands")
                    right, left = self.operand_stack.pop(), self.operand_stack.pop()
                    self.operand_stack.append(left < right)
                elif opcode == OpCode.GT:
                    if len(self.operand_stack) < 2:
                        raise IndexError("GT requires two operands")
                    right, left = self.operand_stack.pop(), self.operand_stack.pop()
                    self.operand_stack.append(left > right)
                elif opcode == OpCode.NOT:
                    if not self.operand_stack:
                        raise IndexError("NOT requires one operand")
                    self.operand_stack.append(not self.operand_stack.pop())
                # --- Variable Access ---
                elif opcode == OpCode.LOAD:
                    self.operand_stack.append(self._lookup(args[0]))
                elif opcode == OpCode.STORE:
                    if not self.operand_stack:
                        raise IndexError(f"STORE '{args[0]}' requires a value")
                    self._store(args[0], self.operand_stack.pop())
                # --- Control Flow ---
                elif opcode == OpCode.JUMP:
                    self.ip = self._get_instruction_index(args[0])
                elif opcode == OpCode.JUMP_IF_FALSE:
                    if not self.operand_stack:
                        raise IndexError("JUMP_IF_FALSE requires a value")
                    if not self.operand_stack.pop():
                        self.ip = self._get_instruction_index(args[0])
                # --- Function Calls & Closures ---
                elif opcode == OpCode.MAKE_CLOSURE:
                    self.operand_stack.append(
                        Closure(code_label=args[0], defining_env=list(self.env_chain))
                    )
                elif opcode == OpCode.CALL:
                    arg_count = args[0]
                    if len(self.operand_stack) < arg_count + 1:
                        raise IndexError("CALL stack underflow")
                    callee = self.operand_stack.pop()
                    if not isinstance(callee, Closure):
                        self.operand_stack.append(callee)
                        raise TypeError(f"CALL expects Closure, got {type(callee)}")
                    # Args are already on stack below the callee, accessible by callee's STOREs
                    self.call_stack.append((self.ip, self.env_chain))
                    self.env_chain = callee.defining_env + [{}]
                    self.ip = self._get_instruction_index(callee.code_label)
                elif opcode == OpCode.RETURN:
                    if not self.call_stack:
                        self.ip = code_len
                        print("Warning: RETURN from top level.")
                        break
                    self.ip, self.env_chain = self.call_stack.pop()
                # --- Struct Operations ---
                elif opcode == OpCode.MAKE_STRUCT:
                    struct_name_str, field_names_tuple = args[0], args[1]
                    field_count = len(field_names_tuple)
                    if len(self.operand_stack) < field_count:
                        raise IndexError(
                            f"MAKE_STRUCT '{struct_name_str}' needs {field_count} values"
                        )
                    struct_instance = {"__type__": struct_name_str}
                    field_values = [
                        self.operand_stack.pop() for _ in range(field_count)
                    ][::-1]
                    for i, name in enumerate(field_names_tuple):
                        struct_instance[name] = field_values[i]
                    self.operand_stack.append(struct_instance)
                elif opcode == OpCode.GET_FIELD:
                    field_name_str = args[0]
                    if not self.operand_stack:
                        raise IndexError(f"GET_FIELD '{field_name_str}' needs struct")
                    instance = self.operand_stack.pop()
                    if not (isinstance(instance, dict) and "__type__" in instance):
                        self.operand_stack.append(instance)
                        raise TypeError(
                            f"GET_FIELD expects struct, got {type(instance)}"
                        )
                    if field_name_str not in instance:
                        self.operand_stack.append(instance)
                        raise AttributeError(
                            f"Struct {instance['__type__']} no field '{field_name_str}'"
                        )
                    self.operand_stack.append(instance[field_name_str])
                elif opcode == OpCode.SET_FIELD:
                    field_name_str = args[0]
                    if len(self.operand_stack) < 2:
                        raise IndexError(
                            f"SET_FIELD '{field_name_str}' needs value and struct"
                        )
                    new_value, instance = (
                        self.operand_stack.pop(),
                        self.operand_stack.pop(),
                    )
                    if not (isinstance(instance, dict) and "__type__" in instance):
                        self.operand_stack.extend([instance, new_value])
                        raise TypeError(
                            f"SET_FIELD expects struct, got {type(instance)}"
                        )
                    if field_name_str not in instance:
                        self.operand_stack.extend([instance, new_value])
                        raise AttributeError(
                            f"Struct {instance['__type__']} no field '{field_name_str}'"
                        )
                    instance[field_name_str] = new_value
                    self.operand_stack.append(instance)
                # --- List Primitives ---
                elif opcode == OpCode.IS_NIL:
                    if not self.operand_stack:
                        raise IndexError("IS_NIL requires one operand")
                    self.operand_stack.append(self.operand_stack.pop() is None)
                elif opcode == OpCode.CONS:
                    if len(self.operand_stack) < 2:
                        raise IndexError("CONS requires item and list")
                    item, lst = self.operand_stack.pop(), self.operand_stack.pop()
                    if lst is None:
                        self.operand_stack.append([item])
                    elif not isinstance(lst, list):
                        self.operand_stack.extend([lst, item])
                        raise TypeError(f"CONS expects list/nil, got {type(lst)}")
                    else:
                        self.operand_stack.append([item] + lst)
                elif opcode == OpCode.FIRST:
                    if not self.operand_stack:
                        raise IndexError("FIRST requires a list")
                    lst = self.operand_stack.pop()
                    if not (isinstance(lst, list) and lst):
                        self.operand_stack.append(lst)
                        raise TypeError(f"FIRST expects non-empty list, got {lst}")
                    self.operand_stack.append(lst[0])
                elif opcode == OpCode.REST:
                    if not self.operand_stack:
                        raise IndexError("REST requires a list")
                    lst = self.operand_stack.pop()
                    if not (isinstance(lst, list) and lst):
                        self.operand_stack.append(lst)
                        raise TypeError(f"REST expects non-empty list, got {lst}")
                    self.operand_stack.append(lst[1:] if len(lst) > 1 else None)
                elif opcode == OpCode.MAKE_LIST:
                    arg_count = args[0]
                    if len(self.operand_stack) < arg_count:
                        raise IndexError(f"MAKE_LIST needs {arg_count} items")
                    self.operand_stack.append(
                        [self.operand_stack.pop() for _ in range(arg_count)][::-1]
                    )
                # --- Type Predicates ---
                elif opcode == OpCode.IS_BOOLEAN:
                    if not self.operand_stack:
                        raise IndexError("IS_BOOLEAN requires one operand")
                    self.operand_stack.append(
                        isinstance(self.operand_stack.pop(), bool)
                    )
                elif opcode == OpCode.IS_NUMBER:
                    if not self.operand_stack:
                        raise IndexError("IS_NUMBER requires one operand")
                    self.operand_stack.append(
                        isinstance(self.operand_stack.pop(), (int, float))
                    )
                elif opcode == OpCode.IS_STRING:
                    if not self.operand_stack:
                        raise IndexError("IS_STRING requires one operand")
                    self.operand_stack.append(isinstance(self.operand_stack.pop(), str))
                elif opcode == OpCode.IS_LIST:
                    if not self.operand_stack:
                        raise IndexError("IS_LIST requires one operand")
                    val = self.operand_stack.pop()
                    self.operand_stack.append(isinstance(val, list) or val is None)
                elif opcode == OpCode.IS_STRUCT:
                    if not self.operand_stack:
                        raise IndexError("IS_STRUCT requires one operand")
                    val = self.operand_stack.pop()
                    self.operand_stack.append(
                        isinstance(val, dict) and "__type__" in val
                    )
                elif opcode == OpCode.IS_FUNCTION:
                    if not self.operand_stack:
                        raise IndexError("IS_FUNCTION requires one operand")
                    self.operand_stack.append(
                        isinstance(self.operand_stack.pop(), Closure)
                    )
                # --- VM Control ---
                elif opcode == OpCode.HALT:
                    print("Execution halted.")
                    self.ip = code_len
                    break
                elif opcode == OpCode.PRINT:
                    if not self.operand_stack:
                        raise IndexError("PRINT requires a value")
                    print("Output:", self.operand_stack.pop())
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
    print("To run tests, please execute 'vm_tests.py'.")
    # Minimal test to ensure it runs
    test_code = [(OpCode.PUSH, 1), (OpCode.PRINT,), (OpCode.HALT,)]
    VirtualMachine(test_code).run()
