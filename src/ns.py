# End-to-end pipeline for lexing, parsing, compiling, and running NotScheme code.
# Can be used as a CLI to run .ns files or to run internal tests.

import io
import sys
import os
from typing import Any, List, Dict, Set

from src.lexer import tokenize
from src.parser import Parser
from src.codegen import CodeGenerator
from src.vm import VirtualMachine, OpCode


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
        print(
            "No file provided to run. To run a NotScheme file, use: python src/run_notscheme.py <file.ns>"
        )
        print("To run all internal tests, use: python tests/test_all.py")
