# Main test runner for the entire NotScheme project.
# This script imports and runs test suites from other test files.

import os
import traceback

from src.tests.test_lexer import run_lexer_tests
from src.tests.test_parser import run_parser_tests
from src.tests.test_codegen import run_codegen_tests
from src.tests.test_language import run_language_feature_tests
from src.tests.test_vm import run_all_vm_tests
from src.tests.test_speed import run_all_speed_tests


def main():
    """Runs all NotScheme test suites."""
    print("=============================================")
    print("====== Running All NotScheme Test Suites ======")
    print("=============================================\n")

    # Store current working directory to restore it later if tests change it
    original_cwd = os.getcwd()
    # Determine the directory of this script (src/)
    # This helps if tests create files and expect them relative to src/ or project root.
    # For now, language tests create files in os.getcwd(). If test_all.py is run from root,
    # this means files are created in root. If run from src/, in src/.
    # This might need more robust path handling for test file creation/cleanup.

    try:
        print("\n\n>>> Running Lexer Tests <<<")
        run_lexer_tests()

        print("\n\n>>> Running Parser Tests <<<")
        run_parser_tests()

        print("\n\n>>> Running Code Generator Tests <<<")
        # Codegen tests might create math_utils.ns, string_ext.ns in CWD.
        # If CWD is src/, they are created in src/.
        run_codegen_tests()

        print("\n\n>>> Running Language Feature Tests <<<")
        # Language tests also create .ns files in CWD.
        run_language_feature_tests()

        print("\n\n>>> Running Virtual Machine (VM) Tests <<<")
        run_all_vm_tests()

        print("\n\n>>> Running Speed/Performance Tests <<<")
        # Speed tests also create .ns files in CWD.
        run_all_speed_tests()

    except Exception as e:
        print(f"\n\n!!!!!! An error occurred during test execution: {e} !!!!!!")
        traceback.print_exc()
    finally:
        # Restore CWD if it was changed by any test.
        # This is a basic safety measure; ideally tests should clean up their CWD changes.
        if os.getcwd() != original_cwd:
            print(
                f"\nWarning: Current working directory changed. Restoring to: {original_cwd}"
            )
            os.chdir(original_cwd)

        # General cleanup of .ns files that might be left in CWD if tests failed before cleanup.
        # This is a bit broad. Better if each test suite cleans its own specific files.
        # For now, this is a fallback.
        # files_to_cleanup = [f for f in os.listdir(original_cwd) if f.endswith(".ns")]
        # if files_to_cleanup:
        #     print(f"\nAttempting cleanup of .ns files in {original_cwd} (if any)...")
        #     for f_to_clean in files_to_cleanup:
        #         # Be careful not to delete source .ns files if tests are run from src/
        #         # This cleanup is risky without more specific naming for test-generated files.
        #         # For now, relying on individual test cleanup.
        #         pass

    print("\n\n=============================================")
    print("====== All Test Suites Completed ======")
    print("=============================================")


if __name__ == "__main__":
    # If running `python src/test_all.py` from project root, imports should work.
    # If running `python test_all.py` from `src/` directory, imports should also work.
    main()
