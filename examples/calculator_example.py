import sys
import os

# Ensure the module's directory is on the Python path (relative to this script)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from add_numbers import add_numbers


def main():
    """
    Demonstrates usage of the add_numbers function.

    add_numbers(num1: int, num2: int) -> int
        Adds two integers after validating that both inputs are of type int.

        Parameters
        ----------
        num1 : int
            The first integer to add.
        num2 : int
            The second integer to add.

        Returns
        -------
        int
            The sum of num1 and num2.

        Raises
        ------
        TypeError
            If either num1 or num2 is not an integer.
    """

    # --- Basic usage ---
    result = add_numbers(10, 20)
    print(f"add_numbers(10, 20) = {result}")  # Output: 30

    # --- Adding negative numbers ---
    result_neg = add_numbers(-5, 3)
    print(f"add_numbers(-5, 3)  = {result_neg}")  # Output: -2

    # --- Adding zeros ---
    result_zero = add_numbers(0, 0)
    print(f"add_numbers(0, 0)   = {result_zero}")  # Output: 0

    # --- Error handling: passing a non-integer raises TypeError ---
    try:
        add_numbers(2.5, 3)
    except TypeError as e:
        print(f"\nExpected error when passing a float: {e}")

    try:
        add_numbers("1", 2)
    except TypeError as e:
        print(f"Expected error when passing a string: {e}")


if __name__ == "__main__":
    main()