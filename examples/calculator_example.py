"""
Example usage of the calculator module.
This demonstrates the intended interface before implementation exists.
"""

from calculator import add, subtract, divide

# Basic addition
result = add(5, 3)
print(f"5 + 3 = {result}")  # Should print: 5 + 3 = 8

# Basic subtraction
result = subtract(10, 4)
print(f"10 - 4 = {result}")  # Should print: 10 - 4 = 6

# Basic division
result = divide(10, 2)
print(f"10 / 2 = {result}")  # Should print: 10 / 2 = 5.0

# Float operations
result = add(2.5, 3.7)
print(f"2.5 + 3.7 = {result}")  # Should print: 2.5 + 3.7 = 6.2

# Error handling - division by zero
try:
    result = divide(10, 0)
    print(f"This shouldn't print")
except ValueError as e:
    print(f"Caught expected error: {e}")  # Should print: Caught expected error: Cannot divide by zero
