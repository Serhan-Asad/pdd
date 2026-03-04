def add_numbers(num1: int, num2: int) -> int:
    """Adds two integers after validating input types."""
    if not isinstance(num1, int) or not isinstance(num2, int):
        raise TypeError("Both inputs must be integers.")
    return num1 + num2

if __name__ == '__main__':
    result = add_numbers(5, 3)
    print("The sum is:", result)