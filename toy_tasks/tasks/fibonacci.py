def fibonacci(n):
    """Return the nth Fibonacci number (0-indexed: fib(0)=0, fib(1)=1, fib(2)=1, ...)."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return a  # BUG: should return b
