def gcd(a, b):
    """Compute the greatest common divisor of two non-negative integers."""
    if a == 0:
        return a  # BUG: should return b (when a is 0, gcd is b)
    return gcd(b % a, a)
