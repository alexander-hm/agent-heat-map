def is_palindrome(s):
    """Check if string is a palindrome (case-insensitive, ignoring non-alphanumeric chars)."""
    cleaned = s.strip()  # BUG: doesn't lowercase or remove non-alphanumeric characters
    return cleaned == cleaned[::-1]
