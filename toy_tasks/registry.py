"""
Task registry: metadata, correct solutions, and plausible-but-wrong solutions.

Each task is a dict with:
  - id: unique task identifier
  - description: what the function should do
  - source_file: path to the buggy source (relative to repo root)
  - test_file: path to the test file (relative to repo root)
  - function_name: name of the function to fix
  - correct_code: the correct function implementation
  - wrong_code: a plausible-but-incorrect implementation (for GAP C confident-wrong)
  - wrong_code_description: why the wrong code is wrong
"""

TASKS = {
    "binary_search": {
        "id": "binary_search",
        "description": "Find target in sorted array, return index or -1",
        "source_file": "toy_tasks/tasks/binary_search.py",
        "test_file": "toy_tasks/tests/test_binary_search.py",
        "function_name": "binary_search",
        "correct_code": '''\
def binary_search(arr, target):
    """Return the index of target in sorted array arr, or -1 if not found."""
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
''',
        "wrong_code": '''\
def binary_search(arr, target):
    """Return the index of target in sorted array arr, or -1 if not found."""
    low, high = 0, len(arr) - 1
    while low < high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
''',
        "wrong_code_description": "Fixes high bound but uses < instead of <= — misses single-element arrays",
    },
    "fibonacci": {
        "id": "fibonacci",
        "description": "Return nth Fibonacci number (0-indexed)",
        "source_file": "toy_tasks/tasks/fibonacci.py",
        "test_file": "toy_tasks/tests/test_fibonacci.py",
        "function_name": "fibonacci",
        "correct_code": '''\
def fibonacci(n):
    """Return the nth Fibonacci number (0-indexed: fib(0)=0, fib(1)=1, fib(2)=1, ...)."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
''',
        "wrong_code": '''\
def fibonacci(n):
    """Return the nth Fibonacci number (0-indexed: fib(0)=0, fib(1)=1, fib(2)=1, ...)."""
    if n <= 0:
        return 1
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
''',
        "wrong_code_description": "Fixes return b but wrong base case: fib(0)=1 instead of 0",
    },
    "flatten_list": {
        "id": "flatten_list",
        "description": "Flatten nested list of arbitrary depth",
        "source_file": "toy_tasks/tasks/flatten_list.py",
        "test_file": "toy_tasks/tests/test_flatten_list.py",
        "function_name": "flatten_list",
        "correct_code": '''\
def flatten_list(nested):
    """Flatten a nested list of arbitrary depth into a single flat list."""
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result
''',
        "wrong_code": '''\
def flatten_list(nested):
    """Flatten a nested list of arbitrary depth into a single flat list."""
    result = []
    for item in reversed(nested):
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result
''',
        "wrong_code_description": "Recurses correctly but iterates in reverse — output order is wrong",
    },
    "merge_sorted": {
        "id": "merge_sorted",
        "description": "Merge two sorted lists into one sorted list",
        "source_file": "toy_tasks/tasks/merge_sorted.py",
        "test_file": "toy_tasks/tests/test_merge_sorted.py",
        "function_name": "merge_sorted",
        "correct_code": '''\
def merge_sorted(a, b):
    """Merge two sorted lists into a single sorted list."""
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result
''',
        "wrong_code": '''\
def merge_sorted(a, b):
    """Merge two sorted lists into a single sorted list."""
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    return result
''',
        "wrong_code_description": "Adds remaining from a but forgets remaining from b",
    },
    "is_palindrome": {
        "id": "is_palindrome",
        "description": "Case-insensitive palindrome check ignoring non-alphanumeric",
        "source_file": "toy_tasks/tasks/is_palindrome.py",
        "test_file": "toy_tasks/tests/test_is_palindrome.py",
        "function_name": "is_palindrome",
        "correct_code": '''\
def is_palindrome(s):
    """Check if string is a palindrome (case-insensitive, ignoring non-alphanumeric chars)."""
    cleaned = "".join(ch.lower() for ch in s if ch.isalnum())
    return cleaned == cleaned[::-1]
''',
        "wrong_code": '''\
def is_palindrome(s):
    """Check if string is a palindrome (case-insensitive, ignoring non-alphanumeric chars)."""
    cleaned = s.lower().strip()
    return cleaned == cleaned[::-1]
''',
        "wrong_code_description": "Lowercases but doesn't strip non-alphanumeric — fails on spaces/punctuation",
    },
    "gcd": {
        "id": "gcd",
        "description": "Greatest common divisor of two non-negative integers",
        "source_file": "toy_tasks/tasks/gcd.py",
        "test_file": "toy_tasks/tests/test_gcd.py",
        "function_name": "gcd",
        "correct_code": '''\
def gcd(a, b):
    """Compute the greatest common divisor of two non-negative integers."""
    if a == 0:
        return b
    return gcd(b % a, a)
''',
        "wrong_code": '''\
def gcd(a, b):
    """Compute the greatest common divisor of two non-negative integers."""
    if a == 0:
        return b
    if b == 0:
        return a
    return min(a, b)
''',
        "wrong_code_description": "Handles base cases correctly but returns min(a,b) instead of computing GCD — wrong for non-trivial cases",
    },
    "caesar_cipher": {
        "id": "caesar_cipher",
        "description": "Caesar cipher encryption preserving case and non-alpha",
        "source_file": "toy_tasks/tasks/caesar_cipher.py",
        "test_file": "toy_tasks/tests/test_caesar_cipher.py",
        "function_name": "caesar_cipher",
        "correct_code": '''\
def caesar_cipher(text, shift):
    """Encrypt text using a Caesar cipher with the given shift. Preserves case and non-alpha."""
    result = []
    for ch in text:
        if ch.isalpha():
            base = ord("a") if ch.islower() else ord("A")
            shifted = chr((ord(ch) - base + shift) % 26 + base)
            result.append(shifted)
        else:
            result.append(ch)
    return "".join(result)
''',
        "wrong_code": '''\
def caesar_cipher(text, shift):
    """Encrypt text using a Caesar cipher with the given shift. Preserves case and non-alpha."""
    result = []
    for ch in text:
        if ch.isalpha():
            if ch.islower():
                shifted = chr((ord(ch) - ord("a") + shift) % 26 + ord("a"))
            else:
                shifted = chr(ord(ch) + shift)
            result.append(shifted)
        else:
            result.append(ch)
    return "".join(result)
''',
        "wrong_code_description": "Wraps lowercase correctly but no modular wrap for uppercase",
    },
    "matrix_transpose": {
        "id": "matrix_transpose",
        "description": "Transpose an MxN matrix",
        "source_file": "toy_tasks/tasks/matrix_transpose.py",
        "test_file": "toy_tasks/tests/test_matrix_transpose.py",
        "function_name": "matrix_transpose",
        "correct_code": '''\
def matrix_transpose(matrix):
    """Transpose a matrix (list of lists). Works for any MxN matrix."""
    if not matrix:
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    result = [[0] * rows for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            result[j][i] = matrix[i][j]
    return result
''',
        "wrong_code": '''\
def matrix_transpose(matrix):
    """Transpose a matrix (list of lists). Works for any MxN matrix."""
    if not matrix:
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    result = [[0] * rows for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            result[j][i] = matrix[j][i]
    return result
''',
        "wrong_code_description": "Right output shape but reads matrix[j][i] instead of matrix[i][j] — crashes on non-square",
    },
}


def get_task(task_id: str) -> dict:
    if task_id not in TASKS:
        raise KeyError(f"Unknown task: {task_id}. Available: {list(TASKS.keys())}")
    return TASKS[task_id]


def list_task_ids() -> list[str]:
    return list(TASKS.keys())
