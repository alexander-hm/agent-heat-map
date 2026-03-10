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
    # BUG: missing extend of remaining elements from whichever list isn't exhausted
    return result
