def binary_search(arr, target):
    """Return the index of target in sorted array arr, or -1 if not found."""
    low, high = 0, len(arr)  # BUG: should be len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
