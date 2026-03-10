def flatten_list(nested):
    """Flatten a nested list of arbitrary depth into a single flat list."""
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(item)  # BUG: doesn't recurse — only flattens one level
        else:
            result.append(item)
    return result
