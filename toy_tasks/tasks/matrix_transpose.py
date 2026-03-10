def matrix_transpose(matrix):
    """Transpose a matrix (list of lists). Works for any MxN matrix."""
    if not matrix:
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    # BUG: creates rows x rows instead of cols x rows — crashes on non-square input
    result = [[0] * rows for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            result[j][i] = matrix[i][j]
    return result
