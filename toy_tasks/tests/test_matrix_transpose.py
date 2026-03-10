from toy_tasks.tasks.matrix_transpose import matrix_transpose


def test_square():
    m = [[1, 2], [3, 4]]
    assert matrix_transpose(m) == [[1, 3], [2, 4]]


def test_rectangular_wide():
    m = [[1, 2, 3], [4, 5, 6]]
    assert matrix_transpose(m) == [[1, 4], [2, 5], [3, 6]]


def test_rectangular_tall():
    m = [[1, 2], [3, 4], [5, 6]]
    assert matrix_transpose(m) == [[1, 3, 5], [2, 4, 6]]


def test_single_row():
    m = [[1, 2, 3]]
    assert matrix_transpose(m) == [[1], [2], [3]]


def test_single_col():
    m = [[1], [2], [3]]
    assert matrix_transpose(m) == [[1, 2, 3]]


def test_empty():
    assert matrix_transpose([]) == []


def test_single_element():
    assert matrix_transpose([[5]]) == [[5]]
