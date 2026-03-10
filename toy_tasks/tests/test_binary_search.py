from toy_tasks.tasks.binary_search import binary_search


def test_found_middle():
    assert binary_search([1, 3, 5, 7, 9], 5) == 2


def test_found_first():
    assert binary_search([1, 3, 5, 7, 9], 1) == 0


def test_found_last():
    assert binary_search([1, 3, 5, 7, 9], 9) == 4


def test_not_found():
    assert binary_search([1, 3, 5, 7, 9], 4) == -1


def test_not_found_above_max():
    assert binary_search([1, 3, 5, 7, 9], 10) == -1


def test_empty():
    assert binary_search([], 1) == -1


def test_single_found():
    assert binary_search([5], 5) == 0


def test_single_not_found():
    assert binary_search([5], 3) == -1
