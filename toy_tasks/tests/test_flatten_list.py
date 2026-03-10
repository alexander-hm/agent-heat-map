from toy_tasks.tasks.flatten_list import flatten_list


def test_flat():
    assert flatten_list([1, 2, 3]) == [1, 2, 3]


def test_one_level():
    assert flatten_list([1, [2, 3], 4]) == [1, 2, 3, 4]


def test_deep_nesting():
    assert flatten_list([1, [2, [3, [4]]]]) == [1, 2, 3, 4]


def test_empty():
    assert flatten_list([]) == []


def test_all_nested():
    assert flatten_list([[1, 2], [3, 4]]) == [1, 2, 3, 4]


def test_mixed_depth():
    assert flatten_list([1, [2, [3]], 4, [[5]]]) == [1, 2, 3, 4, 5]
