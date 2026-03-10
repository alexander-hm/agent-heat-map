from toy_tasks.tasks.merge_sorted import merge_sorted


def test_basic():
    assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]


def test_one_empty():
    assert merge_sorted([], [1, 2, 3]) == [1, 2, 3]


def test_both_empty():
    assert merge_sorted([], []) == []


def test_unequal_lengths():
    assert merge_sorted([1, 2], [3, 4, 5, 6]) == [1, 2, 3, 4, 5, 6]


def test_interleaved():
    assert merge_sorted([1, 4, 7], [2, 3, 5]) == [1, 2, 3, 4, 5, 7]


def test_duplicates():
    assert merge_sorted([1, 3, 3], [2, 3, 4]) == [1, 2, 3, 3, 3, 4]
