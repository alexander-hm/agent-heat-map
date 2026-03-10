from toy_tasks.tasks.is_palindrome import is_palindrome


def test_simple_yes():
    assert is_palindrome("racecar") is True


def test_simple_no():
    assert is_palindrome("hello") is False


def test_mixed_case():
    assert is_palindrome("Racecar") is True


def test_with_spaces():
    assert is_palindrome("a man a plan a canal panama") is True


def test_with_punctuation():
    assert is_palindrome("A man, a plan, a canal: Panama!") is True


def test_empty():
    assert is_palindrome("") is True


def test_single_char():
    assert is_palindrome("a") is True
