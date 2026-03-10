from toy_tasks.tasks.caesar_cipher import caesar_cipher


def test_basic_shift():
    assert caesar_cipher("abc", 3) == "def"


def test_wrap_around_lower():
    assert caesar_cipher("xyz", 3) == "abc"


def test_wrap_around_upper():
    assert caesar_cipher("XYZ", 3) == "ABC"


def test_preserves_case():
    assert caesar_cipher("Hello", 1) == "Ifmmp"


def test_preserves_non_alpha():
    assert caesar_cipher("Hello, World!", 5) == "Mjqqt, Btwqi!"


def test_zero_shift():
    assert caesar_cipher("test", 0) == "test"


def test_full_rotation():
    assert caesar_cipher("abc", 26) == "abc"


def test_negative_shift():
    assert caesar_cipher("def", -3) == "abc"
