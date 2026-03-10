from toy_tasks.tasks.gcd import gcd


def test_basic():
    assert gcd(12, 8) == 4


def test_coprime():
    assert gcd(7, 13) == 1


def test_same():
    assert gcd(5, 5) == 5


def test_one_zero():
    assert gcd(0, 5) == 5


def test_other_zero():
    assert gcd(7, 0) == 7


def test_both_zero():
    assert gcd(0, 0) == 0


def test_large():
    assert gcd(48, 18) == 6
