from src.hash_utils import hash_strings


def test_hash_strings_deterministic():
    values = ["b", "a", "c"]
    assert hash_strings(values) == hash_strings(reversed(values))
