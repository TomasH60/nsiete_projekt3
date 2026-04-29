import pytest

from router.labels import FINANCE, HEALTH, LAW, OOD, normalize_label


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("law", LAW),
        ("legal", LAW),
        ("finance", FINANCE),
        ("fintech", FINANCE),
        ("health", HEALTH),
        ("healthcare", HEALTH),
        ("other", OOD),
        ("3", OOD),
    ],
)
def test_normalize_label(raw: object, expected: int) -> None:
    assert normalize_label(raw) == expected
