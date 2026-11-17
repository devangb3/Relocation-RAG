import sys


def test_python_meets_project_requirement() -> None:
    assert sys.version_info >= (3, 13)
