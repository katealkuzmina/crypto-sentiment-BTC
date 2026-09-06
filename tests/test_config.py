import pytest

from src.config import get_env


def test_get_env_returns_value(monkeypatch):
    monkeypatch.setenv("SOME_TEST_VAR", "hello")
    assert get_env("SOME_TEST_VAR") == "hello"


def test_get_env_raises_when_missing(monkeypatch):
    monkeypatch.delenv("MISSING_TEST_VAR", raising=False)
    with pytest.raises(RuntimeError, match="MISSING_TEST_VAR"):
        get_env("MISSING_TEST_VAR")
