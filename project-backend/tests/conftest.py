"""Pytest configuration and global fixtures for project-backend tests.
Enforces test isolation: tests must never write to production rejects.jsonl or InfluxDB.
"""
from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def isolate_test_rejects(monkeypatch, tmp_path):
    """Isolates rejects logging to a temporary path for every test run."""
    temp_rejects = tmp_path / "test_rejects.jsonl"
    monkeypatch.setenv("REJECTS_PATH", str(temp_rejects))
