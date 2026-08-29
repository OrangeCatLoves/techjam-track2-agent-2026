"""Shared fixtures.

OWNS the one expensive thing in the test suite: the loaded splits. The load takes
about five seconds and several hundred megabytes, so it happens once per session.

MUST NEVER hand a test a hidden-test label. The fixture returns exactly what
``harness.data.load`` returns, which is where the strip already happened.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness import data as hdata  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: minutes-long job')
    config.addinivalue_line('markers', 'data: needs the KuaiRand-Pure CSVs')


@pytest.fixture(scope='session')
def data_dir() -> Path:
    """The configured dataset directory, skipping the test if it is absent."""
    try:
        path = hdata.data_dir()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    if not path.exists():
        pytest.skip(f'KuaiRand-Pure data directory not found: {path}')
    return path


@pytest.fixture(scope='session')
def splits(data_dir):
    """The official splits, label-stripped on test. Loaded once per session."""
    return hdata.load(data_dir)


@pytest.fixture(scope='session')
def expected_rows() -> dict:
    """Row counts from configs/base.yaml, keyed by split name."""
    rows = hdata.load_config()['dataset']['expected_rows']
    return {'train': rows['train'], 'valid': rows['val'], 'test': rows['test']}
