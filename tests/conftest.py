"""Shared fixtures: a small configuration that trains in well under a second."""
import pytest

from alignment.config import BASE


@pytest.fixture
def small_cfg():
    """Base configuration shrunk to a few hundred iterations and a handful of filters."""
    return dict(BASE, D=12, P=3, n=10, h=6, iters=400, log_every=20)
