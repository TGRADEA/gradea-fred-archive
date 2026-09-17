"""Shared pytest fixtures."""

import os
import sys
from pathlib import Path

import pytest

# Make backend modules importable from tests/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Use a temp DB during tests, never the real one. Set BEFORE importing
# settings so the Pydantic Settings singleton reads it on first init.
_TEST_DB = _ROOT / "tests" / ".test.db"
os.environ["GRADEA_DB_PATH"] = str(_TEST_DB)


@pytest.fixture(autouse=True)
def _clean_cache():
    """Drop every TTL-cache entry before each test (FOUNDATION-CACHE-1).

    Without this, a test that patches a data source gets the *previous* test's
    cached answer: `test_regime_contango` warms `signals.market_regime`, then
    `test_regime_backwardation` patches the source, calls the same function,
    and is served the contango result from cache. The failure looks like a
    logic bug in the regime classifier and is genuinely confusing to chase.

    Autouse and cheap — it clears dicts, no I/O.
    """
    import cache

    cache.reset_for_tests()
    yield
    cache.reset_for_tests()


@pytest.fixture(autouse=True)
def _clean_test_db():
    """Wipe the test DB before every test so order doesn't matter."""
    if _TEST_DB.exists():
        _TEST_DB.unlink()
    yield
    if _TEST_DB.exists():
        _TEST_DB.unlink()
