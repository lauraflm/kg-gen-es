"""
Pytest configuration and shared fixtures for the kg-gen tests.

Follows the same pattern as mcp/tests/conftest.py: tests that need a live
model depend on an API-key fixture, which skips when the key is missing
instead of letting the test fail with an authentication error.
"""

import os
import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(scope="session")
def test_api_key():
    """Get API key for testing, skipping the test if missing."""
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        pytest.skip("LLM_API_KEY environment variable not set")
    return api_key


@pytest.fixture(scope="session")
def gemini_api_key():
    """Get the Gemini API key used by the Spanish tests, skipping if missing."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY environment variable not set")
    return api_key
