"""
Pytest configuration and shared fixtures for the kg-gen tests.

Follows the same pattern as mcp/tests/conftest.py: tests that need a live
model depend on an API-key fixture, which skips when the key is missing
instead of letting the test fail with an authentication error.
"""

import os
import pytest
from dotenv import load_dotenv
from huggingface_hub import try_to_load_from_cache
from model2vec import StaticModel

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


@pytest.fixture(scope="session")
def semhash_encoders():
    """Load a SemHash encoder by name, once, or skip if it is not cached.

    Tests that judge real SemHash similarity use this instead of
    `_load_semhash_model`, which downloads a missing model.
    """
    loaded = {}

    def load(name):
        if name not in loaded:
            config = try_to_load_from_cache(name, "config.json")
            if not isinstance(config, str):
                pytest.skip(f"{name} is not in the local Hugging Face cache")
            # A local path: model2vec reads it without going to the Hub.
            loaded[name] = StaticModel.from_pretrained(os.path.dirname(config))
        return loaded[name]

    return load
