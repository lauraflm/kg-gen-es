from dotenv import load_dotenv
from src.kg_gen import KGGen
import os
import pytest

load_dotenv()


@pytest.fixture
def kg(test_api_key):
    return KGGen(
        model=os.getenv("LLM_MODEL", "openai/gpt-5-nano"),
        api_key=test_api_key,
        temperature=float(os.getenv("LLM_TEMPERATURE", "1.0")),
        retrieval_model=os.getenv("RETRIEVAL_MODEL", "all-MiniLM-L6-v2"),
    )
