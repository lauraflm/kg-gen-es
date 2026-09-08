"""End-to-end Spanish extraction test.

Hits a real LLM, so it depends on the `gemini_api_key` fixture and skips
when that key is not configured. Assertions stay loose on purpose: the
model's exact output is not reproducible, so the test checks that Spanish
extraction produced a usable graph, not that it produced one specific graph.
"""

import pytest

from src.kg_gen import KGGen

TEXT = (
    "Gabriel García Márquez nació en Aracataca, Colombia. "
    "Es el autor de Cien años de soledad."
)


@pytest.fixture(scope="module")
def kg_es(gemini_api_key):
    return KGGen(
        model="gemini/gemini-2.5-flash",
        api_key=gemini_api_key,
        retrieval_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        language="es",
        semhash_model_name="minishlab/potion-multilingual-128M",
    )


def test_spanish_extraction(kg_es):
    graph = kg_es.generate(TEXT)

    assert graph.entities, "no entities extracted"
    assert graph.relations, "no relations extracted"

    # The author is the subject of the passage, so some entity should carry
    # his name regardless of how the model chose to split or accent it.
    assert any("rquez" in entity for entity in graph.entities), (
        f"expected an entity for García Márquez, got: {graph.entities}"
    )

    # Every relation must reference known entities and edges.
    for subject, predicate, obj in graph.relations:
        assert subject in graph.entities
        assert obj in graph.entities
        assert predicate in graph.edges
