"""Guided extraction: competency questions reach entity and relation
extraction, and unguided prompts stay exactly as before."""

import pytest

from kg_gen.steps._1_get_entities import _entities_sig, get_entities
from kg_gen.steps._2_get_relations import _relation_model, extraction_sig, get_relations


@pytest.mark.parametrize("language", ["en", "es"])
def test_unguided_entity_signature_is_unchanged(language):
    plain = _entities_sig(False, language)
    assert list(plain.input_fields) == ["source_text"]
    assert "competency" not in plain.instructions.lower()
    assert "competencia" not in plain.instructions.lower()


@pytest.mark.parametrize("language,word", [("en", "competency"), ("es", "competencia")])
def test_guided_entity_signature_adds_questions(language, word):
    plain = _entities_sig(False, language)
    guided = _entities_sig(False, language, guided=True)
    assert list(guided.input_fields) == ["source_text", "guiding_questions"]
    assert list(guided.output_fields) == list(plain.output_fields)
    assert guided.instructions.startswith(plain.instructions.strip()[:40])
    assert word in guided.instructions.lower()


@pytest.mark.parametrize("language,word", [("en", "competency"), ("es", "competencia")])
def test_guided_relation_signature_adds_questions(language, word):
    Relation = _relation_model(language)
    plain = extraction_sig(Relation, False, "", language)
    guided = extraction_sig(Relation, False, "", language, guided=True)
    assert list(plain.input_fields) == ["source_text", "entities"]
    assert list(guided.input_fields) == ["source_text", "entities", "guiding_questions"]
    assert list(guided.output_fields) == list(plain.output_fields)
    assert word not in plain.instructions.lower()
    assert word in guided.instructions.lower()


def test_guidance_is_not_supported_on_the_litellm_route():
    with pytest.raises(NotImplementedError):
        get_entities("text", use_litellm_prompt=True, guiding_questions=["q?"])
    with pytest.raises(NotImplementedError):
        get_relations("text", ["a"], use_litellm_prompt=True, guiding_questions=["q?"])
