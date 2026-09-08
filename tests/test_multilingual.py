"""Offline tests for the multilingual support added on top of kg-gen.

None of these call an LLM or load spaCy, so they run everywhere and take
milliseconds.
"""

import pytest

from src.kg_gen.steps._1_get_entities import _entities_sig
from src.kg_gen.steps._2_get_relations import _relation_model, extraction_sig
from src.kg_gen.utils.deduplicate import _representative_sort_key
from src.kg_gen.utils.i18n import pick, validate_language

# --------------------------------------------------------------------------
# English prompt parity
#
# The expected strings below are written out by hand rather than read from
# the prompt tables. That duplication is the point: it is what catches an
# accidental edit to an "en" value. Changing a prompt for English is a real
# behavior change against upstream kg-gen, so it should have to be made in
# two places, deliberately.
# --------------------------------------------------------------------------

ENTITIES_TEXT = (
    "Extract key entities from the source text. Extracted entities are "
    "subjects or objects.\nThis is for an extraction task, please be "
    "THOROUGH and accurate to the reference text."
)

ENTITIES_CONVERSATION = (
    "Extract key entities from the conversation Extracted entities are "
    "subjects or objects.\nConsider both explicit entities and participants "
    "in the conversation.\nThis is for an extraction task, please be "
    "THOROUGH and accurate."
)

RELATIONS_TEXT = (
    "Extract subject-predicate-object triples from the source text.\n"
    "Subject and object must be from entities list. Entities provided were "
    "previously extracted from the same source text.\nThis is for an "
    "extraction task, please be thorough, accurate, and faithful to the "
    "reference text. "
)

RELATIONS_CONVERSATION = (
    "Extract subject-predicate-object triples from the conversation, "
    "including:\n1. Relations between concepts discussed\n2. Relations "
    "between speakers and concepts (e.g. user asks about X)\n3. Relations "
    "between speakers (e.g. assistant responds to user)\nSubject and object "
    "must be from entities list. Entities provided were previously extracted "
    "from the same source text.\nThis is for an extraction task, please be "
    "thorough, accurate, and faithful to the reference text. "
)


def test_english_entity_instructions_unchanged():
    assert _entities_sig(False, "en").instructions == ENTITIES_TEXT
    assert _entities_sig(True, "en").instructions == ENTITIES_CONVERSATION


def test_english_entity_output_desc_unchanged():
    desc = _entities_sig(False, "en").output_fields["entities"]
    assert desc.json_schema_extra["desc"] == "THOROUGH list of key entities"


def test_english_relation_instructions_unchanged():
    relation = _relation_model("en")
    assert extraction_sig(relation, False, "", "en").instructions == RELATIONS_TEXT
    assert (
        extraction_sig(relation, True, "", "en").instructions
        == RELATIONS_CONVERSATION
    )


def test_english_relation_output_desc_unchanged():
    relation = _relation_model("en")
    text_desc = extraction_sig(relation, False, "", "en").output_fields["relations"]
    conv_desc = extraction_sig(relation, True, "", "en").output_fields["relations"]
    assert (
        text_desc.json_schema_extra["desc"]
        == "List of subject-predicate-object tuples. Be thorough."
    )
    assert conv_desc.json_schema_extra["desc"] == (
        "List of subject-predicate-object tuples where subject and object "
        "are exact matches to items in entities list. Be thorough"
    )


def test_english_relation_model_unchanged():
    """The field examples ride along in the JSON schema sent to the model,
    so they are part of the prompt, not just documentation."""
    relation = _relation_model("en")
    assert relation.__doc__ == "Knowledge graph subject-predicate-object tuple."

    fields = relation.model_fields
    assert fields["subject"].json_schema_extra["desc"] == "Subject entity"
    assert fields["predicate"].json_schema_extra["desc"] == "Predicate"
    assert fields["object"].json_schema_extra["desc"] == "Object entity"
    assert fields["subject"].examples == ["Kevin"]
    assert fields["predicate"].examples == ["is brother of"]
    assert fields["object"].examples == ["Vicky"]


def test_spanish_prompts_actually_differ():
    """Guards against a table silently falling back to English."""
    relation_en = _relation_model("en")
    relation_es = _relation_model("es")

    assert _entities_sig(False, "es").instructions != ENTITIES_TEXT
    assert extraction_sig(relation_es, False, "", "es").instructions != RELATIONS_TEXT
    assert relation_es.model_fields["predicate"].examples == ["es hermano de"]
    assert relation_en.model_fields["predicate"].examples == ["is brother of"]


def test_unknown_language_falls_back_to_english():
    """`pick` degrades to English; only `validate_language` rejects."""
    assert _entities_sig(False, "fr").instructions == ENTITIES_TEXT
    assert pick({"en": "fallback", "es": "respaldo"}, "fr") == "fallback"


# --------------------------------------------------------------------------
# Representative selection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variants, expected",
    [
        # Accents must survive: sorting by codepoint alone prefers "Garcia".
        ({"Gabriel García Márquez", "Gabriel Garcia Marquez"}, "Gabriel García Márquez"),
        ({"México", "MÉXICO", "méxico"}, "México"),
        ({"BOGOTÁ", "Bogotá"}, "Bogotá"),
        # Mixed case beats shouting, in English too.
        ({"PARIS", "Paris"}, "Paris"),
        ({"united states", "United States"}, "United States"),
        # Acronyms have no mixed-case variant to lose to, so they stay upper.
        ({"ONU", "onu"}, "ONU"),
        ({"ONU"}, "ONU"),
        # Shorter wins once case and accents tie.
        ({"cat", "cat "}, "cat"),
    ],
)
def test_representative_prefers_well_formed_variant(variants, expected):
    assert min(variants, key=_representative_sort_key) == expected


def test_representative_is_order_independent():
    """The rule must not depend on set iteration order, which varies with
    PYTHONHASHSEED between processes."""
    variants = ["MÉXICO", "México", "méxico"]
    results = {
        min(set(perm), key=_representative_sort_key)
        for perm in (variants, list(reversed(variants)), variants[1:] + variants[:1])
    }
    assert results == {"México"}


# --------------------------------------------------------------------------
# Language validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("language", ["en", "es"])
def test_validate_language_accepts_supported(language):
    assert validate_language(language) == language


@pytest.mark.parametrize("language", ["fr", "pt", "", "EN"])
def test_validate_language_rejects_unsupported(language):
    with pytest.raises(ValueError, match="Unsupported language"):
        validate_language(language)
