"""Offline tests for the multilingual support added on top of kg-gen.

None of these call an LLM or load a spaCy model, so they take milliseconds.
Accepting "es" does check that `es_core_news_md` is installed, without
loading it, so that model has to be present like for the rest of the
Spanish pipeline (see "Requirements and fallback behavior" in the README).
"""

import socket

import pytest

from src.kg_gen import kg_gen as kg_gen_module
from src.kg_gen.steps._1_get_entities import _entities_sig, _load_entities_prompt
from src.kg_gen.steps._2_get_relations import (
    _load_relations_prompt,
    _relation_model,
    extraction_sig,
    fallback_extraction_sig,
)
from src.kg_gen.utils.deduplicate import _representative_sort_key
from src.kg_gen.utils.i18n import pick, validate_language

# The package's own modules import each other as `kg_gen.*`, which Python
# keeps as a module object separate from the `src.kg_gen.*` copy imported
# above. KGGen and validate_language read `_SPACY_MODELS` from this copy, so
# it is the one the tests patch.
from kg_gen.utils import text_normalization as package_text_normalization

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


def test_prompt_files_fall_back_to_english():
    """The litellm-path prompt files follow the same fallback as `pick`."""
    assert _load_entities_prompt("fr") == _load_entities_prompt("en")
    assert _load_relations_prompt("fr") == _load_relations_prompt("en")
    assert _load_entities_prompt("es") != _load_entities_prompt("en")
    assert _load_relations_prompt("es") != _load_relations_prompt("en")


# --------------------------------------------------------------------------
# Signature caching
# --------------------------------------------------------------------------


def test_signature_builders_reuse_one_class_per_arguments():
    """Built once per argument set, not once per chunk."""
    assert _entities_sig(False, "es") is _entities_sig(False, "es")
    assert _entities_sig(False, "es") is not _entities_sig(False, "en")
    assert _entities_sig(False, "es") is not _entities_sig(True, "es")

    relation_es = _relation_model("es")
    relation_en = _relation_model("en")
    assert relation_es is _relation_model("es")
    assert relation_es is not relation_en

    assert extraction_sig(relation_es, False, "", "es") is extraction_sig(
        relation_es, False, "", "es"
    )
    assert extraction_sig(relation_es, False, "", "es") is not extraction_sig(
        relation_en, False, "", "en"
    )
    assert extraction_sig(relation_es, False, "", "es") is not extraction_sig(
        relation_es, True, "", "es"
    )


def test_fallback_signature_is_built_per_entity_list_and_not_cached():
    """The fallback inlines each chunk's entity list into the Relation doc,
    so it has to be built fresh, and must not pile up in the caches."""
    relation_size = _relation_model.cache_info().currsize
    sig_size = extraction_sig.cache_info().currsize

    relation_a, sig_a = fallback_extraction_sig(["Ana", "Madrid"], False, "", "es")
    relation_b, sig_b = fallback_extraction_sig(["Luis", "Lima"], False, "", "es")

    assert "Madrid" in relation_a.__doc__ and "Lima" not in relation_a.__doc__
    assert "Lima" in relation_b.__doc__ and "Madrid" not in relation_b.__doc__
    assert sig_a is not sig_b
    assert _relation_model.cache_info().currsize == relation_size
    assert extraction_sig.cache_info().currsize == sig_size
    # The cached default model keeps its own doc.
    assert _relation_model("es").__doc__ == (
        "Tripleta sujeto-predicado-objeto de un grafo de conocimiento."
    )


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


MISSING_SPACY_MODEL = "kg_gen_test_model_not_installed"


@pytest.fixture
def no_network(monkeypatch):
    """Fail the test on any attempt to open a connection."""

    def refuse(*args, **kwargs):
        raise AssertionError("offline test tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture
def spacy_models_missing(monkeypatch):
    """Configure every language with a spaCy model that is not installed."""
    for language in list(package_text_normalization._SPACY_MODELS):
        monkeypatch.setitem(
            package_text_normalization._SPACY_MODELS, language, MISSING_SPACY_MODEL
        )


def _offline_kggen(**kwargs):
    # A fake model name and key: constructing dspy.LM only stores them.
    return kg_gen_module.KGGen(model="openai/fake-model", api_key="test-key", **kwargs)


def test_validate_language_rejects_missing_spacy_model(spacy_models_missing):
    # Regression: only the code was checked, so a missing model surfaced as
    # an OSError at deduplication, after every extraction call was paid for.
    with pytest.raises(
        OSError, match=f"python -m spacy download {MISSING_SPACY_MODEL}"
    ):
        validate_language("es")


def test_kggen_rejects_missing_spacy_model_at_construction(
    no_network, spacy_models_missing
):
    with pytest.raises(OSError, match="python -m spacy download"):
        _offline_kggen(language="es")


def test_generate_rejects_missing_spacy_model_before_extraction(
    no_network, spacy_models_missing, monkeypatch
):
    def extraction_must_not_run(*args, **kwargs):
        raise AssertionError("extraction ran before the language was validated")

    monkeypatch.setattr(kg_gen_module, "get_entities", extraction_must_not_run)
    monkeypatch.setattr(kg_gen_module, "get_relations", extraction_must_not_run)

    kg = _offline_kggen()
    with pytest.raises(OSError, match="python -m spacy download"):
        kg.generate("Ana vive en Madrid.", language="es")


def test_english_needs_no_spacy_model(no_network, spacy_models_missing):
    # English is singularized with inflect, so en_core_web_sm must not
    # become a requirement even though `_SPACY_MODELS` lists it.
    assert validate_language("en") == "en"
    assert _offline_kggen(language="en").language == "en"


def test_kggen_construction_is_offline(no_network, monkeypatch):
    # The model check reads installed-package metadata; it must not load
    # the model or reach the network.
    def load_must_not_run(*args, **kwargs):
        raise AssertionError("the spaCy model was loaded, not just looked up")

    monkeypatch.setattr(package_text_normalization.spacy, "load", load_must_not_run)

    assert _offline_kggen(language="es").language == "es"
    assert _offline_kggen(language="en").language == "en"


# --------------------------------------------------------------------------
# SemHash model loading under concurrency
# --------------------------------------------------------------------------


def test_default_semhash_model_loads_once_across_threads(monkeypatch):
    # Regression: leaving the default to SemHash reloaded it on every call,
    # and concurrent loads crashed inside tqdm with model2vec >= 0.8.
    import threading
    import time

    from src.kg_gen.utils import deduplicate as dedup_module

    loads = []

    class FakeStaticModel:
        @staticmethod
        def from_pretrained(name):
            loads.append(name)
            time.sleep(0.01)  # widen the race window
            return object()

    monkeypatch.setattr(dedup_module, "StaticModel", FakeStaticModel)
    monkeypatch.setattr(dedup_module, "_M2V_CACHE", {})

    results = []
    threads = [
        threading.Thread(target=lambda: results.append(dedup_module._load_semhash_model()))
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert loads == ["minishlab/potion-base-8M"]
    assert len({id(model) for model in results}) == 1


# --------------------------------------------------------------------------
# spaCy model loading under concurrency
# --------------------------------------------------------------------------


def test_spacy_model_loads_once_across_threads(monkeypatch):
    # Regression: the cache was filled without a lock, so graphs
    # deduplicated from several threads each loaded the same model.
    import threading
    import time

    from src.kg_gen.utils import text_normalization

    loads = []

    def fake_load(name, **kwargs):
        loads.append(name)
        time.sleep(0.01)  # widen the race window
        return object()

    monkeypatch.setattr(text_normalization.spacy, "load", fake_load)
    monkeypatch.setattr(text_normalization, "_NLP_CACHE", {})

    results = []
    threads = [
        threading.Thread(target=lambda: results.append(text_normalization._get_nlp("es")))
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert loads == [text_normalization._SPACY_MODELS["es"]]
    assert len({id(nlp) for nlp in results}) == 1
