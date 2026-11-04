"""Pins Spanish predicate dedup behavior that loses meaning and is kept in v1.1.

Every test here asserts the current behavior on purpose, wrong as it is. The
fix changes every Spanish graph since `v1-piloto`, so it is left out of v1.1
(README, "Frozen at v1.1"). A failure means that behavior changed: update the
README along with the test.

No LLM and no network. The prefusion tests replace SemHash with the stub from
tests/test_deduplicate.py, so any merge they see was made by the dedup key.
The SemHash tests load the real encoders from a local path, and are skipped
when a model is not already in the Hugging Face cache.
"""

import pytest

from src.kg_gen.utils import deduplicate as dedup_module
from src.kg_gen.utils.deduplicate import DeduplicateList
from src.kg_gen.utils.text_normalization import TextNormalizer

from test_deduplicate import KeepAllSemHash

BASE_ENCODER = "minishlab/potion-base-8M"
MULTILINGUAL_ENCODER = "minishlab/potion-multilingual-128M"


def _displays(dedup, items):
    """The label each original item ends up with in the graph."""
    return {item: dedup.items_map[dedup.original_map[item]] for item in items}


def _prefuse(items, monkeypatch):
    monkeypatch.setattr(dedup_module, "SemHash", KeepAllSemHash)
    dedup = DeduplicateList(language="es", item_type="edge")
    dedup.deduplicate(items)
    return dedup


def _semhash(items, encoder):
    dedup = DeduplicateList(language="es", semhash_model=encoder, item_type="edge")
    dedup.deduplicate(items)
    return dedup


@pytest.fixture(scope="module")
def normalizer_es():
    return TextNormalizer(language="es")


# --------------------------------------------------------------------------
# Tense and mood: dropped by the dedup key
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "predicate", ["fundó", "fundaron", "funda", "fundará", "fundaría", "fundar"]
)
def test_key_drops_tense_and_mood(normalizer_es, predicate):
    assert normalizer_es.lemmatize_predicate(predicate) == "fundar"


@pytest.mark.parametrize(
    "predicate, key",
    [("no fundó", "no fundar"), ("fue fundado por", "ser fundar por")],
)
def test_key_keeps_negation_and_voice(normalizer_es, predicate, key):
    """The key loses neither: when these merge, SemHash merged them."""
    assert normalizer_es.lemmatize_predicate(predicate) == key


def test_default_labels_turn_plans_into_facts(monkeypatch):
    """The default prompts conjugate, and the past tense wins the label."""
    items = ["fundó", "fundará", "fundaría"]
    dedup = _prefuse(items, monkeypatch)

    assert len(dedup.deduplicated) == 1
    assert set(_displays(dedup, items).values()) == {"fundó"}


@pytest.mark.parametrize(
    "items, label",
    [
        (["fundar", "fundará"], "fundará"),
        (["fundar", "fundaría"], "fundaría"),
        (["fundar", "fundará", "fundaría"], "fundará"),
    ],
)
def test_es_v2_labels_turn_facts_into_plans(monkeypatch, items, label):
    """es-v2 asks for infinitives but keeps futures and conditionals. Those
    still share the infinitive's key, and the label goes to the form with
    more accents, which the infinitive never is."""
    dedup = _prefuse(items, monkeypatch)

    assert len(dedup.deduplicated) == 1
    assert set(_displays(dedup, items).values()) == {label}


# --------------------------------------------------------------------------
# Negation and voice: kept by the key, merged by SemHash
# --------------------------------------------------------------------------


@pytest.mark.parametrize("encoder_name", [BASE_ENCODER, MULTILINGUAL_ENCODER])
@pytest.mark.parametrize(
    "affirmative, negated", [("nacer en", "no nacer en"), ("causar", "no causar")]
)
def test_semhash_merges_negation_into_the_affirmative(
    semhash_encoders, encoder_name, affirmative, negated
):
    dedup = _semhash([affirmative, negated], semhash_encoders(encoder_name))

    assert _displays(dedup, [affirmative, negated]) == {
        affirmative: affirmative,
        negated: affirmative,
    }


@pytest.mark.parametrize(
    "items, encoder_name, labels",
    [
        # The negated form sorts first, so it labels the merged edge.
        (["ser parte de", "no ser parte de"], BASE_ENCODER, {"no ser parte de"}),
        (
            ["ser parte de", "no ser parte de"],
            MULTILINGUAL_ENCODER,
            {"ser parte de", "no ser parte de"},
        ),
        # Close to the threshold: cosine 0.948 base, 0.990 multilingual.
        (["fundar", "no fundar"], BASE_ENCODER, {"fundar", "no fundar"}),
        (["fundar", "no fundar"], MULTILINGUAL_ENCODER, {"fundar"}),
        # Voice. Relations keep their endpoints, so a passive triple merged
        # into the active verb states the opposite direction.
        (["fundar", "ser fundado por"], BASE_ENCODER, {"fundar", "ser fundado por"}),
        (["fundar", "ser fundado por"], MULTILINGUAL_ENCODER, {"fundar"}),
    ],
)
def test_semhash_merge_depends_on_the_encoder(
    semhash_encoders, items, encoder_name, labels
):
    dedup = _semhash(items, semhash_encoders(encoder_name))

    assert set(_displays(dedup, items).values()) == labels
