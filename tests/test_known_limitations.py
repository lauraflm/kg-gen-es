"""Pins Spanish predicate dedup behavior that still loses meaning.

Every test here asserts the current behavior on purpose, wrong as it is. A
failure means that behavior changed: update the README ("Known limitations of
predicate dedup") along with the test. What is fixed, factuality and
polarity, is tested in tests/test_dedup_factuality.py.

No LLM and no network. The prefusion tests replace SemHash with the stub from
tests/test_deduplicate.py, so any merge they see was made by the dedup key.
The SemHash tests load the real encoders from a local path, and are skipped
when a model is not already in the Hugging Face cache.
"""

import pytest

from src.kg_gen.utils.deduplicate import DeduplicateList
from src.kg_gen.utils.text_normalization import TextNormalizer

BASE_ENCODER = "minishlab/potion-base-8M"
MULTILINGUAL_ENCODER = "minishlab/potion-multilingual-128M"


def _displays(dedup, items):
    """The label each original item ends up with in the graph."""
    return {item: dedup.items_map[dedup.original_map[item]] for item in items}


def _semhash(items, encoder, **kwargs):
    dedup = DeduplicateList(
        language="es", semhash_model=encoder, item_type="edge", **kwargs
    )
    dedup.deduplicate(items)
    return dedup


@pytest.fixture(scope="module")
def normalizer_es():
    return TextNormalizer(language="es")


# --------------------------------------------------------------------------
# Voice: kept by the key, merged by SemHash
# --------------------------------------------------------------------------


def test_key_keeps_voice(normalizer_es):
    """The key loses it: when these merge, SemHash merged them."""
    assert normalizer_es.lemmatize_predicate("fue fundado por") == "ser fundar por"


@pytest.mark.parametrize(
    "encoder_name, labels",
    [
        (BASE_ENCODER, {"fundar", "ser fundado por"}),
        # Relations keep their endpoints, so a passive triple merged into the
        # active verb states the opposite direction.
        (MULTILINGUAL_ENCODER, {"fundar"}),
    ],
)
def test_semhash_merges_voice_with_the_multilingual_encoder(
    semhash_encoders, encoder_name, labels
):
    items = ["fundar", "ser fundado por"]
    dedup = _semhash(items, semhash_encoders(encoder_name))

    assert set(_displays(dedup, items).values()) == labels


# --------------------------------------------------------------------------
# Negation, when keep_negation_apart=False: upstream's SemHash merges
# --------------------------------------------------------------------------


@pytest.mark.parametrize("encoder_name", [BASE_ENCODER, MULTILINGUAL_ENCODER])
@pytest.mark.parametrize(
    "affirmative, negated", [("nacer en", "no nacer en"), ("causar", "no causar")]
)
def test_without_the_guard_semhash_merges_negation_into_the_affirmative(
    semhash_encoders, encoder_name, affirmative, negated
):
    dedup = _semhash(
        [affirmative, negated],
        semhash_encoders(encoder_name),
        keep_negation_apart=False,
    )

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
    ],
)
def test_without_the_guard_negation_merge_depends_on_the_encoder(
    semhash_encoders, items, encoder_name, labels
):
    dedup = _semhash(items, semhash_encoders(encoder_name), keep_negation_apart=False)

    assert set(_displays(dedup, items).values()) == labels
