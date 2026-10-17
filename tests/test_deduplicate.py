"""Offline tests for the grouping that DeduplicateList does before SemHash.

`TextNormalizer.fold()` decides which items collapse deterministically,
whatever the similarity threshold. SemHash is replaced by a stub that merges
nothing, so any merge observed here was made by `fold()` and not by an
encoder. No LLM, no network, no model download; the Spanish cases do load
spaCy's `es_core_news_md`, which has to be installed separately.
"""

from types import SimpleNamespace

import pytest

from src.kg_gen.utils import deduplicate as dedup_module
from src.kg_gen.utils.deduplicate import DeduplicateList
from src.kg_gen.utils.text_normalization import TextNormalizer


class KeepAllSemHash:
    """Stands in for SemHash and keeps every record it is given."""

    def __init__(self, records):
        self.records = records

    @classmethod
    def from_records(cls, records, **_):
        return cls(records)

    def self_deduplicate(self, threshold):
        return SimpleNamespace(selected=list(self.records), filtered=[])


@pytest.fixture(scope="module")
def normalizer_es():
    return TextNormalizer(language="es")


# --------------------------------------------------------------------------
# fold() itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a, b",
    [
        # Different Spanish words that differ only by a diacritic.
        ("año", "ano"),
        ("papá", "papa"),
        ("té", "te"),
        ("él", "el"),
        ("sí", "si"),
        ("más", "mas"),
        ("público", "publico"),
        ("Peña", "Pena"),
        # The same mark, in the same position, also separates spellings of
        # one entity. No character-level rule tells Perú/Peru apart from
        # papá/papa, so accent omission is left to the similarity stages.
        ("México", "Mexico"),
        ("Perú", "Peru"),
    ],
)
def test_fold_keeps_diacritics(normalizer_es, a, b):
    assert normalizer_es.fold(a) != normalizer_es.fold(b)


@pytest.mark.parametrize(
    "language, variants",
    [
        ("es", ["México", "MÉXICO", "méxico"]),
        ("es", ["Año", "AÑO", "año"]),
        ("es", ["Gabriel García Márquez", "GABRIEL GARCÍA MÁRQUEZ"]),
        ("en", ["Paris", "PARIS", "paris"]),
        ("en", ["United States", "united states", "UNITED STATES"]),
    ],
)
def test_fold_merges_case_variants(language, variants):
    normalizer = TextNormalizer(language=language)
    assert len({normalizer.fold(v) for v in variants}) == 1


def test_fold_ignores_composed_vs_decomposed_spelling(normalizer_es):
    # "á" can arrive as one codepoint or as "a" + combining acute; both are
    # the same text and must keep folding together now that marks are kept.
    composed = "Bogotá"
    decomposed = "Bogotá"
    assert normalizer_es.fold(composed) == normalizer_es.fold(decomposed)
    assert normalizer_es.fold("BOGOTÁ") == normalizer_es.fold(composed)


# --------------------------------------------------------------------------
# DeduplicateList grouping before SemHash
# --------------------------------------------------------------------------


def _deduplicate(items, language, monkeypatch):
    monkeypatch.setattr(dedup_module, "SemHash", KeepAllSemHash)
    dedup = DeduplicateList(language=language)
    dedup.deduplicate(items)
    return dedup


def _display(dedup, item):
    return dedup.items_map[dedup.original_map[item]]


def test_spanish_words_differing_by_diacritic_are_not_merged(monkeypatch):
    items = ["año", "ano", "papá", "papa", "México", "MÉXICO"]
    dedup = _deduplicate(items, "es", monkeypatch)

    assert sorted(dedup.deduplicated) == sorted(["año", "ano", "papá", "papa", "México"])
    for word in ("año", "ano", "papá", "papa"):
        assert _display(dedup, word) == word
    # Case variants still collapse, onto the well-formed form.
    assert _display(dedup, "MÉXICO") == "México"
    assert _display(dedup, "México") == "México"
    assert dedup.duplicate_items == 1


def test_english_case_variants_still_merge(monkeypatch):
    dedup = _deduplicate(["Paris", "PARIS", "London"], "en", monkeypatch)

    assert len(dedup.deduplicated) == 2
    assert _display(dedup, "PARIS") == "Paris"
    assert _display(dedup, "Paris") == "Paris"
