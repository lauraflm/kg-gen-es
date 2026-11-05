"""Predicate dedup keeps facts apart from plans and possibilities, and
affirmations apart from their negation.

Factuality applies where the key lemmatizes verbs (every language but
English), so "fundó" and "fundará" no longer share one edge. Polarity applies
in every language, and `keep_negation_apart=False` turns it off; with it off,
tests/test_known_limitations.py pins what upstream's SemHash does. When
nothing is irrealis or negated, the result is the plain dedup's.

No LLM and no network. SemHash is replaced by stubs, or loaded from the local
Hugging Face cache and skipped when a model is not there.
"""

from types import SimpleNamespace

import pytest

from src.kg_gen import kg_gen as kg_gen_module
from src.kg_gen.models import Graph
from src.kg_gen.steps import _3_deduplicate as dedup_step
from src.kg_gen.steps._3_deduplicate import DeduplicateMethod, run_deduplication
from src.kg_gen.utils import deduplicate as dedup_module
from src.kg_gen.utils.deduplicate import DeduplicateList
from src.kg_gen.utils.text_normalization import TextNormalizer

from test_deduplicate import KeepAllSemHash

BASE_ENCODER = "minishlab/potion-base-8M"
MULTILINGUAL_ENCODER = "minishlab/potion-multilingual-128M"


class MergeAllSemHash:
    """Stands in for SemHash and calls every record a duplicate of the first,
    so any two items that stay apart were kept apart by the guard."""

    def __init__(self, records):
        self.records = records

    @classmethod
    def from_records(cls, records, **_):
        return cls(records)

    def self_deduplicate(self, threshold):
        first, *rest = self.records
        return SimpleNamespace(
            selected=[first],
            filtered=[
                SimpleNamespace(record=r, duplicates=[(first, 1.0)]) for r in rest
            ],
        )


def _labels(dedup, items):
    """The label each original item ends up with in the graph."""
    return {item: dedup.items_map[dedup.original_map[item]] for item in items}


def _dedup(items, language="es", semhash_model=None, **kwargs):
    dedup = DeduplicateList(
        language=language, semhash_model=semhash_model, item_type="edge", **kwargs
    )
    dedup.deduplicate(items)
    return dedup


@pytest.fixture(scope="module")
def normalizer_es():
    return TextNormalizer(language="es")


@pytest.fixture(scope="module")
def normalizer_en():
    return TextNormalizer(language="en")


# --------------------------------------------------------------------------
# What the normalizer reads
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "predicate", ["fundó", "fundaron", "funda", "fundará", "fundaría", "fundar"]
)
def test_key_still_drops_tense_and_mood(normalizer_es, predicate):
    """The key is unchanged; factuality is read beside it."""
    assert normalizer_es.lemmatize_predicate(predicate) == "fundar"


@pytest.mark.parametrize(
    "predicate",
    [
        "fundó",
        "fundaba",
        "funda",
        "ha fundado",
        "fue fundado por",
        "fundar",
        "puede afectar",
        "va a fundar",
        "es parte de",
        "Fundó",
    ],
)
def test_facts_are_realis(normalizer_es, predicate):
    assert normalizer_es.factuality(predicate) == "realis"


@pytest.mark.parametrize(
    "predicate",
    [
        "fundará",
        "fundaría",
        "será fundado por",
        "habría fundado",
        "podría afectar",
        "sería parte de",
        "fundase",
        "hubiera fundado",
        "Fundará",
    ],
)
def test_futures_conditionals_and_subjunctives_are_irrealis(normalizer_es, predicate):
    assert normalizer_es.factuality(predicate) == "irrealis"


def test_english_has_no_factuality(normalizer_en):
    """English keys keep the verb as written, so they already differ."""
    assert normalizer_en.factuality("will found") is None


@pytest.mark.parametrize(
    "language, predicate, polarity",
    [
        ("es", "no nacer en", "negative"),
        ("es", "nunca fundó", "negative"),
        ("es", "jamás visitó", "negative"),
        ("es", "No causar", "negative"),
        ("es", "nacer en", "positive"),
        ("es", "nombrar", "positive"),
        ("en", "was not born in", "negative"),
        ("en", "didn't cause", "negative"),
        ("en", "never founded", "negative"),
        ("en", "cannot attend", "negative"),
        ("en", "born in", "positive"),
        ("en", "is notable for", "positive"),
    ],
)
def test_polarity(normalizer_es, normalizer_en, language, predicate, polarity):
    normalizer = normalizer_es if language == "es" else normalizer_en
    assert normalizer.polarity(predicate) == polarity


# --------------------------------------------------------------------------
# Factuality before SemHash: one key, separate edges
# --------------------------------------------------------------------------


def test_default_labels_keep_plans_apart_from_facts(monkeypatch):
    """The default prompts conjugate. Past forms still merge with each other,
    and so do the irrealis ones."""
    monkeypatch.setattr(dedup_module, "SemHash", KeepAllSemHash)
    items = ["fundó", "fundaron", "fundará", "fundaría"]

    assert _labels(_dedup(items), items) == {
        "fundó": "fundó",
        "fundaron": "fundó",
        "fundará": "fundará",
        "fundaría": "fundará",
    }


def test_es_v2_labels_keep_facts_apart_from_plans(monkeypatch):
    """es-v2 asks for infinitives but keeps futures and conditionals. The
    infinitive used to lose the label to the accented future."""
    monkeypatch.setattr(dedup_module, "SemHash", KeepAllSemHash)
    items = ["fundar", "fundará", "fundaría"]

    assert _labels(_dedup(items, linguistic_profile="es-v2"), items) == {
        "fundar": "fundar",
        "fundará": "fundará",
        "fundaría": "fundará",
    }


def test_factuality_counts_what_merged(monkeypatch):
    monkeypatch.setattr(dedup_module, "SemHash", KeepAllSemHash)
    dedup = _dedup(["fundó", "fundaron", "fundará"])

    assert (dedup.total_items, dedup.deduplicated_items, dedup.duplicate_items) == (
        3,
        2,
        1,
    )


def test_factuality_is_kept_even_without_the_negation_guard(monkeypatch):
    monkeypatch.setattr(dedup_module, "SemHash", KeepAllSemHash)
    items = ["fundó", "fundará"]

    labels = _labels(_dedup(items, keep_negation_apart=False), items)

    assert labels == {"fundó": "fundó", "fundará": "fundará"}


# --------------------------------------------------------------------------
# The guard after SemHash
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "language, items",
    [
        ("es", ["nacer en", "no nacer en", "nacerá en"]),
        ("en", ["born in", "not born in"]),
    ],
)
def test_a_semhash_merge_never_joins_different_features(monkeypatch, language, items):
    monkeypatch.setattr(dedup_module, "SemHash", MergeAllSemHash)

    labels = _labels(_dedup(items, language=language), items)

    assert labels == {item: item for item in items}


def test_a_semhash_merge_still_joins_what_agrees(monkeypatch):
    monkeypatch.setattr(dedup_module, "SemHash", MergeAllSemHash)
    items = ["nacer en", "nació en", "llegar a", "no nacer en", "no llegar a"]

    labels = _labels(_dedup(items), items)

    assert labels == {
        "nacer en": "llegar a",
        "nació en": "llegar a",
        "llegar a": "llegar a",
        "no llegar a": "no llegar a",
        # SemHash matched it only against the record it kept, "llegar a", so
        # it never meets "no llegar a" and stays on its own.
        "no nacer en": "no nacer en",
    }


@pytest.mark.parametrize("encoder_name", [BASE_ENCODER, MULTILINGUAL_ENCODER])
@pytest.mark.parametrize(
    "items",
    [
        ["nacer en", "no nacer en"],
        ["causar", "no causar"],
        ["fundar", "no fundar"],
        ["ser parte de", "no ser parte de"],
        ["born in", "not born in"],
        ["cause", "not cause"],
    ],
)
def test_real_semhash_keeps_negation_apart(semhash_encoders, encoder_name, items):
    language = "en" if items[0] in ("born in", "cause") else "es"
    dedup = _dedup(items, language=language, semhash_model=semhash_encoders(encoder_name))

    assert _labels(dedup, items) == {item: item for item in items}


# --------------------------------------------------------------------------
# Nothing irrealis or negated: the plain dedup
# --------------------------------------------------------------------------

_SPANISH_FACTS = [
    "nacer en",
    "nació en",
    "nacieron en",
    "ser parte de",
    "es parte de",
    "formar parte de",
    "fundar",
    "fundó",
    "Fundó",
    "pertenecer a",
    "dirigir",
]
_ENGLISH_FACTS = ["born in", "was born in", "part of", "is part of", "founded", "Founded"]


@pytest.mark.parametrize("encoder_name", [BASE_ENCODER, MULTILINGUAL_ENCODER])
@pytest.mark.parametrize(
    "language, items", [("es", _SPANISH_FACTS), ("en", _ENGLISH_FACTS)]
)
def test_without_irrealis_or_negation_the_result_is_the_plain_dedup(
    semhash_encoders, monkeypatch, encoder_name, language, items
):
    encoder = semhash_encoders(encoder_name)
    guarded = _dedup(items, language=language, semhash_model=encoder)

    plain = DeduplicateList(
        language=language, semhash_model=encoder, item_type="edge"
    )
    monkeypatch.setattr(plain, "_node_features", lambda: None)
    plain.deduplicate(items)

    assert _labels(guarded, items) == _labels(plain, items)
    assert guarded.deduplicated_items == plain.deduplicated_items


# --------------------------------------------------------------------------
# Graph level
# --------------------------------------------------------------------------


def test_graph_keeps_a_plan_and_a_denial_on_their_own_edges(monkeypatch):
    monkeypatch.setattr(dedup_module, "SemHash", MergeAllSemHash)
    graph = Graph(
        entities={"Ana", "Pedro", "la escuela", "un colegio"},
        edges={"fundó", "fundará", "no fundó"},
        relations={
            ("Ana", "fundó", "la escuela"),
            ("Pedro", "fundará", "un colegio"),
            ("Pedro", "no fundó", "la escuela"),
        },
    )

    out = dedup_module.run_semhash_deduplication(graph, language="es")

    assert out.edges == {"fundó", "fundará", "no fundó"}
    assert {r[1] for r in out.relations} == {"fundó", "fundará", "no fundó"}


def test_graph_merges_negation_when_the_guard_is_off(monkeypatch):
    monkeypatch.setattr(dedup_module, "SemHash", MergeAllSemHash)
    graph = Graph(
        entities={"Ana", "Lima"},
        edges={"nació en", "no nació en"},
        relations={("Ana", "nació en", "Lima"), ("Ana", "no nació en", "Lima")},
    )

    out = dedup_module.run_semhash_deduplication(
        graph, language="es", keep_negation_apart=False
    )

    assert len(out.edges) == 1


# --------------------------------------------------------------------------
# The flag reaches deduplication
# --------------------------------------------------------------------------


@pytest.fixture
def recorded_flags(monkeypatch):
    seen = []

    def fake_semhash(graph, threshold, **kwargs):
        seen.append(kwargs["keep_negation_apart"])
        return graph

    monkeypatch.setattr(dedup_step, "run_semhash_deduplication", fake_semhash)
    return seen


def test_run_deduplication_guards_negation_by_default(recorded_flags):
    graph = Graph(entities={"Ana"}, edges=set(), relations=set())

    run_deduplication(None, graph, method=DeduplicateMethod.SEMHASH, language="es")

    assert recorded_flags == [True]


@pytest.fixture
def recorded_kggen_flags(monkeypatch):
    """Patched on KGGen's module: it imports the package as `kg_gen.*`, a
    separate copy from the `src.kg_gen.*` modules imported above."""
    seen = []

    def fake_run_deduplication(**kwargs):
        seen.append(kwargs["keep_negation_apart"])
        return kwargs["graph"]

    monkeypatch.setattr(kg_gen_module, "run_deduplication", fake_run_deduplication)
    return seen


def _offline_kggen(**kwargs):
    # A fake model name and key: constructing dspy.LM only stores them.
    return kg_gen_module.KGGen(model="openai/fake-model", api_key="test-key", **kwargs)


def _graph():
    return Graph(entities={"Ana"}, edges=set(), relations=set())


@pytest.mark.parametrize(
    "instance, call, expected",
    [(True, None, True), (False, None, False), (True, False, False), (False, True, True)],
)
def test_kggen_deduplicate_resolves_the_flag(
    recorded_kggen_flags, instance, call, expected
):
    kg = _offline_kggen(language="es", keep_negation_apart=instance)

    kg.deduplicate(_graph(), method="semhash", keep_negation_apart=call)

    assert recorded_kggen_flags == [expected]


def test_kggen_generate_passes_its_flag(monkeypatch, recorded_kggen_flags):
    monkeypatch.setattr(kg_gen_module, "get_entities", lambda *a, **k: ["Ana"])
    monkeypatch.setattr(kg_gen_module, "get_relations", lambda *a, **k: [])
    kg = _offline_kggen(language="es")

    kg.generate("Ana.", deduplication_method="semhash", keep_negation_apart=False)

    assert recorded_kggen_flags == [False]
