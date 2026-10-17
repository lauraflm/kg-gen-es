"""Offline tests for ExtractionStats and the counters wired into relation parsing.

No LLM calls: the dspy route is exercised with `dspy.Predict` patched out,
and `KGGen.generate` with its step functions patched out.
"""

import json
import socket
import threading
from types import SimpleNamespace

import pytest

from src.kg_gen import kg_gen as kg_gen_module
from src.kg_gen.steps import _2_get_relations as relations_step
from src.kg_gen.steps._2_get_relations import parse_relations_response
from src.kg_gen.utils.extraction_stats import ExtractionStats

ENTITIES = ["Ana", "Madrid"]


def _response(*items) -> str:
    return json.dumps({"relations": list(items)})


def _rel(subject, predicate, obj) -> dict:
    return {"subject": subject, "predicate": predicate, "object": obj}


# --------------------------------------------------------------------------
# ExtractionStats itself
# --------------------------------------------------------------------------


def test_snapshot_and_summary_do_not_raise():
    # Regression: snapshot() used dataclasses.asdict, which deep-copies the
    # lock and raised TypeError on every call.
    stats = ExtractionStats()
    stats.record_chunk("uno dos tres")
    snapshot = stats.snapshot()
    assert "_lock" not in snapshot
    assert snapshot["chunks"] == 1
    assert stats.summary()["rates"]["mean_chunk_words"] == 3


def test_snapshot_is_a_copy():
    stats = ExtractionStats()
    stats.record_chunk("a b")
    snapshot = stats.snapshot()
    stats.record_chunk("c d e")
    assert snapshot["chunk_chars"] == [3]


def test_write_produces_json(tmp_path):
    stats = ExtractionStats()
    stats.record_route("dspy")
    path = tmp_path / "stats.json"
    stats.write(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["route"] == "dspy"
    assert "rates" in data


def test_rates_are_none_when_empty():
    rates = ExtractionStats().summary()["rates"]
    assert all(value is None for value in rates.values())


def test_rates():
    stats = ExtractionStats()
    stats.record_chunk("uno dos")  # 7 chars, 2 words
    stats.record_entities(2)
    stats.record_relations(
        [("Ana", "vive en", "Madrid"), ("Ana", "visitó", "Lima")], ENTITIES
    )
    rates = stats.summary()["rates"]
    assert rates["endpoint_missing"] == 0.5
    assert rates["relations_per_entity"] == 1.0
    assert rates["chars_per_word"] == 3.5


def test_record_relations_accepts_objects_and_tuples():
    stats = ExtractionStats()
    stats.record_relations(
        [
            SimpleNamespace(subject="Lima", predicate="p", object="Madrid"),
            ("Ana", "p", "Quito"),
            ("Lima", "p", "Quito"),
        ],
        ENTITIES,
    )
    assert stats.relations_returned == 3
    assert stats.endpoint_missing_subject == 2
    assert stats.endpoint_missing_object == 2
    assert stats.endpoint_missing_either == 3


def test_concurrent_recording_loses_no_counts():
    stats = ExtractionStats()

    def work():
        for _ in range(500):
            stats.record_chunk("a b")
            stats.increment("fallback_invoked")

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert stats.chunks == 4000
    assert stats.fallback_invoked == 4000
    assert len(stats.chunk_chars) == 4000


def test_merge_sums_counters_extends_chunks_and_sets_route():
    total = ExtractionStats()
    total.record_chunk("uno dos")
    total.record_entities(2)
    total.increment("fallback_invoked")

    attempt = ExtractionStats()
    attempt.record_route("dspy")
    attempt.record_chunk("tres cuatro cinco")
    attempt.record_entities(3)
    attempt.record_relations([("Ana", "visitó", "Lima")], ENTITIES)
    attempt.increment("fallback_invoked")
    attempt.increment("fallback_dropped", 2)

    total.merge(attempt)

    assert total.chunks == 2
    assert total.chunk_chars == [7, 17]
    assert total.chunk_words == [2, 3]
    assert total.entities_extracted == 5
    assert total.relations_returned == 1
    assert total.endpoint_missing_object == 1
    assert total.endpoint_missing_either == 1
    assert total.fallback_invoked == 2
    assert total.fallback_dropped == 2
    assert total.route == "dspy"
    # The merged-in collector is read, not drained.
    assert attempt.chunks == 1


def test_merge_keeps_route_when_other_has_none():
    total = ExtractionStats()
    total.record_route("litellm")
    total.merge(ExtractionStats())
    assert total.route == "litellm"


def test_concurrent_merges_lose_no_counts():
    total = ExtractionStats()

    def work():
        for _ in range(200):
            attempt = ExtractionStats()
            attempt.record_chunk("a b")
            attempt.increment("fallback_invoked")
            total.merge(attempt)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert total.chunks == 1600
    assert total.fallback_invoked == 1600
    assert len(total.chunk_chars) == 1600


# --------------------------------------------------------------------------
# parse_relations_response (litellm route)
# --------------------------------------------------------------------------


def test_parse_counts_proposed_triples_before_filtering():
    stats = ExtractionStats()
    result = parse_relations_response(
        _response(
            _rel("Ana", "vive en", "Madrid"),
            _rel("Ana", "visitó", "Lima"),
            {"subject": "Ana"},
        ),
        ENTITIES,
        stats=stats,
    )
    assert result == [("Ana", "vive en", "Madrid")]
    assert stats.relations_returned == 2  # the incomplete item is not a triple
    assert stats.endpoint_missing_either == 1
    assert stats.items_missing_field == 1


def test_parse_invalid_json_counts_response():
    stats = ExtractionStats()
    assert parse_relations_response("no es json", ENTITIES, stats=stats) == []
    assert stats.responses_json_invalid == 1
    assert stats.relations_returned == 0


def test_parse_strict_model_success_and_failure():
    _, model = relations_step._create_relations_model(ENTITIES)

    stats = ExtractionStats()
    parse_relations_response(
        _response(_rel("Ana", "vive en", "Madrid")), ENTITIES, model, stats=stats
    )
    assert stats.strict_validation_failed == 0
    assert stats.relations_returned == 1

    parse_relations_response(
        _response(_rel("Ana", "visitó", "Lima")), ENTITIES, model, stats=stats
    )
    assert stats.strict_validation_failed == 1
    assert stats.relations_returned == 2
    assert stats.endpoint_missing_either == 1


@pytest.mark.parametrize(
    "raw",
    [
        _response(_rel("Ana", "vive en", "Madrid"), _rel("Ana", "visitó", "Lima")),
        "no es json",
        json.dumps([_rel("Ana", "vive en", "Madrid"), {"subject": "Ana"}]),
    ],
)
def test_parse_result_is_unchanged_by_stats(raw):
    assert parse_relations_response(raw, ENTITIES) == parse_relations_response(
        raw, ENTITIES, stats=ExtractionStats()
    )


# --------------------------------------------------------------------------
# get_relations dspy route, with dspy.Predict patched out
# --------------------------------------------------------------------------


class _FakePredict:
    def __init__(self, relations):
        self._relations = relations

    def __call__(self, signature):
        relations = self._relations

        def run(**_):
            return SimpleNamespace(relations=relations)

        return run


def test_dspy_route_keeps_dangling_triples_and_counts_them(monkeypatch):
    relations = [
        SimpleNamespace(subject="Ana", predicate="vive en", object="Madrid"),
        SimpleNamespace(subject="Ana", predicate="visitó", object="Lima"),
    ]
    monkeypatch.setattr(relations_step.dspy, "Predict", _FakePredict(relations))

    stats = ExtractionStats()
    result = relations_step.get_relations(
        "Ana vive en Madrid y visitó Lima.", ENTITIES, language="es", stats=stats
    )

    # The dspy route does not filter: the dangling triple survives.
    assert ("Ana", "visitó", "Lima") in result
    assert stats.route == "dspy"
    assert stats.relations_returned == 2
    assert stats.endpoint_missing_either == 1


class _ScriptedPredict:
    """Stands in for `dspy.Predict`, playing one scripted outcome per
    signature it is built with, in order: an exception to raise or the
    relations to return.
    """

    def __init__(self, *outcomes):
        self._outcomes = list(outcomes)

    def __call__(self, signature):
        outcome = self._outcomes.pop(0)

        def run(**_):
            if isinstance(outcome, Exception):
                raise outcome
            return SimpleNamespace(relations=outcome)

        return run


class _FakeChainOfThought:
    """Stands in for the `FixedRelations` step of the fallback."""

    def __init__(self, fixed_relations):
        self._fixed_relations = fixed_relations

    def __call__(self, signature):
        fixed_relations = self._fixed_relations

        def run(**_):
            return SimpleNamespace(fixed_relations=fixed_relations)

        return run


FALLBACK_TEXT = "Ana vive en Madrid y visitó Lima."

# What the fallback signature proposes, before the fix step. It differs from
# what the fix step returns on purpose, so the tests can tell which of the two
# lists the counters were taken from.
FALLBACK_PROPOSED = [
    SimpleNamespace(subject="Ana Pérez", predicate="vive en", object="Madrid"),
    SimpleNamespace(subject="Ana", predicate="visitó", object="Lima"),
    SimpleNamespace(subject="Ana", predicate="nació en", object="Quito"),
]
FALLBACK_FIXED = [
    SimpleNamespace(subject="Ana", predicate="vive en", object="Madrid"),
    SimpleNamespace(subject="Ana", predicate="visitó", object="Lima"),
]


def _patch_failing_primary(monkeypatch):
    # The primary Predict raises, so get_relations takes the except branch:
    # the fallback signature runs, then FixedRelations repairs its output.
    monkeypatch.setattr(
        relations_step.dspy,
        "Predict",
        _ScriptedPredict(RuntimeError("primary failed"), FALLBACK_PROPOSED),
    )
    monkeypatch.setattr(
        relations_step.dspy, "ChainOfThought", _FakeChainOfThought(FALLBACK_FIXED)
    )


def test_dspy_fallback_route_counts_its_triples(monkeypatch):
    _patch_failing_primary(monkeypatch)

    stats = ExtractionStats()
    result = relations_step.get_relations(
        FALLBACK_TEXT, ENTITIES, language="es", stats=stats
    )

    # Unlike the primary route, the fallback filters: the dangling triple is
    # dropped.
    assert result == [("Ana", "vive en", "Madrid")]
    assert stats.route == "dspy"
    assert stats.fallback_invoked == 1
    # Counted from the fix step's output before the endpoint filter, as the
    # primary route counts its own output before anything else touches it.
    assert stats.relations_returned == 2
    assert stats.endpoint_missing_object == 1
    assert stats.endpoint_missing_either == 1
    # Every fallback triple with a missing endpoint is the one that is dropped.
    assert stats.fallback_dropped == stats.endpoint_missing_either


def test_dspy_fallback_result_is_unchanged_by_stats(monkeypatch):
    _patch_failing_primary(monkeypatch)
    without_stats = relations_step.get_relations(FALLBACK_TEXT, ENTITIES, language="es")

    _patch_failing_primary(monkeypatch)
    with_stats = relations_step.get_relations(
        FALLBACK_TEXT, ENTITIES, language="es", stats=ExtractionStats()
    )

    assert with_stats == without_stats


# --------------------------------------------------------------------------
# KGGen.generate, with its step functions patched out
# --------------------------------------------------------------------------

FULL_TEXT = "Ana vive en Madrid. Luis trabaja en Lima."
CHUNKS = ["Ana vive en Madrid.", "Luis trabaja en Lima."]
CHUNK_ENTITIES = {
    FULL_TEXT: ["Ana", "Madrid", "Luis", "Lima"],
    CHUNKS[0]: ["Ana", "Madrid"],
    CHUNKS[1]: ["Luis", "Lima"],
}
CHUNK_RELATIONS = {
    CHUNKS[0]: [("Ana", "vive en", "Madrid")],
    CHUNKS[1]: [("Luis", "trabaja en", "Lima")],
}


@pytest.fixture
def no_network(monkeypatch):
    """Fail the test on any attempt to open a connection."""

    def refuse(*args, **kwargs):
        raise AssertionError("offline test tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture
def stubbed_steps(monkeypatch):
    """Patch the extraction steps that `KGGen.generate` calls.

    The unchunked call fails the way the dspy route does on a context-length
    error: the primary Predict raises, the except branch counts the fallback,
    and the fallback raises again on the same oversized input. Each chunk then
    succeeds; the second one through the fallback, so that a fallback that
    succeeded is still expected in the totals.

    Returns the `stats` argument of every get_relations call, in call order.
    """
    received_stats = []

    def fake_get_entities(content, *args, **kwargs):
        return list(CHUNK_ENTITIES[content])

    def fake_get_relations(content, entities, *args, stats=None, **kwargs):
        received_stats.append(stats)
        if stats is not None:
            stats.record_route("dspy")
        if content == FULL_TEXT:
            if stats is not None:
                stats.increment("fallback_invoked")
            raise RuntimeError("This model's maximum context length is 8192 tokens")
        if content == CHUNKS[1] and stats is not None:
            stats.increment("fallback_invoked")
        relations = CHUNK_RELATIONS[content]
        if stats is not None:
            stats.record_relations(relations, entities)
        return list(relations)

    def fake_chunk_text(text, max_chunk_size=500, language="en"):
        assert text == FULL_TEXT
        return list(CHUNKS)

    monkeypatch.setattr(kg_gen_module, "get_entities", fake_get_entities)
    monkeypatch.setattr(kg_gen_module, "get_relations", fake_get_relations)
    monkeypatch.setattr(kg_gen_module, "chunk_text", fake_chunk_text)
    return received_stats


def _offline_kggen():
    # A fake model name and key: constructing dspy.LM only stores them, and
    # the steps that would use them are patched out.
    return kg_gen_module.KGGen(model="openai/fake-model", api_key="test-key")


def test_generate_counts_only_successful_attempts(no_network, stubbed_steps):
    stats = ExtractionStats()
    graph = _offline_kggen().generate(
        FULL_TEXT, deduplication_method=None, language="es", stats=stats
    )

    # The failed unchunked attempt contributes nothing; the two chunks do.
    assert graph.relations == set(CHUNK_RELATIONS[CHUNKS[0]]) | set(
        CHUNK_RELATIONS[CHUNKS[1]]
    )
    assert stats.route == "dspy"
    assert stats.chunks == 2
    assert sorted(stats.chunk_chars) == sorted(len(chunk) for chunk in CHUNKS)
    assert stats.entities_extracted == 4
    assert stats.relations_returned == 2
    assert stats.endpoint_missing_either == 0
    # Only the chunk whose fallback succeeded; not the failed attempt's.
    assert stats.fallback_invoked == 1


def test_generate_graph_is_unchanged_by_stats(no_network, stubbed_steps):
    kg = _offline_kggen()
    without_stats = kg.generate(FULL_TEXT, deduplication_method=None, language="es")
    calls_without_stats = len(stubbed_steps)
    with_stats = kg.generate(
        FULL_TEXT, deduplication_method=None, language="es", stats=ExtractionStats()
    )

    assert with_stats == without_stats
    # Without a caller collector no step receives one either, so the
    # uninstrumented run does no counting at all.
    assert stubbed_steps[:calls_without_stats] == [None] * calls_without_stats
    assert all(stats is not None for stats in stubbed_steps[calls_without_stats:])
