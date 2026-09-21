"""Offline tests for ExtractionStats and the counters wired into relation parsing.

No LLM calls: the dspy route is exercised with `dspy.Predict` patched out.
"""

import json
import threading
from types import SimpleNamespace

import pytest

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
