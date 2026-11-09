"""SemHash merges that chain through a record SemHash also discarded.

SemHash's `self_deduplicate` discards a record when any of its neighbours was
already seen next to a kept record, and names as its duplicate the first
neighbour in index order, which may be a record it discards too. The MINE
run of 7 Nov 2026 hit it twice:

- EN e0045, entities: "ancient Egyptian" -> "ancient Egyptian hieroglyph" ->
  "Egyptian hieroglyph" (kept).
- ES e0033, predicates: "fue utilizada por" -> "es utilizado en" ->
  "se utilizaban en" (kept).

The map back to the originals used to resolve one step, in the order SemHash
lists the discarded records. When the first link came first, its relations
kept the middle name, which no longer was an entity (or an edge), and the
LLM stage of `full` passed it through untouched.

A merge now follows the chain to the record SemHash kept, whatever the
order. A record whose chain reaches no kept record stays on its own. No LLM
and no network: SemHash is replaced by stubs, except in the last test, which
uses the cached multilingual encoder and skips without it.
"""

from types import SimpleNamespace

import pytest

from src.kg_gen.models import Graph
from src.kg_gen.utils import deduplicate as dedup_module
from src.kg_gen.utils.deduplicate import DeduplicateList

MULTILINGUAL_ENCODER = "minishlab/potion-multilingual-128M"


def _stub(plan):
    """A SemHash stand-in. `plan(records)` gets the records in the order
    DeduplicateList passes them (sorted) and returns (selected, targets),
    where `targets` maps each discarded record to its listed duplicate, or
    to None for an empty duplicate list. Discarded records are reported in
    index order, as SemHash does."""

    class PlannedSemHash:
        def __init__(self, records):
            self.records = records

        @classmethod
        def from_records(cls, records, **_):
            return cls(records)

        def self_deduplicate(self, threshold):
            selected, targets = plan(list(self.records))
            filtered = [
                SimpleNamespace(
                    record=r,
                    duplicates=[] if targets[r] is None else [(targets[r], 0.96)],
                )
                for r in self.records
                if r in targets
            ]
            return SimpleNamespace(selected=selected, filtered=filtered)

    return PlannedSemHash


def forward_chain(records):
    """r0 -> r1 -> ... -> r_last (kept): every link is listed before the
    record it points to, the order that used to break."""
    return [records[-1]], {r: records[i + 1] for i, r in enumerate(records[:-1])}


def backward_chain(records):
    """r_last -> ... -> r1 -> r0 (kept): every link is listed after the
    record it points to, the order that already worked."""
    return [records[0]], {r: records[i - 1] for i, r in enumerate(records) if i}


def dangling_target(records):
    """r0 -> r1, and r1 is discarded with no duplicate listed; r2 is kept."""
    r0, r1, r2 = records
    return [r2], {r0: r1, r1: None}


def cycle(records):
    """r0 -> r1 -> r0, neither kept; r2 is kept."""
    r0, r1, r2 = records
    return [r2], {r0: r1, r1: r0}


@pytest.fixture
def no_encoder(monkeypatch):
    monkeypatch.setattr(dedup_module, "_load_semhash_model", lambda name=None: None)


def _labels(dedup, items):
    return {item: dedup.items_map[dedup.original_map[item]] for item in items}


def _surviving_labels(dedup):
    return {dedup.items_map[x] for x in dedup.deduplicated}


def _assert_closed(graph):
    """Every relation names entities and an edge the graph has."""
    ends = {x for s, _, o in graph.relations for x in (s, o)}
    assert ends <= set(graph.entities)
    assert {p for _, p, _ in graph.relations} <= set(graph.edges)


# --------------------------------------------------------------------------
# Plain dedup (no feature guard): English entities, as in e0045
# --------------------------------------------------------------------------

E0045 = ["Egyptian hieroglyphs", "ancient Egyptian hieroglyphs", "ancient Egyptians"]


@pytest.mark.parametrize("plan", [forward_chain, backward_chain])
def test_chain_resolves_to_the_kept_record_in_either_order(monkeypatch, plan):
    monkeypatch.setattr(dedup_module, "SemHash", _stub(plan))
    dedup = DeduplicateList(language="en", item_type="entity")
    dedup.deduplicate(E0045)

    labels = _labels(dedup, E0045)
    assert len(set(labels.values())) == 1
    assert set(labels.values()) == _surviving_labels(dedup)


@pytest.mark.parametrize("plan", [dangling_target, cycle])
def test_chain_that_reaches_no_kept_record_leaves_its_records_alone(monkeypatch, plan):
    items = ["alpha", "beta", "gamma"]
    monkeypatch.setattr(dedup_module, "SemHash", _stub(plan))
    dedup = DeduplicateList(language="en", item_type="entity")
    dedup.deduplicate(items)

    assert _labels(dedup, items) == {item: item for item in items}
    assert _surviving_labels(dedup) == set(items)
    assert dedup.duplicate_items == 0


def test_e0045_graph_has_no_relation_outside_its_entities(monkeypatch, no_encoder):
    monkeypatch.setattr(dedup_module, "SemHash", _stub(forward_chain))
    graph = Graph(
        entities=set(E0045) | {"Champollion's work", "Rosetta Stone"},
        edges={"opened new era of understanding of", "contains inscription written in"},
        relations={
            (
                "Champollion's work",
                "opened new era of understanding of",
                "ancient Egyptians",
            ),
            (
                "Rosetta Stone",
                "contains inscription written in",
                "ancient Egyptian hieroglyphs",
            ),
            (
                "Rosetta Stone",
                "contains inscription written in",
                "Egyptian hieroglyphs",
            ),
        },
    )

    out = dedup_module.run_semhash_deduplication(graph, language="en")

    _assert_closed(out)


# --------------------------------------------------------------------------
# Dedup by node (factuality guard): Spanish predicates, as in e0033
# --------------------------------------------------------------------------

E0033 = ["fue utilizada por", "es utilizado en", "se utilizaban en"]


@pytest.mark.parametrize("plan", [forward_chain, backward_chain])
def test_predicate_chain_resolves_under_the_factuality_guard(monkeypatch, plan):
    monkeypatch.setattr(dedup_module, "SemHash", _stub(plan))
    dedup = DeduplicateList(language="es", item_type="edge", keep_factuality_apart=True)
    assert dedup._node_features() is not None  # the by-node path runs

    dedup.deduplicate(E0033)

    labels = _labels(dedup, E0033)
    assert len(set(labels.values())) == 1
    assert set(labels.values()) <= _surviving_labels(dedup)


def test_e0033_graph_has_no_relation_outside_its_edges(monkeypatch, no_encoder):
    monkeypatch.setattr(dedup_module, "SemHash", _stub(forward_chain))
    graph = Graph(
        entities={
            "escritura cuneiforme",
            "sumerios",
            "alfabeto latino",
            "inglés",
            "jeroglíficos",
            "Egipto",
        },
        edges=set(E0033),
        relations={
            ("escritura cuneiforme", "fue utilizada por", "sumerios"),
            ("alfabeto latino", "es utilizado en", "inglés"),
            ("jeroglíficos", "se utilizaban en", "Egipto"),
        },
    )

    out = dedup_module.run_semhash_deduplication(graph, language="es")

    _assert_closed(out)


def test_by_node_chain_does_not_join_nodes_that_disagree(monkeypatch):
    """Agreement lets an unmarked feature match anything, so it is not
    transitive: a feminine node may join an unmarked one that joins a
    masculine one. Following the chain must not put the feminine node on the
    masculine one; it stays on its own, as when it matches no node."""
    items = ["alpha", "beta", "gamma"]
    features = {"alpha": ("f", "sg"), "beta": (None, "sg"), "gamma": ("m", "sg")}
    monkeypatch.setattr(dedup_module, "SemHash", _stub(forward_chain))
    dedup = DeduplicateList(language="en", item_type="entity")
    monkeypatch.setattr(dedup, "_node_features", lambda: features.get)

    dedup.deduplicate(items)

    assert _labels(dedup, items) == {
        "alpha": "alpha",
        "beta": "gamma",
        "gamma": "gamma",
    }
    assert _surviving_labels(dedup) == {"alpha", "gamma"}


# --------------------------------------------------------------------------
# The real encoder, on e0045's names
# --------------------------------------------------------------------------


def test_e0045_with_the_multilingual_encoder(semhash_encoders):
    dedup = DeduplicateList(
        language="en",
        item_type="entity",
        semhash_model=semhash_encoders(MULTILINGUAL_ENCODER),
    )
    dedup.deduplicate(E0045)

    # The precondition: SemHash really lists a discarded record as the
    # duplicate of another. Without it this test proves nothing.
    assert any(dedup.duplicates.get(r) in dedup.duplicates for r in dedup.duplicates)
    labels = _labels(dedup, E0045)
    assert set(labels.values()) == {"Egyptian hieroglyphs"}
    assert _surviving_labels(dedup) == {"Egyptian hieroglyphs"}
