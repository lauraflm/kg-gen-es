"""Offline regression tests for the robustness bugs fixed before freezing v1.1.

Every bug here is in code inherited from upstream kg-gen, and every fix applies
to all languages alike. None of them changes which triples are extracted when
the run goes well; they only change what happens when something goes wrong
(an empty list, a failed LLM call, a non-UTF-8 default encoding, ...). Nothing
here calls an LLM or touches the network.
"""

import builtins
import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.kg_gen import kg_gen as kg_gen_module
from src.kg_gen.models import Graph
from src.kg_gen.steps import _1_get_entities as entities_step
from src.kg_gen.steps import _2_get_relations as relations_step
from src.kg_gen.steps._3_deduplicate import DeduplicateMethod, run_deduplication
from src.kg_gen.utils import llm_deduplicate as llm_dedup_module
from src.kg_gen.utils.deduplicate import DeduplicateList
from src.kg_gen.utils.llm_deduplicate import LLMDeduplicate
from src.kg_gen.utils.visualize_kg import HTML_TEMPLATE, visualize

from tests.test_llm_deduplicate import FakeEncoder, ScriptedPredict, build

# --------------------------------------------------------------------------
# Empty inputs
# --------------------------------------------------------------------------


def test_semhash_list_deduplicate_accepts_an_empty_list():
    """SemHash raised IndexError on `records[0]` when given no records."""
    dedup = DeduplicateList(0.95, language="en", item_type="edge")

    assert dedup.deduplicate([]) is None

    assert dedup.deduplicated == []
    assert dedup.items_map == {}
    assert dedup.total_items == 0
    assert dedup.reduction == 0.0


@pytest.mark.parametrize(
    "graph",
    [
        Graph(entities=set(), edges=set(), relations=set()),
        Graph(entities={"Ana"}, edges=set(), relations=set()),
    ],
    ids=["empty", "entities-without-edges"],
)
def test_llm_dedup_accepts_a_graph_with_an_empty_component(monkeypatch, graph):
    """BM25Okapi and KMeans both fail on zero samples."""
    deduper, _ = build(graph, {}, monkeypatch=monkeypatch)

    deduper.cluster()
    result = deduper.deduplicate()

    assert result.entities == graph.entities
    assert result.edges == graph.edges
    assert result.relations == graph.relations


