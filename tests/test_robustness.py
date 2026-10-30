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


def test_litellm_relations_with_no_entities_make_no_call(monkeypatch):
    """`Literal[()]` is an AssertionError; an empty list is a valid extraction."""

    def refuse(**kwargs):
        raise AssertionError("no LLM call expected for an empty entity list")

    monkeypatch.setattr(relations_step.litellm, "responses", refuse)

    result = relations_step._get_relations_litellm(
        "Un texto.", [], model="openai/fake-model"
    )

    assert result == []


# --------------------------------------------------------------------------
# A failing LLM call must not delete the cluster
# --------------------------------------------------------------------------


class _FailingOn(ScriptedPredict):
    """ScriptedPredict that raises when asked about any item in `bad`."""

    def __init__(self, script, bad):
        super().__init__(script)
        self.bad = set(bad)

    def _predict(self, item, set):
        if item in self.bad:
            raise RuntimeError("provider unavailable")
        return super()._predict(item, set)


def test_failed_cluster_is_kept_unmerged(monkeypatch):
    """The exception was only logged, so the cluster's entities vanished
    while the relations that mention them stayed in the graph."""
    graph = Graph(
        entities={"oruga", "larva", "mariposa"},
        edges={"se convierte en"},
        relations={("oruga", "se convierte en", "mariposa")},
    )
    predictor = _FailingOn({}, bad={"oruga", "larva"})
    monkeypatch.setattr(llm_dedup_module.dspy, "Predict", lambda signature: predictor._predict)
    deduper = LLMDeduplicate(
        retrieval_model=FakeEncoder(), lm=None, graph=graph, language="es"
    )
    deduper.node_clusters = [["oruga", "larva"], ["mariposa"]]
    deduper.edge_clusters = [["se convierte en"]]

    result = deduper.deduplicate()

    assert result.entities == {"oruga", "larva", "mariposa"}
    assert result.edges == {"se convierte en"}
    assert result.relations == {("oruga", "se convierte en", "mariposa")}


# --------------------------------------------------------------------------
# Relations of an original name that is also another item's alias
# --------------------------------------------------------------------------


def test_relation_follows_original_name_not_alias_with_same_spelling(monkeypatch):
    """A is merged into B while the original B is merged into C.

    Entities end up {B, C}. The relation on the original B must go to C, but
    the old `if s not in entities` guard saw "B" in the entities and left it,
    so the relation silently pointed at A's representative instead.
    """
    graph = Graph(
        entities={"A", "B", "C"},
        edges={"p"},
        relations={("B", "p", "C"), ("A", "p", "C")},
    )
    script = {"A": ("B", []), "B": ("C", [])}
    deduper, _ = build(graph, script, monkeypatch=monkeypatch)
    deduper.node_clusters = [["C", "B", "A"]]  # pop() takes A, then B, then C
    deduper.edge_clusters = [["p"]]

    result = deduper.deduplicate()

    assert result.entities == {"B", "C"}
    assert result.entity_clusters == {"B": {"A"}, "C": {"B", "C"}}
    assert result.relations == {("C", "p", "C"), ("B", "p", "C")}


# --------------------------------------------------------------------------
# Explicit UTF-8
# --------------------------------------------------------------------------


@pytest.fixture
def cp1252_default(monkeypatch):
    """Make every text read/write that omits `encoding` use cp1252, as a
    Windows or `LANG=C` machine would."""
    real_open = builtins.open
    real_read_text = Path.read_text

    def fake_open(file, mode="r", buffering=-1, encoding=None, *args, **kwargs):
        if "b" not in mode and encoding is None:
            encoding = "cp1252"
        return real_open(file, mode, buffering, encoding, *args, **kwargs)

    def fake_read_text(self, encoding=None, errors=None):
        return real_read_text(self, encoding=encoding or "cp1252", errors=errors)

    monkeypatch.setattr(kg_gen_module, "open", fake_open, raising=False)
    monkeypatch.setattr(Path, "read_text", fake_read_text)


@pytest.mark.parametrize(
    "load",
    [entities_step._load_entities_prompt, relations_step._load_relations_prompt],
    ids=["entities", "relations"],
)
def test_spanish_prompt_files_are_read_as_utf8(cp1252_default, load):
    """"Extraerás" became "ExtraerÃ¡s" before reaching the model."""
    prompt = load("es")

    assert "Ã" not in prompt
    assert any(ch in prompt for ch in "áéíóúñ")


# --------------------------------------------------------------------------
# Prompt placeholders inside entity names
# --------------------------------------------------------------------------


@pytest.mark.parametrize("language", ["en", "es"])
def test_entity_named_like_a_placeholder_is_not_replaced_by_the_text(language):
    """Two successive `.replace` calls: the entity list was substituted first,
    so a literal `{input_data}` entity was then replaced with the source text."""
    prompt = relations_step._build_relations_user_prompt(
        language, ["{input_data}", "Ana"], "TEXTO {entities_str} fuente"
    )

    assert "- {input_data}\n- Ana" in prompt
    assert "TEXTO {entities_str} fuente" in prompt


