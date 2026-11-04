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

# By module name, like `fixtures`: a bare `pytest` also collects mcp/tests, a
# package that takes the name `tests`.
from test_llm_deduplicate import FakeEncoder, ScriptedPredict, build

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


def test_from_file_reads_utf8(cp1252_default, tmp_path):
    path = tmp_path / "grafo.json"
    path.write_text(
        json.dumps(
            {
                "entities": ["Niño", "José"],
                "edges": ["quiere a"],
                "relations": [["Niño", "quiere a", "José"]],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    graph = kg_gen_module.KGGen.from_file(str(path))

    assert graph.entities == {"Niño", "José"}


# --------------------------------------------------------------------------
# export_graph
# --------------------------------------------------------------------------


def test_export_graph_to_a_bare_filename(tmp_path, monkeypatch):
    """`os.path.dirname("grafo.json")` is "", and `os.makedirs("")` raises."""
    monkeypatch.chdir(tmp_path)
    graph = Graph(entities={"Niño"}, edges={"es"}, relations=set())

    kg_gen_module.KGGen.export_graph(graph, "grafo.json")

    assert json.loads((tmp_path / "grafo.json").read_text(encoding="utf-8"))[
        "entities"
    ] == ["Niño"]


# --------------------------------------------------------------------------
# temperature=0.0 as an override
# --------------------------------------------------------------------------


@pytest.fixture
def recorded_init(monkeypatch):
    """A KGGen whose extraction steps are stubbed and whose `init_model`
    calls are recorded."""
    monkeypatch.setattr(
        kg_gen_module, "get_entities", lambda *a, **k: ["Ana", "Luis"]
    )
    monkeypatch.setattr(
        kg_gen_module, "get_relations", lambda *a, **k: [("Ana", "conoce a", "Luis")]
    )
    kg = kg_gen_module.KGGen(
        model="openai/fake-model", api_key="test-key", temperature=0.8
    )
    calls = []
    real_init = kg.init_model

    def spy(**kwargs):
        calls.append(kwargs)
        return real_init(**kwargs)

    monkeypatch.setattr(kg, "init_model", spy)
    return kg, calls


def test_generate_applies_a_zero_temperature_override(recorded_init):
    """`any([..., 0.0])` is False and `0.0 or self.temperature` is 0.8."""
    kg, calls = recorded_init

    kg.generate("Ana conoce a Luis.", deduplication_method=None, temperature=0.0)

    assert [c["temperature"] for c in calls] == [0.0]
    assert kg.temperature == 0.0


def test_generate_without_overrides_does_not_reinitialize(recorded_init):
    kg, calls = recorded_init

    kg.generate("Ana conoce a Luis.", deduplication_method=None)

    assert calls == []
    assert kg.temperature == 0.8


def test_deduplicate_applies_a_zero_temperature_override(recorded_init, monkeypatch):
    kg, calls = recorded_init
    monkeypatch.setattr(kg_gen_module, "run_deduplication", lambda **kw: kw["graph"])
    graph = Graph(entities={"Ana"}, edges=set(), relations=set())

    kg.deduplicate(graph, temperature=0.0)

    assert [c["temperature"] for c in calls] == [0.0]


# --------------------------------------------------------------------------
# Deduplication method validation
# --------------------------------------------------------------------------


def test_unknown_dedup_method_is_rejected_up_front():
    """It used to reach `return deduplicated_graph` unassigned
    (UnboundLocalError) or hide behind "No retrieval model provided"."""
    graph = Graph(entities={"Ana"}, edges=set(), relations=set())

    with pytest.raises(ValueError, match="bogus"):
        run_deduplication(None, graph, method="bogus", retrieval_model=object())


def test_dedup_method_accepts_its_string_value(monkeypatch):
    seen = {}

    def fake_semhash(graph, threshold, language, semhash_model_name, linguistic_profile):
        seen["called"] = True
        return graph

    monkeypatch.setattr(
        "src.kg_gen.steps._3_deduplicate.run_semhash_deduplication", fake_semhash
    )
    graph = Graph(entities={"Ana"}, edges=set(), relations=set())

    run_deduplication(None, graph, method="semhash")

    assert seen == {"called": True}
    assert DeduplicateMethod("semhash") is DeduplicateMethod.SEMHASH


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


# --------------------------------------------------------------------------
# Visualization: labels are data, not markup
# --------------------------------------------------------------------------

PAYLOAD = "</script><script id=proof>void 0</script>"


def test_visualize_does_not_let_a_label_close_the_data_script(tmp_path):
    graph = Graph(
        entities={PAYLOAD, "Ana"},
        edges={"conoce a"},
        relations={(PAYLOAD, "conoce a", "Ana")},
    )

    html = visualize(graph, str(tmp_path / "g.html")).read_text(encoding="utf-8")

    assert "<script id=proof>" not in html
    assert "</script><script" not in html


def test_template_escapes_labels_before_innerhtml():
    # Every dynamic fragment goes through `safeHTML`, so no template literal
    # that is assigned to innerHTML interpolates a raw value.
    assert "function safeHTML" in HTML_TEMPLATE
