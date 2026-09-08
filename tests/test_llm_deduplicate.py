"""Offline tests for the LLM-based deduplication path.

`dspy.Predict` is replaced by a scripted stub and the retrieval model by a
deterministic fake encoder, so nothing here calls an LLM, downloads weights,
or touches the network.

Two concerns are covered together, deliberately:

  - The Spanish adaptation: that `language` reaches the prompt tables and
    that an alias produced in Spanish survives into the graph untouched.
  - The shared bookkeeping invariants, which are not language-specific and
    are asserted for English too, so a fix there cannot quietly favor the
    Spanish configuration in the es/en comparison.
"""

import numpy as np
import pytest

from src.kg_gen.models import Graph
from src.kg_gen.utils import llm_deduplicate as llm_dedup_module
from src.kg_gen.utils.llm_deduplicate import (
    _ALIAS_OUTPUT_DESC,
    _DEDUPLICATE_INSTRUCTIONS,
    _DUPLICATES_OUTPUT_DESC,
    _ITEM_NOUNS,
    LLMDeduplicate,
)

# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------


class FakeEncoder:
    """Deterministic stand-in for SentenceTransformer.

    Vectors are derived from a stable hash of the string, so they carry no
    real semantics. That is fine here: every test scripts the LLM's answers
    directly, so retrieval only has to be reproducible, not meaningful.
    """

    def encode(self, texts, show_progress_bar=False):
        vectors = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vectors.append(rng.random(8, dtype=np.float32))
        return np.array(vectors, dtype=np.float32)


class ScriptedPredict:
    """Stands in for `dspy.Predict(Signature)`.

    Built from a mapping of item -> (alias, duplicates). Items missing from
    the script resolve to themselves with no duplicates, which keeps each
    test's script down to the case it is actually about.
    """

    def __init__(self, script):
        self.script = script
        self.signatures = []

    def __call__(self, signature):
        # `dspy.Predict(Signature)` returns the callable predictor; record the
        # signature so tests can assert on the prompt text it was built with.
        self.signatures.append(signature)
        return self._predict

    def _predict(self, item, set):
        alias, duplicates = self.script.get(item, (item, []))
        return type("Prediction", (), {"alias": alias, "duplicates": duplicates})()


def build(graph, script, language="en", monkeypatch=None):
    """Construct an LLMDeduplicate with the LLM and encoder stubbed out."""
    predictor = ScriptedPredict(script)
    monkeypatch.setattr(llm_dedup_module.dspy, "Predict", predictor)

    deduper = LLMDeduplicate(
        retrieval_model=FakeEncoder(),
        lm=None,
        graph=graph,
        language=language,
    )
    return deduper, predictor


# --------------------------------------------------------------------------
# Prompt tables
#
# As in test_multilingual.py, the English strings are written out by hand so
# that an accidental edit to an "en" value has to be made twice, deliberately:
# changing it is a real behavior change against upstream kg-gen.
# --------------------------------------------------------------------------

EN_INSTRUCTIONS = (
    "Find duplicate {plural} for the item and an alias that best represents "
    "the duplicates. Duplicates are those that are the same in meaning, such "
    "as with variation in tense, plural form, stem form, case, abbreviation, "
    "shorthand. Return an empty list if there are none."
)


def test_english_dedup_instructions_unchanged():
    assert _DEDUPLICATE_INSTRUCTIONS["en"] == EN_INSTRUCTIONS


def test_english_output_descriptions_unchanged():
    assert _DUPLICATES_OUTPUT_DESC["en"] == "Exact matches to items in {plural} set"
    assert _ALIAS_OUTPUT_DESC["en"] == (
        "Best {singular} name to represent the duplicates, ideally from the "
        "{plural} set"
    )


def test_english_item_nouns_unchanged():
    assert _ITEM_NOUNS["en"]["node"] == ("entities", "entity")
    assert _ITEM_NOUNS["en"]["edge"] == ("edges", "edge")


def test_spanish_item_nouns():
    assert _ITEM_NOUNS["es"]["node"] == ("entidades", "entidad")
    assert _ITEM_NOUNS["es"]["edge"] == ("aristas", "arista")


@pytest.mark.parametrize(
    "table", [_DEDUPLICATE_INSTRUCTIONS, _DUPLICATES_OUTPUT_DESC, _ALIAS_OUTPUT_DESC]
)
def test_spanish_prompts_actually_differ(table):
    """Guards against an untranslated table being copied from English."""
    assert table["es"] != table["en"]


@pytest.mark.parametrize(
    "item_type,expected_plural",
    [("node", "entidades"), ("edge", "aristas")],
)
def test_spanish_signature_reaches_the_prompt(monkeypatch, item_type, expected_plural):
    """`language` must survive all the way into the built dspy signature.

    This is the step that names entities in the final graph, so an English
    prompt here would rename Spanish entities into English.
    """
    graph = Graph(entities={"libros"}, edges={"trata de"}, relations=set())
    deduper, predictor = build(graph, {}, language="es", monkeypatch=monkeypatch)

    items = ["libros"] if item_type == "node" else ["trata de"]
    deduper.deduplicate_cluster(items, item_type)

    instructions = predictor.signatures[0].instructions
    assert expected_plural in instructions
    assert "Encuentra" in instructions
    # The English variant must not leak through alongside it.
    assert "Find duplicate" not in instructions


def test_english_signature_unaffected(monkeypatch):
    graph = Graph(entities={"books"}, edges={"is about"}, relations=set())
    deduper, predictor = build(graph, {}, language="en", monkeypatch=monkeypatch)

    deduper.deduplicate_cluster(["books"], "node")

    instructions = predictor.signatures[0].instructions
    assert instructions == EN_INSTRUCTIONS.format(plural="entities")


# --------------------------------------------------------------------------
# Alias collision
# --------------------------------------------------------------------------


def test_colliding_aliases_keep_every_member(monkeypatch):
    """Two items resolving to one alias must both stay in the cluster.

    Spanish raises the collision rate, since more morphological variants
    collapse onto the same lemma, but the bookkeeping is shared so this is
    asserted on the shared path.
    """
    # A non-empty edge set is required only because BM25Okapi divides by the
    # average document length, which is zero for an empty corpus.
    graph = Graph(
        entities={"libros", "libro", "revista"}, edges={"trata de"}, relations=set()
    )
    script = {"libros": ("libro", []), "libro": ("libro", [])}
    deduper, _ = build(graph, script, language="es", monkeypatch=monkeypatch)

    _, clusters = deduper.deduplicate_cluster(["libros", "libro", "revista"], "node")

    assert clusters["libro"] == {"libros", "libro"}


def test_alias_differing_from_item_is_still_reachable(monkeypatch):
    """An alias with no duplicates found must still map its item.

    `items` takes the alias unconditionally, so keying the cluster by the
    item instead would put "libro" in the graph while relations still said
    "libros", with nothing to bridge them.
    """
    graph = Graph(entities={"libros"}, edges={"trata de"}, relations=set())
    deduper, _ = build(
        graph, {"libros": ("libro", [])}, language="es", monkeypatch=monkeypatch
    )

    items, clusters = deduper.deduplicate_cluster(["libros"], "node")

    assert items == {"libro"}
    assert clusters == {"libro": {"libros"}}


def test_merge_clusters_unions_on_collision(monkeypatch):
    """The same alias arising in two clusters must union, not overwrite."""
    graph = Graph(entities={"libros", "libro"}, edges={"trata de"}, relations=set())
    deduper, _ = build(graph, {}, language="es", monkeypatch=monkeypatch)

    target = {"libro": {"libros"}}
    deduper._merge_clusters(target, {"libro": {"libro"}, "revista": {"revistas"}})

    assert target == {"libro": {"libros", "libro"}, "revista": {"revistas"}}


def test_deduplicate_merges_aliases_shared_by_two_clusters(monkeypatch):
    """`deduplicate` must union across clusters, not just via the helper.

    The clusters are assigned directly rather than via `cluster()`, which
    would need more than 128 items to produce more than one of them.
    """
    graph = Graph(
        entities={"libros", "libro", "revistas", "revista"},
        edges={"trata de"},
        relations={
            ("libros", "trata de", "revistas"),
            ("libro", "trata de", "revista"),
        },
    )
    script = {
        "libros": ("libro", []),
        "libro": ("libro", []),
        "revistas": ("revista", []),
        "revista": ("revista", []),
    }
    deduper, _ = build(graph, script, language="es", monkeypatch=monkeypatch)
    # Same alias reached from two separate clusters processed in parallel.
    deduper.node_clusters = [["libros", "revistas"], ["libro", "revista"]]
    deduper.edge_clusters = [["trata de"]]

    result = deduper.deduplicate()

    assert result.entity_clusters["libro"] == {"libros", "libro"}
    assert result.entity_clusters["revista"] == {"revistas", "revista"}
    assert_referentially_intact(result)


# --------------------------------------------------------------------------
# Referential integrity
#
# The invariant that should outlive every other test in this file: after
# deduplication, no relation may point at an entity or edge that is not in
# the graph. A dangling endpoint is an orphan node, which is precisely the
# fragmentation the thesis measures -- it must not be produced by the
# pipeline itself.
# --------------------------------------------------------------------------


def assert_referentially_intact(graph):
    for subject, predicate, object_ in graph.relations:
        assert subject in graph.entities, f"dangling subject: {subject!r}"
        assert object_ in graph.entities, f"dangling object: {object_!r}"
        assert predicate in graph.edges, f"dangling predicate: {predicate!r}"


@pytest.mark.parametrize(
    "language,graph,script",
    [
        pytest.param(
            "es",
            Graph(
                entities={"libros", "libro", "autora"},
                edges={"fue escrito por", "fueron escritos por"},
                relations={
                    ("libros", "fueron escritos por", "autora"),
                    ("libro", "fue escrito por", "autora"),
                },
            ),
            {
                "libros": ("libro", []),
                "libro": ("libro", []),
                "fueron escritos por": ("ser escrito por", []),
                "fue escrito por": ("ser escrito por", []),
            },
            id="spanish-morphological-collapse",
        ),
        pytest.param(
            "en",
            Graph(
                entities={"books", "book", "author"},
                edges={"was written by", "were written by"},
                relations={
                    ("books", "were written by", "author"),
                    ("book", "was written by", "author"),
                },
            ),
            {
                "books": ("book", []),
                "book": ("book", []),
                "were written by": ("written by", []),
                "was written by": ("written by", []),
            },
            id="english-non-regression",
        ),
    ],
)
def test_deduplicate_preserves_referential_integrity(
    monkeypatch, language, graph, script
):
    deduper, _ = build(graph, script, language=language, monkeypatch=monkeypatch)
    deduper.cluster()

    result = deduper.deduplicate()

    assert_referentially_intact(result)


def test_spanish_alias_propagates_unchanged(monkeypatch):
    """A Spanish alias must reach the graph verbatim.

    No accidental casing, accent, or whitespace transformation may be applied
    on the way -- the alias the LLM returns is the entity's final name.
    """
    graph = Graph(
        entities={"Ciudades", "ciudad"},
        edges={"está en"},
        relations={("Ciudades", "está en", "ciudad")},
    )
    script = {"Ciudades": ("Ciudad de México", []), "ciudad": ("Ciudad de México", [])}
    deduper, _ = build(graph, script, language="es", monkeypatch=monkeypatch)
    deduper.cluster()

    result = deduper.deduplicate()

    assert "Ciudad de México" in result.entities
    assert "está en" in result.edges
    assert_referentially_intact(result)
