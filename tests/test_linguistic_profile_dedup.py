"""The es-v2 profile in deduplication: gender and number tell entities apart,
and a predicate's agreement with its subject does not tell relations apart.

Off by default, and off has to be v1.1's dedup, so every case states the
default result next to the es-v2 one. No LLM and no network: SemHash is
replaced by stubs, or loaded from the local Hugging Face cache, and
dspy.Predict by the ScriptedPredict of tests/test_llm_deduplicate.py.
"""

from types import SimpleNamespace

import pytest

from src.kg_gen import kg_gen as kg_gen_module
from src.kg_gen.models import Graph
from src.kg_gen.steps import _3_deduplicate as dedup_step
from src.kg_gen.steps._3_deduplicate import DeduplicateMethod, run_deduplication
from src.kg_gen.utils import deduplicate as dedup_module
from src.kg_gen.utils import llm_deduplicate as llm_dedup_module
from src.kg_gen.utils.deduplicate import DeduplicateList
from src.kg_gen.utils.llm_deduplicate import _DEDUPLICATE_INSTRUCTIONS, LLMDeduplicate
from src.kg_gen.utils.text_normalization import TextNormalizer

from test_deduplicate import KeepAllSemHash
from test_llm_deduplicate import FakeEncoder, ScriptedPredict

BASE_ENCODER = "minishlab/potion-base-8M"
MULTILINGUAL_ENCODER = "minishlab/potion-multilingual-128M"


def scripted_semhash(merges):
    """A SemHash stub that merges each record in `merges` into its value,
    the way SemHash reports a near-duplicate, and keeps everything else."""

    class ScriptedSemHash:
        def __init__(self, records):
            self.records = records

        @classmethod
        def from_records(cls, records, **_):
            return cls(records)

        def self_deduplicate(self, threshold):
            return SimpleNamespace(
                selected=[r for r in self.records if r not in merges],
                filtered=[
                    SimpleNamespace(record=r, duplicates=[(merges[r], 0.99)])
                    for r in self.records
                    if r in merges
                ],
            )

    return ScriptedSemHash


def _nodes(dedup, items):
    """The items grouped by the label each one ends up with in the graph."""
    by_label = {}
    for item in items:
        by_label.setdefault(dedup.items_map[dedup.original_map[item]], set()).add(item)
    return {frozenset(group) for group in by_label.values()}


def _dedup(items, profile, item_type="entity", semhash_model=None):
    dedup = DeduplicateList(
        language="es",
        semhash_model=semhash_model,
        item_type=item_type,
        linguistic_profile=profile,
    )
    dedup.deduplicate(items)
    return dedup


def _apart(*items):
    return {frozenset([item]) for item in items}


def _together(*items):
    return {frozenset(items)}


# --------------------------------------------------------------------------
# What the entity key reads
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, features",
    [
        ("el investigador", ("Masc", "Sing")),
        ("los investigadores", ("Masc", "Plur")),
        # The head marks only number; the article supplies the gender.
        ("la estudiante", ("Fem", "Sing")),
        ("las investigadoras", ("Fem", "Plur")),
        # The head's own gender wins over the article's.
        ("el agua", ("Fem", "Sing")),
        ("estudiante", (None, "Sing")),
        ("Ana Ríos", (None, None)),
    ],
)
def test_grammatical_features(text, features):
    assert TextNormalizer(language="es").grammatical_features(text) == features


def test_english_has_no_grammatical_features():
    assert TextNormalizer(language="en").grammatical_features("the actress") == (
        None,
        None,
    )


# --------------------------------------------------------------------------
# Grouping before SemHash
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "items, default, es_v2",
    [
        (
            ["el investigador", "los investigadores"],
            _together("el investigador", "los investigadores"),
            _apart("el investigador", "los investigadores"),
        ),
        (
            ["la investigadora", "las investigadoras"],
            _together("la investigadora", "las investigadoras"),
            _apart("la investigadora", "las investigadoras"),
        ),
        (
            ["dinosaurio", "dinosaurios"],
            _together("dinosaurio", "dinosaurios"),
            _apart("dinosaurio", "dinosaurios"),
        ),
        # Common-gender noun: only the article marks the gender.
        (
            ["el estudiante", "la estudiante"],
            _together("el estudiante", "la estudiante"),
            _apart("el estudiante", "la estudiante"),
        ),
        # Homonyms told apart only by gender.
        (
            ["el capital", "la capital"],
            _together("el capital", "la capital"),
            _apart("el capital", "la capital"),
        ),
        # Gender carried by the noun itself already stayed apart in v1.1.
        (
            ["el investigador", "la investigadora"],
            _apart("el investigador", "la investigadora"),
            _apart("el investigador", "la investigadora"),
        ),
        # An article that changes neither still merges.
        (
            ["la ciudad", "ciudad"],
            _together("la ciudad", "ciudad"),
            _together("la ciudad", "ciudad"),
        ),
        (
            ["el agua", "agua"],
            _together("el agua", "agua"),
            _together("el agua", "agua"),
        ),
        # A bare common-gender noun joins the only gendered form there is...
        (
            ["estudiante", "el estudiante"],
            _together("estudiante", "el estudiante"),
            _together("estudiante", "el estudiante"),
        ),
        # ...and stays on its own when it could be either.
        (
            ["estudiante", "el estudiante", "la estudiante"],
            _together("estudiante", "el estudiante", "la estudiante"),
            _apart("estudiante", "el estudiante", "la estudiante"),
        ),
    ],
)
def test_entity_grouping(monkeypatch, items, default, es_v2):
    monkeypatch.setattr(dedup_module, "SemHash", KeepAllSemHash)

    assert _nodes(_dedup(items, None), items) == default
    assert _nodes(_dedup(items, "es-v2"), items) == es_v2


@pytest.mark.parametrize(
    "items, default, es_v2",
    [
        # spaCy tags the participle after "estar" as an adjective, which
        # v1.1's predicate key leaves inflected.
        (
            ["estar casado con", "estar casada con", "estar casados con"],
            _apart("estar casado con", "estar casada con", "estar casados con"),
            _together("estar casado con", "estar casada con", "estar casados con"),
        ),
        (
            ["ser rico en", "ser rica en"],
            _apart("ser rico en", "ser rica en"),
            _together("ser rico en", "ser rica en"),
        ),
        # The participle of a passive is a verb: already one key in v1.1.
        (
            ["ser fundado por", "ser fundada por"],
            _together("ser fundado por", "ser fundada por"),
            _together("ser fundado por", "ser fundada por"),
        ),
        # A noun with its own gender still tells relations apart.
        (
            ["ser hermano de", "ser hermana de"],
            _apart("ser hermano de", "ser hermana de"),
            _apart("ser hermano de", "ser hermana de"),
        ),
    ],
)
def test_predicate_grouping(monkeypatch, items, default, es_v2):
    monkeypatch.setattr(dedup_module, "SemHash", KeepAllSemHash)

    assert _nodes(_dedup(items, None, "edge"), items) == default
    assert _nodes(_dedup(items, "es-v2", "edge"), items) == es_v2


# --------------------------------------------------------------------------
# Merges SemHash proposes between different keys
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "items, merges, default, es_v2",
    [
        (
            ["el profesor", "la profesora"],
            {"profesora": "profesor"},
            _together("el profesor", "la profesora"),
            _apart("el profesor", "la profesora"),
        ),
        (
            ["la ciudad", "la urbe"],
            {"urbe": "ciudad"},
            _together("la ciudad", "la urbe"),
            _together("la ciudad", "la urbe"),
        ),
        # The surviving record holds a masculine and a feminine node, and a
        # bare common-gender noun could join either.
        (
            ["el estudiante", "la estudiante", "periodista"],
            {"periodista": "estudiante"},
            _together("el estudiante", "la estudiante", "periodista"),
            _apart("el estudiante", "la estudiante", "periodista"),
        ),
    ],
)
def test_semhash_merges_follow_gender_and_number(
    monkeypatch, items, merges, default, es_v2
):
    monkeypatch.setattr(dedup_module, "SemHash", scripted_semhash(merges))

    assert _nodes(_dedup(items, None), items) == default
    assert _nodes(_dedup(items, "es-v2"), items) == es_v2


@pytest.mark.parametrize(
    "encoder_name, items, default, es_v2",
    [
        (
            BASE_ENCODER,
            ["el investigador principal", "los investigadores principales"],
            _together("el investigador principal", "los investigadores principales"),
            _apart("el investigador principal", "los investigadores principales"),
        ),
        (
            MULTILINGUAL_ENCODER,
            ["el estudiante", "la estudiante"],
            _together("el estudiante", "la estudiante"),
            _apart("el estudiante", "la estudiante"),
        ),
        (
            MULTILINGUAL_ENCODER,
            ["el capital", "la capital"],
            _together("el capital", "la capital"),
            _apart("el capital", "la capital"),
        ),
        # Merges between forms that agree go through as in v1.1.
        (
            BASE_ENCODER,
            ["el cambio climático", "cambio climatico"],
            _together("el cambio climático", "cambio climatico"),
            _together("el cambio climático", "cambio climatico"),
        ),
        (
            MULTILINGUAL_ENCODER,
            ["el sistema solar", "sistema solar interior"],
            _together("el sistema solar", "sistema solar interior"),
            _together("el sistema solar", "sistema solar interior"),
        ),
    ],
)
def test_real_semhash(semhash_encoders, encoder_name, items, default, es_v2):
    encoder = semhash_encoders(encoder_name)

    assert _nodes(_dedup(items, None, semhash_model=encoder), items) == default
    assert _nodes(_dedup(items, "es-v2", semhash_model=encoder), items) == es_v2


def test_es_v2_counts_what_merged(monkeypatch):
    monkeypatch.setattr(dedup_module, "SemHash", KeepAllSemHash)
    items = ["la ciudad", "ciudad", "el estudiante", "la estudiante"]

    dedup = _dedup(items, "es-v2")

    assert dedup.total_items == 4
    assert dedup.deduplicated_items == 3
    assert dedup.duplicate_items == 1


def test_es_v2_graph_keeps_its_relations(monkeypatch):
    monkeypatch.setattr(dedup_module, "SemHash", KeepAllSemHash)
    graph = Graph(
        entities={"el estudiante", "la estudiante", "la ciudad", "ciudad"},
        edges={"vivir en"},
        relations={
            ("el estudiante", "vivir en", "la ciudad"),
            ("la estudiante", "vivir en", "ciudad"),
        },
    )

    out = dedup_module.run_semhash_deduplication(
        graph, language="es", linguistic_profile="es-v2"
    )

    assert out.entities == {"el estudiante", "la estudiante", "ciudad"}
    assert out.relations == {
        ("el estudiante", "vivir en", "ciudad"),
        ("la estudiante", "vivir en", "ciudad"),
    }


# --------------------------------------------------------------------------
# LLM dedup prompt
# --------------------------------------------------------------------------


def _instructions(language, profile, item_type, monkeypatch):
    predictor = ScriptedPredict({})
    monkeypatch.setattr(llm_dedup_module.dspy, "Predict", predictor)
    item = "Ana" if item_type == "node" else "conocer a"
    deduper = LLMDeduplicate(
        retrieval_model=FakeEncoder(),
        lm=None,
        graph=Graph(entities={"Ana"}, edges={"conocer a"}, relations=set()),
        language=language,
        linguistic_profile=profile,
    )
    deduper.deduplicate_cluster([item], item_type)
    return predictor.signatures[0].instructions


@pytest.mark.parametrize("language", ["en", "es"])
@pytest.mark.parametrize("item_type, plural", [("node", "entities"), ("edge", "edges")])
def test_default_llm_dedup_prompt_is_v1_1(monkeypatch, language, item_type, plural):
    if language == "es":
        plural = {"entities": "entidades", "edges": "aristas"}[plural]

    assert _instructions(
        language, None, item_type, monkeypatch
    ) == _DEDUPLICATE_INSTRUCTIONS[language].format(plural=plural)


@pytest.mark.parametrize(
    "item_type, plural, rules_file, rule",
    [
        ("node", "entidades", "reglas_es_v2_fusion_entidades.txt", "la estudiante"),
        ("edge", "aristas", "reglas_es_v2_fusion_aristas.txt", "estar casada con"),
    ],
)
def test_es_v2_llm_dedup_prompt_appends_its_rules(
    monkeypatch, item_type, plural, rules_file, rule
):
    rules = llm_dedup_module._read_prompt(rules_file)

    instructions = _instructions("es", "es-v2", item_type, monkeypatch)

    default = _DEDUPLICATE_INSTRUCTIONS["es"].format(plural=plural)
    # DSPy strips the file's trailing newline from the instructions.
    assert instructions == f"{default}\n\n{rules.rstrip()}"
    assert rule in rules


def test_dedup_rules_have_no_braces():
    for name in (
        "reglas_es_v2_fusion_entidades.txt",
        "reglas_es_v2_fusion_aristas.txt",
    ):
        rules = llm_dedup_module._read_prompt(name)
        assert "{" not in rules and "}" not in rules


# --------------------------------------------------------------------------
# The profile reaches deduplication
# --------------------------------------------------------------------------


@pytest.fixture
def recorded_dedup(monkeypatch):
    """Stub both dedup stages and record the profile each one receives."""
    seen = []

    def fake_semhash(graph, threshold, **kwargs):
        seen.append(("semhash", kwargs["linguistic_profile"]))
        return graph

    class FakeLLMDeduplicate:
        def __init__(self, retrieval_model, lm, graph, **kwargs):
            seen.append(("llm", kwargs["linguistic_profile"]))
            self.graph = graph

        def cluster(self):
            pass

        def deduplicate(self):
            return self.graph

    monkeypatch.setattr(dedup_step, "run_semhash_deduplication", fake_semhash)
    monkeypatch.setattr(dedup_step, "LLMDeduplicate", FakeLLMDeduplicate)
    return seen


@pytest.mark.parametrize("profile", [None, "es-v2"])
def test_run_deduplication_passes_the_profile_to_both_stages(recorded_dedup, profile):
    graph = Graph(entities={"Ana"}, edges=set(), relations=set())

    run_deduplication(
        None,
        graph,
        method=DeduplicateMethod.FULL,
        retrieval_model=object(),
        language="es",
        linguistic_profile=profile,
    )

    assert recorded_dedup == [("semhash", profile), ("llm", profile)]


def _offline_kggen(**kwargs):
    # A fake model name and key: constructing dspy.LM only stores them.
    return kg_gen_module.KGGen(model="openai/fake-model", api_key="test-key", **kwargs)


def _graph():
    return Graph(entities={"Ana"}, edges=set(), relations=set())


@pytest.fixture
def recorded_profiles(monkeypatch):
    """Stub KGGen's run_deduplication and record the profile it receives.

    Patched on KGGen's module: it imports the package as `kg_gen.*`, a
    separate copy from the `src.kg_gen.*` modules imported above.
    """
    seen = []

    def fake_run_deduplication(**kwargs):
        seen.append(kwargs["linguistic_profile"])
        return kwargs["graph"]

    monkeypatch.setattr(kg_gen_module, "run_deduplication", fake_run_deduplication)
    return seen


def test_deduplicate_uses_the_instance_profile(recorded_profiles):
    kg = _offline_kggen(language="es", linguistic_profile="es-v2")
    kg.deduplicate(_graph(), method="semhash")
    assert recorded_profiles == ["es-v2"]


def test_deduplicate_accepts_a_per_call_profile(recorded_profiles):
    kg = _offline_kggen(language="es")
    kg.deduplicate(_graph(), method="semhash", linguistic_profile="es-v2")
    assert recorded_profiles == ["es-v2"]


def test_deduplicate_without_a_profile_passes_none(recorded_profiles):
    _offline_kggen(language="es").deduplicate(_graph(), method="semhash")
    assert recorded_profiles == [None]


def test_deduplicate_rejects_a_profile_for_another_language(recorded_profiles):
    kg = _offline_kggen(language="es", linguistic_profile="es-v2")
    with pytest.raises(ValueError, match="not 'en'"):
        kg.deduplicate(_graph(), method="semhash", language="en")
    assert recorded_profiles == []


def test_generate_passes_its_profile_to_deduplication(monkeypatch, recorded_profiles):
    monkeypatch.setattr(kg_gen_module, "get_entities", lambda *a, **k: ["Ana"])
    monkeypatch.setattr(kg_gen_module, "get_relations", lambda *a, **k: [])
    kg = _offline_kggen(language="es")

    kg.generate("Ana.", deduplication_method="semhash", linguistic_profile="es-v2")

    assert recorded_profiles == ["es-v2"]
