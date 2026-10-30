"""Offline tests for the optional linguistic profile (`linguistic_profile`).

The profile is off by default, and off has to mean "identical to v1.1": the
results of the attribution comparison come from that code. That is checked
against a snapshot of every prompt v1.1 sends, rendered the way DSPy renders
it (tests/snapshots/prompts_v1_1.json, written by tests/prompt_capture.py from the
v1.1 tag).

With "es-v2", the Spanish prompts get a block of Spanish rules appended and
nothing else changes, apart from the predicate examples that would contradict
those rules. Nothing here calls an LLM or touches the network.
"""

import json
import re
import typing
from pathlib import Path

import pytest
from pydantic import BaseModel

from src.kg_gen import kg_gen as kg_gen_module
from src.kg_gen.steps._1_get_entities import _entities_sig
from src.kg_gen.steps._2_get_relations import (
    _relation_model,
    extraction_sig,
    fallback_extraction_sig,
)
from src.kg_gen.utils.i18n import LINGUISTIC_PROFILES, validate_linguistic_profile
from tests.prompt_capture import capture_calls, capture_prompts, render

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "src" / "kg_gen" / "prompts"
SNAPSHOT = json.loads(
    (ROOT / "tests" / "snapshots" / "prompts_v1_1.json").read_text(encoding="utf-8")
)

RULES = {
    "entities": (PROMPTS / "reglas_es_v2_entidades.txt").read_text(encoding="utf-8"),
    "relations": (PROMPTS / "reglas_es_v2_relaciones.txt").read_text(encoding="utf-8"),
}


def _read(name):
    return (PROMPTS / name).read_text(encoding="utf-8")


def _with_rules(default_instructions, step):
    """What a DSPy signature's instructions become with the profile: the
    default ones, as DSPy renders them (cleandoc), then the rules. cleandoc
    also drops the rules' trailing newline."""
    return f"{default_instructions}\n\n{RULES[step].rstrip()}"


# --------------------------------------------------------------------------
# Off by default, and off is v1.1
# --------------------------------------------------------------------------


@pytest.mark.parametrize("language", ["en", "es"])
def test_default_prompts_are_exactly_v1_1(language):
    """Every route: DSPy text and conversation, fallback and fix, litellm."""
    assert capture_prompts(language) == SNAPSHOT[language]


def test_snapshot_covers_every_route():
    """A snapshot that lost a scenario would make the test above vacuous."""
    for language in ("en", "es"):
        calls = {name: [c["call"] for c in v] for name, v in SNAPSHOT[language].items()}
        assert calls == {
            "dspy_text": ["Predict", "Predict"],
            "dspy_text_fallback": ["Predict", "Predict", "Predict", "ChainOfThought"],
            "dspy_conversation": ["Predict", "Predict"],
            "litellm_text": ["litellm", "litellm"],
        }


def _offline_kggen(**kwargs):
    # A fake model name and key: constructing dspy.LM only stores them.
    return kg_gen_module.KGGen(model="openai/fake-model", api_key="test-key", **kwargs)


@pytest.mark.parametrize("language", ["en", "es"])
def test_kggen_has_no_profile_by_default(language):
    assert _offline_kggen(language=language).linguistic_profile is None


# --------------------------------------------------------------------------
# es-v2 appends the rules and changes nothing else
# --------------------------------------------------------------------------


def _relation_models(signature):
    """The Relation classes a signature carries in its list-typed fields."""
    models = {}
    for name, field in signature.fields.items():
        args = typing.get_args(field.annotation)
        if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
            models[name] = args[0]
    return models


def _plain(inputs):
    """Inputs with Relation instances as dicts: the two runs build their own
    Relation classes, and pydantic equality compares the class too."""
    return {
        key: [v.model_dump() if isinstance(v, BaseModel) else v for v in value]
        if isinstance(value, list)
        else value
        for key, value in inputs.items()
    }


def _schema_without_predicate_example(model):
    schema = model.model_json_schema()
    schema["properties"]["predicate"].pop("examples")
    return schema


@pytest.mark.parametrize(
    "scenario", ["dspy_text", "dspy_text_fallback", "dspy_conversation"]
)
def test_es_v2_dspy_signatures_differ_only_by_the_rules(scenario):
    default = capture_calls("es")[scenario]
    profile = capture_calls("es", "es-v2")[scenario]
    assert [c[0] for c in profile] == [c[0] for c in default]

    for (kind, default_sig, default_inputs), (_, profile_sig, profile_inputs) in zip(
        default, profile
    ):
        assert _plain(profile_inputs) == _plain(default_inputs)

        if kind == "ChainOfThought":
            # The fix step keeps predicates as they are: no rules there.
            assert profile_sig.instructions == default_sig.instructions
        else:
            step = "entities" if "entities" in default_sig.output_fields else "relations"
            assert profile_sig.instructions == _with_rules(default_sig.instructions, step)

        assert list(profile_sig.fields) == list(default_sig.fields)
        for name in default_sig.fields:
            assert (
                profile_sig.fields[name].json_schema_extra
                == default_sig.fields[name].json_schema_extra
            )

        default_models = _relation_models(default_sig)
        profile_models = _relation_models(profile_sig)
        assert profile_models.keys() == default_models.keys()
        for name, model in profile_models.items():
            assert model.model_fields["predicate"].examples == ["ser hermano de"]
            assert _schema_without_predicate_example(model) == (
                _schema_without_predicate_example(default_models[name])
            )


def test_es_v2_dspy_prompts_keep_the_dspy_output_format():
    """The rules show triples as (s | p | o), never as a JSON object that
    would compete with the format DSPy asks for (the ES-v2 draft did)."""
    for scenario, calls in capture_prompts("es", "es-v2").items():
        for call in calls:
            if call["call"] == "litellm":
                continue
            system = call["messages"][0]["content"]
            assert '{"relations"' not in system and '{"entities"' not in system
            assert "[[ ## completed ## ]]" in system


def test_es_v2_litellm_prompts_differ_only_by_rules_and_examples():
    default = capture_prompts("es")["litellm_text"]
    profile = capture_prompts("es", "es-v2")["litellm_text"]

    expected_system = [
        f"{_read('entidades_es_v2.txt').rstrip()}\n\n{RULES['entities']}",
        f"{_read('relaciones_es_v2.txt').rstrip()}\n\n{RULES['relations']}",
    ]
    for default_call, profile_call, system in zip(default, profile, expected_system):
        assert profile_call["messages"][0]["content"] == system
        assert profile_call["messages"][1:] == default_call["messages"][1:]
        assert profile_call["schema"] == default_call["schema"]


@pytest.mark.parametrize(
    "default_file, profile_file, replaced",
    [
        ("entidades.txt", "entidades_es_v2.txt", ["U.S."]),
        ("relaciones.txt", "relaciones_es_v2.txt", ["fundado_por", "relacionado_con"]),
    ],
)
def test_es_v2_litellm_files_only_replace_contradicting_examples(
    default_file, profile_file, replaced
):
    default_lines = _read(default_file).splitlines()
    profile_lines = _read(profile_file).splitlines()
    assert len(profile_lines) == len(default_lines)

    changed = [(a, b) for a, b in zip(default_lines, profile_lines) if a != b]
    assert changed
    for before, _ in changed:
        assert any(word in before for word in replaced)
    for word in replaced:
        assert word not in _read(profile_file)


def test_english_is_untouched_by_the_profile_code():
    """The profile never applies to English, and KGGen refuses the mix."""
    assert capture_prompts("en") == SNAPSHOT["en"]
    with pytest.raises(ValueError, match="applies to language 'es'"):
        _offline_kggen(language="en", linguistic_profile="es-v2")


# --------------------------------------------------------------------------
# The rules' own examples follow the rules
# --------------------------------------------------------------------------

_TRIPLE = re.compile(r"\(([^|()]+)\|([^|()]+)\|([^|()]+)\)")


def _examples(rules):
    """(entities, triples) for each example block of a rules file."""
    blocks = []
    for block in rules.split("\n\n"):
        lines = dict(
            line.split(": ", 1) for line in block.splitlines() if ": " in line
        )
        if "Entidades" in lines:
            entities = [e.strip() for e in lines["Entidades"].split(",")]
            triples = [
                tuple(part.strip() for part in match)
                for match in _TRIPLE.findall(lines.get("Relaciones", ""))
            ]
            blocks.append((entities, triples))
    return blocks


def test_rules_have_no_braces_or_json():
    for rules in RULES.values():
        assert "{" not in rules and "}" not in rules
        assert "salida:" not in rules


def test_relation_examples_stay_within_their_entity_list():
    """The same closure the hint for dangling relations asks the model for."""
    examples = _examples(RULES["relations"])
    assert len(examples) == 5
    for entities, triples in examples:
        assert triples
        for subject, _, obj in triples:
            assert subject in entities and obj in entities


def test_relation_examples_use_infinitive_predicates():
    """Rule 2: infinitive (or pronominal infinitive) first, spaces not underscores."""
    for _, triples in _examples(RULES["relations"]):
        for _, predicate, _ in triples:
            assert "_" not in predicate
            first = predicate.split()[0]
            assert re.search(r"(ar|er|ir)(se)?$", first), predicate


def test_entity_examples_extract_no_pronouns():
    examples = _examples(RULES["entities"])
    assert len(examples) == 3
    pronouns = {"él", "ella", "ellos", "ellas", "lo", "la", "le", "se"}
    for entities, _ in examples:
        assert not pronouns & {e.lower() for e in entities}


# --------------------------------------------------------------------------
# Validation and plumbing through KGGen
# --------------------------------------------------------------------------


def test_every_profile_names_a_supported_language():
    assert LINGUISTIC_PROFILES == {"es-v2": "es"}
    assert validate_linguistic_profile(None, "en") is None
    assert validate_linguistic_profile("es-v2", "es") == "es-v2"


def test_unknown_profile_is_rejected_at_construction():
    with pytest.raises(ValueError, match="Unknown linguistic_profile 'es-v3'"):
        _offline_kggen(language="es", linguistic_profile="es-v3")


@pytest.fixture
def recorded_steps(monkeypatch):
    """Stub both extraction steps and record the profile each one receives."""
    seen = []

    def fake_entities(*args, **kwargs):
        seen.append(("entities", kwargs["linguistic_profile"]))
        return ["Ana", "Luis"]

    def fake_relations(*args, **kwargs):
        seen.append(("relations", kwargs["linguistic_profile"]))
        return [("Ana", "conocer a", "Luis")]

    monkeypatch.setattr(kg_gen_module, "get_entities", fake_entities)
    monkeypatch.setattr(kg_gen_module, "get_relations", fake_relations)
    return seen


def test_generate_passes_the_instance_profile(recorded_steps):
    kg = _offline_kggen(language="es", linguistic_profile="es-v2")
    kg.generate("Ana conoce a Luis.", deduplication_method=None)
    assert recorded_steps == [("entities", "es-v2"), ("relations", "es-v2")]


def test_generate_without_a_profile_passes_none(recorded_steps):
    _offline_kggen(language="es").generate("Ana conoce a Luis.", deduplication_method=None)
    assert recorded_steps == [("entities", None), ("relations", None)]


def test_generate_accepts_a_per_call_profile(recorded_steps):
    kg = _offline_kggen(language="es")
    kg.generate("Ana conoce a Luis.", deduplication_method=None, linguistic_profile="es-v2")
    assert recorded_steps == [("entities", "es-v2"), ("relations", "es-v2")]


def test_profile_that_stops_applying_raises_before_extraction(recorded_steps):
    """A per-call language override can leave the instance profile without
    its language; that must not reach the LLM as Spanish rules in English."""
    kg = _offline_kggen(language="es", linguistic_profile="es-v2")
    with pytest.raises(ValueError, match="not 'en'"):
        kg.generate("Ana knows Luis.", deduplication_method=None, language="en")
    assert recorded_steps == []


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------


def test_profile_signatures_are_cached_apart_from_the_default():
    assert _entities_sig(False, "es", "es-v2") is _entities_sig(False, "es", "es-v2")
    assert _entities_sig(False, "es", "es-v2") is not _entities_sig(False, "es")

    relation = _relation_model("es", linguistic_profile="es-v2")
    assert relation is _relation_model("es", linguistic_profile="es-v2")
    assert relation is not _relation_model("es")
    assert extraction_sig(relation, False, "", "es", "es-v2") is extraction_sig(
        relation, False, "", "es", "es-v2"
    )


def test_profile_fallback_signature_is_not_cached():
    relation_size = _relation_model.cache_info().currsize
    sig_size = extraction_sig.cache_info().currsize

    relation, sig = fallback_extraction_sig(["Ana", "Madrid"], False, "", "es", "es-v2")

    assert "Madrid" in relation.__doc__
    assert relation.model_fields["predicate"].examples == ["ser hermano de"]
    assert sig.instructions.endswith(RULES["relations"].rstrip())
    assert _relation_model.cache_info().currsize == relation_size
    assert extraction_sig.cache_info().currsize == sig_size


def test_render_of_a_profile_call_is_stable():
    """Rendering twice gives the same prompt: no order-dependent output."""
    calls = capture_calls("es", "es-v2")["dspy_text"]
    assert [render(c) for c in calls] == [render(c) for c in calls]
