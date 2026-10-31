"""Offline tests for `forward_context` (exp/core-extraction).

Upstream kg-gen accepts `context` in generate() but never shows it to the
extractors. With forward_context=True it is appended to the entity and
relation prompts on every route. Off, nothing changes: the v1.1 snapshot test
in test_linguistic_profile.py keeps checking that. Nothing here calls an LLM
or touches the network.
"""

from pathlib import Path

import pytest

from src.kg_gen import kg_gen as kg_gen_module
from src.kg_gen.steps._1_get_entities import _entities_sig

# By module name, like `fixtures`: a bare `pytest` also collects mcp/tests, a
# package that takes the name `tests`.
from prompt_capture import capture_calls

PROMPTS = Path(__file__).resolve().parent.parent / "src" / "kg_gen" / "prompts"
CONTEXT = "El texto es un ensayo escolar sobre los dinosaurios."
MULTILINE_CONTEXT = "Ensayo escolar.\nTema: los dinosaurios."


def _with_context(default_instructions, context):
    # The relation defaults end in a space (`f"{instructions} {context}"` with
    # an empty context); the entity defaults do not.
    return f"{default_instructions.rstrip(' ')} {context}"


def _step(signature):
    return "entidades" if "entities" in signature.output_fields else "relaciones"


# --------------------------------------------------------------------------
# What the extractors receive
# --------------------------------------------------------------------------


@pytest.mark.parametrize("language", ["en", "es"])
@pytest.mark.parametrize(
    "scenario", ["dspy_text", "dspy_text_fallback", "dspy_conversation"]
)
def test_context_follows_the_instructions_of_every_extraction_signature(
    language, scenario
):
    default = capture_calls(language)[scenario]
    with_context = capture_calls(language, context=CONTEXT)[scenario]
    assert [c[0] for c in with_context] == [c[0] for c in default]

    for (kind, default_sig, default_inputs), (_, sig, inputs) in zip(
        default, with_context
    ):
        for key in ("source_text", "entities"):
            assert inputs.get(key) == default_inputs.get(key)
        assert list(sig.fields) == list(default_sig.fields)

        if kind == "ChainOfThought":
            # The fix step keeps what the extraction proposed: no context.
            assert sig.instructions == default_sig.instructions
        else:
            assert sig.instructions == _with_context(default_sig.instructions, CONTEXT)


def test_multiline_context_keeps_the_instructions_dedented():
    """DSPy dedents the inline instructions only while every continuation
    line shares their indent; an unindented context line would stop it."""
    default = capture_calls("es")["dspy_text"]
    with_context = capture_calls("es", context=MULTILINE_CONTEXT)["dspy_text"]
    for (_, default_sig, _), (_, sig, _) in zip(default, with_context):
        assert sig.instructions == _with_context(
            default_sig.instructions, MULTILINE_CONTEXT
        )


@pytest.mark.parametrize("language", ["en", "es"])
def test_litellm_context_follows_the_system_prompt(language):
    default = capture_calls(language)["litellm_text"]
    with_context = capture_calls(language, context=CONTEXT)["litellm_text"]
    for (_, default_kwargs, _), (_, kwargs, _) in zip(default, with_context):
        (default_system, default_user), (system, user) = (
            default_kwargs["input"],
            kwargs["input"],
        )
        assert system["content"] == f"{default_system['content'].rstrip()}\n\n{CONTEXT}"
        assert user == default_user
        assert kwargs["text"] == default_kwargs["text"]


def test_context_goes_before_the_profile_rules():
    default = capture_calls("es")["dspy_text"]
    with_both = capture_calls("es", "es-v2", context=CONTEXT)["dspy_text"]
    for (_, default_sig, _), (_, sig, _) in zip(default, with_both):
        rules = (PROMPTS / f"reglas_es_v2_{_step(default_sig)}.txt").read_text(
            encoding="utf-8"
        )
        assert sig.instructions == (
            f"{_with_context(default_sig.instructions, CONTEXT)}\n\n{rules.rstrip()}"
        )


def test_entity_signatures_with_context_are_cached_in_a_bounded_cache():
    assert _entities_sig(False, "es", None, CONTEXT) is _entities_sig(
        False, "es", None, CONTEXT
    )
    assert _entities_sig(False, "es", None, CONTEXT) is not _entities_sig(False, "es")
    assert _entities_sig.cache_info().maxsize == 128


# --------------------------------------------------------------------------
# KGGen forwards it only when asked
# --------------------------------------------------------------------------


def _offline_kggen(**kwargs):
    # A fake model name and key: constructing dspy.LM only stores them.
    return kg_gen_module.KGGen(model="openai/fake-model", api_key="test-key", **kwargs)


@pytest.fixture
def recorded_context(monkeypatch):
    """Stub both extraction steps and record the context each one receives."""
    seen = []

    def fake_entities(*args, **kwargs):
        seen.append(("entities", kwargs["context"]))
        return ["Ana", "Luis"]

    def fake_relations(*args, **kwargs):
        seen.append(("relations", kwargs["context"]))
        return [("Ana", "conocer a", "Luis")]

    monkeypatch.setattr(kg_gen_module, "get_entities", fake_entities)
    monkeypatch.setattr(kg_gen_module, "get_relations", fake_relations)
    return seen


def _generate(kg, **kwargs):
    kg.generate("Ana conoce a Luis.", context=CONTEXT, deduplication_method=None, **kwargs)


def test_kggen_does_not_forward_context_by_default(recorded_context):
    kg = _offline_kggen(language="es")
    assert kg.forward_context is False
    _generate(kg)
    assert recorded_context == [("entities", ""), ("relations", "")]


def test_generate_forwards_context_when_the_instance_asks(recorded_context):
    _generate(_offline_kggen(language="es", forward_context=True))
    assert recorded_context == [("entities", CONTEXT), ("relations", CONTEXT)]


@pytest.mark.parametrize("instance, call", [(False, True), (True, False)])
def test_per_call_flag_overrides_the_instance(recorded_context, instance, call):
    _generate(_offline_kggen(language="es", forward_context=instance), forward_context=call)
    expected = CONTEXT if call else ""
    assert recorded_context == [("entities", expected), ("relations", expected)]
