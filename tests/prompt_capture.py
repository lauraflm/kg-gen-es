"""Capture every prompt the extraction steps send, without calling an LLM.

`capture_prompts(language)` runs get_entities and get_relations with
dspy.Predict, dspy.ChainOfThought and litellm.responses replaced by
recorders, and renders each recorded DSPy call with the ChatAdapter, the way
DSPy sends it. It covers every route: DSPy on text and on a conversation, the
DSPy fallback with its fix step, and the litellm route.

With `linguistic_profile=None` it only passes arguments that v1.1 already
had, so the same file regenerates the v1.1 snapshot from a checkout of the
tag (needed only if DSPy is upgraded and changes how it renders prompts):

    git worktree add ../kg-gen-es-v1.1 v1.1
    cp tests/prompt_capture.py ../kg-gen-es-v1.1/tests/
    cd ../kg-gen-es-v1.1 && python -m tests.prompt_capture \\
        <this checkout>/tests/snapshots/prompts_v1_1.json
"""

import json
import sys
import typing
from contextlib import contextmanager
from types import SimpleNamespace

import dspy
import litellm
from dspy.adapters.chat_adapter import ChatAdapter

from src.kg_gen.steps import _1_get_entities as entities_step
from src.kg_gen.steps import _2_get_relations as relations_step

SOURCE_TEXT = "Ana Ríos fundó el Instituto Sur. Luego lo dirigió."
CONVERSATION = "user: ¿Quién fundó el Instituto Sur?\nassistant: Lo fundó Ana Ríos."
ENTITIES = ["Ana Ríos", "Instituto Sur"]
TRIPLE = {"subject": "Ana Ríos", "predicate": "fundar", "object": "Instituto Sur"}

SCENARIOS = {
    "dspy_text": {"input_data": SOURCE_TEXT, "conversation": False, "litellm": False, "fail": False},
    # The primary route raises, so get_relations runs the fallback and the fix step.
    "dspy_text_fallback": {"input_data": SOURCE_TEXT, "conversation": False, "litellm": False, "fail": True},
    "dspy_conversation": {"input_data": CONVERSATION, "conversation": True, "litellm": False, "fail": False},
    "litellm_text": {"input_data": SOURCE_TEXT, "conversation": False, "litellm": True, "fail": False},
}


def _relations_for(signature, field):
    relation = typing.get_args(signature.output_fields[field].annotation)[0]
    return [relation(**TRIPLE)]


@contextmanager
def recording(calls, fail_primary_relations=False):
    """Replace the LLM entry points with recorders that append to `calls`."""
    real_predict, real_cot, real_responses = dspy.Predict, dspy.ChainOfThought, litellm.responses
    failed = []

    def predict(signature):
        def run(**inputs):
            calls.append(("Predict", signature, inputs))
            if "entities" in signature.output_fields:
                return SimpleNamespace(entities=list(ENTITIES))
            if fail_primary_relations and not failed:
                failed.append(True)
                raise RuntimeError("primary route fails on purpose")
            return SimpleNamespace(relations=_relations_for(signature, "relations"))

        return run

    def chain_of_thought(signature):
        # ChainOfThought adds a reasoning field; record the signature it would
        # really send. Building it calls dspy.Predict, so the real one is
        # restored for that.
        dspy.Predict = real_predict
        try:
            extended = real_cot(signature).predict.signature
        finally:
            dspy.Predict = predict

        def run(**inputs):
            calls.append(("ChainOfThought", extended, inputs))
            return SimpleNamespace(fixed_relations=_relations_for(signature, "fixed_relations"))

        return run

    def responses(**kwargs):
        calls.append(("litellm", kwargs, None))
        if kwargs["text"]["format"]["name"] == "entities_response":
            body = {"entities": ENTITIES}
        else:
            body = {"relations": [TRIPLE]}
        content = SimpleNamespace(text=json.dumps(body, ensure_ascii=False))
        return SimpleNamespace(output=[SimpleNamespace(content=[content])])

    dspy.Predict, dspy.ChainOfThought, litellm.responses = predict, chain_of_thought, responses
    try:
        yield
    finally:
        dspy.Predict, dspy.ChainOfThought, litellm.responses = real_predict, real_cot, real_responses


def render(call):
    """The prompt of one recorded call, as the model would receive it."""
    kind, payload, inputs = call
    if kind == "litellm":
        return {
            "call": kind,
            "messages": payload["input"],
            "schema": payload["text"]["format"]["schema"],
        }
    messages = ChatAdapter().format(payload, demos=[], inputs=inputs)
    return {
        "call": kind,
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
    }


def capture_calls(language, linguistic_profile=None):
    """The raw recorded calls of every scenario: (kind, signature, inputs)."""
    extra = {} if linguistic_profile is None else {"linguistic_profile": linguistic_profile}
    captured = {}
    for name, scenario in SCENARIOS.items():
        calls = []
        with recording(calls, fail_primary_relations=scenario["fail"]):
            entities = entities_step.get_entities(
                scenario["input_data"],
                scenario["conversation"],
                use_litellm_prompt=scenario["litellm"],
                model="openai/fake-model",
                language=language,
                **extra,
            )
            relations_step.get_relations(
                scenario["input_data"],
                entities,
                is_conversation=scenario["conversation"],
                use_litellm_prompt=scenario["litellm"],
                model="openai/fake-model",
                language=language,
                **extra,
            )
        captured[name] = calls
    return captured


def capture_prompts(language, linguistic_profile=None):
    """Every prompt of every scenario, rendered."""
    return {
        name: [render(call) for call in calls]
        for name, calls in capture_calls(language, linguistic_profile).items()
    }


if __name__ == "__main__":
    snapshot = {language: capture_prompts(language) for language in ("en", "es")}
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=1)
        f.write("\n")
