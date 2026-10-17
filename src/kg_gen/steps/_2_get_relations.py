from functools import lru_cache
from typing import List, Tuple, Optional, Literal, Type
from pathlib import Path
import json
import dspy
import litellm
from pydantic import BaseModel, create_model, ValidationError

from kg_gen.utils.extraction_stats import ExtractionStats
from kg_gen.utils.i18n import pick as _pick


def parse_relations_response(
    raw_json: str,
    entities: List[str],
    response_model: Optional[Type[BaseModel]] = None,
    stats: Optional[ExtractionStats] = None,
) -> List[Tuple[str, str, str]]:
    """
    Parse a relations JSON response with graceful fallback.

    First attempts strict Pydantic validation. If that fails (e.g., due to
    EntityLiteral validation), falls back to raw JSON parsing and filters
    out items with invalid subject/object.

    Args:
        raw_json: The raw JSON string from the LLM response
        entities: List of valid entity strings
        response_model: Optional Pydantic model for strict validation
        stats: Optional counter collector; recording never changes the result

    Returns:
        List of (subject, predicate, object) tuples with valid entities
    """
    entities_set = set(entities)

    # Try strict Pydantic validation first if model provided
    if response_model is not None:
        try:
            parsed = response_model.model_validate_json(raw_json)
            if stats is not None:
                stats.record_relations(parsed.relations, entities)
            return [(r.subject, r.predicate, r.object) for r in parsed.relations]
        except ValidationError:
            # Strict validation is what enforces the entity Literal, so this
            # is the signal that some endpoint was not an exact match.
            if stats is not None:
                stats.increment("strict_validation_failed")
            pass  # Fall through to JSON parsing

    # Fallback: parse as raw JSON and filter
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        # The whole response is lost; how many triples it held is unknowable.
        if stats is not None:
            stats.increment("responses_json_invalid")
        return []

    # Handle both {"relations": [...]} and direct list formats
    items = data.get("relations", data) if isinstance(data, dict) else data

    if not isinstance(items, list):
        return []

    relations = []
    returned = []
    for item in items:
        if not isinstance(item, dict):
            continue

        subject = item.get("subject")
        predicate = item.get("predicate")
        obj = item.get("object")

        # Skip if missing required fields
        if not all([subject, predicate, obj]):
            if stats is not None:
                stats.increment("items_missing_field")
            continue

        # Count before filtering, so the endpoint-mismatch rate is measured
        # against what the model actually proposed rather than what survived.
        returned.append((subject, predicate, obj))

        # Skip if subject or object not in valid entities
        if subject not in entities_set or obj not in entities_set:
            continue

        relations.append((subject, predicate, obj))

    if stats is not None:
        stats.record_relations(returned, entities)

    return relations


_RELATIONS_PROMPT_FILES = {"es": "relaciones.txt", "en": "relations.txt"}

# Inline instructions for the default dspy.Predict path (as opposed to the
# litellm path, which loads the more elaborate prompts/relations.txt or
# relaciones.txt files). Kept language-selectable independent of the
# TextNormalizer/dedup lemmatization pipeline, which is unrelated.
_TEXT_RELATIONS_INSTRUCTIONS = {
    "en": """Extract subject-predicate-object triples from the source text.
      Subject and object must be from entities list. Entities provided were previously extracted from the same source text.
      This is for an extraction task, please be thorough, accurate, and faithful to the reference text.""",
    "es": """Extrae tripletas sujeto-predicado-objeto del texto fuente.
      El sujeto y el objeto deben ser una entidad de la lista de entidades. Las entidades fueron previamente extraídas de este mismo texto fuente.
      Esta es una tarea de extracción: sé exhaustivo, preciso y fiel al texto de referencia.""",
}

_CONVERSATION_RELATIONS_INSTRUCTIONS = {
    "en": """Extract subject-predicate-object triples from the conversation, including:
      1. Relations between concepts discussed
      2. Relations between speakers and concepts (e.g. user asks about X)
      3. Relations between speakers (e.g. assistant responds to user)
      Subject and object must be from entities list. Entities provided were previously extracted from the same source text.
      This is for an extraction task, please be thorough, accurate, and faithful to the reference text.""",
    "es": """Extrae tripletas sujeto-predicado-objeto de la conversación, incluyendo:
      1. Relaciones entre los conceptos discutidos
      2. Relaciones entre los interlocutores y los conceptos (ej. el usuario pregunta sobre X)
      3. Relaciones entre los interlocutores (ej. el asistente responde al usuario)
      El sujeto y el objeto deben ser una entidad de la lista de entidades. Las entidades fueron previamente extraídas de este mismo texto fuente.
      Esta es una tarea de extracción: sé exhaustivo, preciso y fiel al texto de referencia.""",
}

_FIXED_RELATIONS_INSTRUCTIONS = {
    "en": "Fix the relations so that every subject and object of the relations are exact matches to an entity. Keep the predicate the same. The meaning of every relation should stay faithful to the reference text. If you cannot maintain the meaning of the original relation relative to the source text, then do not return it.",
    "es": "Corrige las relaciones para que el sujeto y el objeto de cada una coincidan exactamente con una entidad. Mantén el mismo predicado. El significado de cada relación debe seguir siendo fiel al texto de referencia. Si no puedes conservar el significado de la relación original respecto al texto fuente, no la devuelvas.",
}

# User-turn wrapper for the litellm path. The XML-ish tags are structural and
# stay untranslated; only the prose around them varies by language.
# Filled with str.replace rather than str.format: the source text is
# arbitrary user input and a stray "{" in it would blow up .format().
_RELATIONS_USER_PROMPT = {
    "en": """
Here is the list of entities that were previously extracted from the source text:

<entities>
{entities_str}
</entities>

Here is the source text to analyze:

<text>
{input_data}
</text>
    """,
    "es": """
Esta es la lista de entidades extraídas previamente del texto fuente:

<entities>
{entities_str}
</entities>

Este es el texto fuente que debes analizar:

<text>
{input_data}
</text>
    """,
}

_RELATIONS_OUTPUT_DESC = {
    "en": "List of subject-predicate-object tuples. Be thorough.",
    "es": "Lista de tripletas sujeto-predicado-objeto. Sé exhaustivo.",
}

_CONVERSATION_RELATIONS_OUTPUT_DESC = {
    "en": "List of subject-predicate-object tuples where subject and object are exact matches to items in entities list. Be thorough",
    "es": "Lista de tripletas sujeto-predicado-objeto donde el sujeto y el objeto coinciden exactamente con elementos de la lista de entidades. Sé exhaustivo.",
}

_RELATION_DOC = {
    "en": "Knowledge graph subject-predicate-object tuple.",
    "es": "Tripleta sujeto-predicado-objeto de un grafo de conocimiento.",
}

_FALLBACK_RELATION_DOC = {
    "en": "Knowledge graph subject-predicate-object tuple. Subject and object entities must be one of: {entities_str}",
    "es": "Tripleta sujeto-predicado-objeto de un grafo de conocimiento. El sujeto y el objeto deben ser una de estas entidades: {entities_str}",
}

_RELATION_FIELD_DESCS = {
    "en": {
        "subject": "Subject entity",
        "predicate": "Predicate",
        "object": "Object entity",
    },
    "es": {
        "subject": "Entidad sujeto",
        "predicate": "Predicado",
        "object": "Entidad objeto",
    },
}

# These are few-shot examples embedded in every prompt, so the predicate
# example is what tells the model which language to emit predicates in.
# An English "is brother of" pulls the output toward English regardless of
# how the instructions are worded.
_RELATION_EXAMPLES = {
    "en": {
        "subject": ["Kevin"],
        "predicate": ["is brother of"],
        "object": ["Vicky"],
    },
    "es": {
        "subject": ["Carlos"],
        "predicate": ["es hermano de"],
        "object": ["Lucía"],
    },
}


# Cached so get_relations does not rebuild the model for every chunk; its
# callers pass only `language` here. The fallback path's `doc` embeds that
# chunk's entity list, so fallback_extraction_sig calls `__wrapped__` instead:
# caching it would keep one class per chunk alive for nothing.
@lru_cache(maxsize=None)
def _relation_model(language: str, doc: Optional[str] = None) -> Type[BaseModel]:
    """Build the subject-predicate-object model with language-appropriate
    field descriptions and few-shot examples.

    `doc` overrides the class docstring; the fallback path uses it to inline
    the allowed entity list.
    """
    descs = _pick(_RELATION_FIELD_DESCS, language)
    examples = _pick(_RELATION_EXAMPLES, language)

    class Relation(BaseModel):
        __doc__ = doc if doc is not None else _pick(_RELATION_DOC, language)

        subject: str = dspy.InputField(
            desc=descs["subject"], examples=examples["subject"]
        )
        predicate: str = dspy.InputField(
            desc=descs["predicate"], examples=examples["predicate"]
        )
        object: str = dspy.InputField(desc=descs["object"], examples=examples["object"])

    return Relation


def _load_relations_prompt(language: str = "en") -> str:
    """Load the relations prompt template from file."""
    filename = _pick(_RELATIONS_PROMPT_FILES, language)
    prompt_path = Path(__file__).parent.parent / "prompts" / filename
    return prompt_path.read_text()


def _create_relations_model(entities: List[str]):
    """Dynamically create Pydantic models with entity literals for subject/object."""
    # Create a Literal type from the entities list
    EntityLiteral = Literal[tuple(entities)]  # type: ignore

    # Create RelationItem with constrained subject/object
    RelationItem = create_model(
        "RelationItem",
        subject=(EntityLiteral, ...),
        predicate=(str, ...),
        object=(EntityLiteral, ...),
    )

    # Create RelationsResponse containing list of RelationItem
    RelationsResponse = create_model(
        "RelationsResponse",
        relations=(List[RelationItem], ...),
    )

    return RelationItem, RelationsResponse


def _get_relations_litellm(
    input_data: str,
    entities: List[str],
    model: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
    language: str = "en",
    stats: Optional[ExtractionStats] = None,
) -> List[Tuple[str, str, str]]:
    prompt_template = _load_relations_prompt(language)
    entities_str = "\n".join(f"- {e}" for e in entities)
    user_prompt = (
        _pick(_RELATIONS_USER_PROMPT, language)
        .replace("{entities_str}", entities_str)
        .replace("{input_data}", input_data)
    )

    # Create dynamic model with entity constraints
    _, RelationsResponse = _create_relations_model(entities)

    # Build schema with additionalProperties: false (required by OpenAI)
    schema = RelationsResponse.model_json_schema()
    schema["additionalProperties"] = False
    # Also need to set additionalProperties on nested objects
    if "$defs" in schema:
        for def_schema in schema["$defs"].values():
            if def_schema.get("type") == "object":
                def_schema["additionalProperties"] = False

    kwargs = {
        "model": model,
        "input": [
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "relations_response",
                "schema": schema,
                "strict": True,
            }
        },
    }

    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    response = litellm.responses(**kwargs)
    raw_json = response.output[-1].content[0].text
    return parse_relations_response(raw_json, entities, RelationsResponse, stats=stats)


# Cached for the same reason as _relation_model; the key includes the
# Relation class itself, which is hashable by identity and, from
# get_relations, the cached one. Bounded rather than unbounded because
# `context` is free text from the caller. The fallback bypasses this cache
# too, since its Relation class is new for every chunk.
@lru_cache(maxsize=128)
def extraction_sig(
    Relation: BaseModel,
    is_conversation: bool,
    context: str = "",
    language: str = "en",
) -> dspy.Signature:
    if not is_conversation:
        instructions = _pick(_TEXT_RELATIONS_INSTRUCTIONS, language)

        class ExtractTextRelations(dspy.Signature):
            __doc__ = f"{instructions} {context}"

            source_text: str = dspy.InputField()
            entities: list[str] = dspy.InputField()
            relations: list[Relation] = dspy.OutputField(
                desc=_pick(_RELATIONS_OUTPUT_DESC, language)
            )

        return ExtractTextRelations
    else:
        instructions = _pick(_CONVERSATION_RELATIONS_INSTRUCTIONS, language)

        class ExtractConversationRelations(dspy.Signature):
            __doc__ = f"{instructions} {context}"

            source_text: str = dspy.InputField()
            entities: list[str] = dspy.InputField()
            relations: list[Relation] = dspy.OutputField(
                desc=_pick(_CONVERSATION_RELATIONS_OUTPUT_DESC, language)
            )

        return ExtractConversationRelations


def fallback_extraction_sig(
    entities, is_conversation, context: str = "", language: str = "en"
) -> dspy.Signature:
    """This fallback extraction does not strictly type the subject and object strings."""

    entities_str = "\n- ".join(entities)

    # TODO: should use literal's here instead.
    # `__wrapped__` builds both classes without the caches: the doc embeds
    # this chunk's entities, so a cached entry would never be hit again.
    Relation = _relation_model.__wrapped__(
        language,
        doc=_pick(_FALLBACK_RELATION_DOC, language).format(entities_str=entities_str),
    )

    return Relation, extraction_sig.__wrapped__(
        Relation, is_conversation, context, language
    )


def _filter_entities(entities: List[str]) -> List[str]:
    """Filter out entities that contain backslashes."""
    return [e for e in entities if '"' not in e] # not received by oai api


def get_relations(
    input_data: str,
    entities: list[str],
    is_conversation: bool = False,
    context: str = "",
    use_litellm_prompt: bool = False,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
    language: str = "en",
    stats: Optional[ExtractionStats] = None,
) -> List[Tuple[str, str, str]]:
    # Filter out entities containing backslashes
    entities = _filter_entities(entities)

    if use_litellm_prompt and not is_conversation:
        if stats is not None:
            stats.record_route("litellm")
        return _get_relations_litellm(
            input_data,
            entities,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            language=language,
            stats=stats,
        )

    if stats is not None:
        stats.record_route("dspy")

    Relation = _relation_model(language)

    ExtractRelations = extraction_sig(Relation, is_conversation, context, language)

    try:
        extract = dspy.Predict(ExtractRelations)
        result = extract(source_text=input_data, entities=entities)
        # Note that this route returns the triples UNFILTERED: `Relation`
        # declares subject and object as plain `str`, so nothing enforces
        # that they are extracted entities. Any endpoint counted as missing
        # here reaches the Graph as a relation pointing at an entity the
        # graph does not contain.
        if stats is not None:
            stats.record_relations(result.relations, entities)
        return [(r.subject, r.predicate, r.object) for r in result.relations]

    except Exception as _:
        # print("get_relations: fallback extraction")
        if stats is not None:
            stats.increment("fallback_invoked")
        Relation, ExtractRelations = fallback_extraction_sig(
            entities, is_conversation, context, language
        )
        extract = dspy.Predict(ExtractRelations)
        result = extract(source_text=input_data, entities=entities)

        class FixedRelations(dspy.Signature):
            __doc__ = _pick(_FIXED_RELATIONS_INSTRUCTIONS, language)

            source_text: str = dspy.InputField()
            entities: list[str] = dspy.InputField()
            relations: list[Relation] = dspy.InputField()
            fixed_relations: list[Relation] = dspy.OutputField()

        fix = dspy.ChainOfThought(FixedRelations)

        fix_res = fix(
            source_text=input_data, entities=entities, relations=result.relations
        )

        good_relations = []
        for rel in fix_res.fixed_relations:
            if rel.subject in entities and rel.object in entities:
                good_relations.append(rel)
        if stats is not None:
            # Counted from the fix step's output before the endpoint filter:
            # it is this route's final proposal, as `result.relations` is the
            # primary route's, so the two routes count at the same point. The
            # fallback signature's own output is not counted; it is an
            # intermediate the fix step rewrites, and counting both would
            # count each triple twice. With the filtered list taken from the
            # same input, fallback_dropped is exactly this call's share of
            # endpoint_missing_either.
            stats.record_relations(fix_res.fixed_relations, entities)
            stats.increment(
                "fallback_dropped", len(fix_res.fixed_relations) - len(good_relations)
            )
        return [(r.subject, r.predicate, r.object) for r in good_relations]
