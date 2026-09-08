from typing import List, Optional
from pathlib import Path
import dspy
import litellm
from pydantic import BaseModel

from kg_gen.utils.i18n import pick as _pick

_TEXT_ENTITIES_INSTRUCTIONS = {
    "en": """Extract key entities from the source text. Extracted entities are subjects or objects.
    This is for an extraction task, please be THOROUGH and accurate to the reference text.""",
    "es": """Extrae las entidades clave del texto fuente. Las entidades extraídas son sujetos u objetos.
    Esta es una tarea de extracción: sé EXHAUSTIVO y preciso respecto al texto de referencia.""",
}

_CONVERSATION_ENTITIES_INSTRUCTIONS = {
    "en": """Extract key entities from the conversation Extracted entities are subjects or objects.
    Consider both explicit entities and participants in the conversation.
    This is for an extraction task, please be THOROUGH and accurate.""",
    "es": """Extrae las entidades clave de la conversación. Las entidades extraídas son sujetos u objetos.
    Considera tanto las entidades explícitas como los participantes de la conversación.
    Esta es una tarea de extracción: sé EXHAUSTIVO y preciso.""",
}

_ENTITIES_OUTPUT_DESC = {
    "en": "THOROUGH list of key entities",
    "es": "Lista EXHAUSTIVA de entidades clave",
}

_ENTITIES_PROMPT_FILES = {"es": "entidades.txt", "en": "entities.txt"}

# User-turn wrapper for the litellm path. The XML-ish tags are structural and
# stay untranslated; only the prose around them varies by language.
# Filled with str.replace rather than str.format: the source text is
# arbitrary user input and a stray "{" in it would blow up .format().
_ENTITIES_USER_PROMPT = {
    "en": """
Here is the text to extract entities from:

<article>
{input_data}
</article>
    """,
    "es": """
Este es el texto del que debes extraer entidades:

<article>
{input_data}
</article>
    """,
}


def _entities_sig(is_conversation: bool, language: str = "en") -> dspy.Signature:
    """Build the entity-extraction signature for the requested language.

    Replaces upstream's two fixed signature classes (TextEntities /
    ConversationEntities); the fields are identical, only the instructions
    and field descriptions vary by language.
    """
    instructions = _pick(
        _CONVERSATION_ENTITIES_INSTRUCTIONS
        if is_conversation
        else _TEXT_ENTITIES_INSTRUCTIONS,
        language,
    )

    class EntitiesSignature(dspy.Signature):
        __doc__ = instructions

        source_text: str = dspy.InputField()
        entities: list[str] = dspy.OutputField(
            desc=_pick(_ENTITIES_OUTPUT_DESC, language)
        )

    return EntitiesSignature


class EntitiesResponse(BaseModel):
    """Structured response for entity extraction."""

    entities: List[str]


def _load_entities_prompt(language: str = "en") -> str:
    """Load the entities prompt template from file."""
    filename = _ENTITIES_PROMPT_FILES.get(language, "entities.txt")
    prompt_path = Path(__file__).parent.parent / "prompts" / filename
    return prompt_path.read_text()


def _get_entities_litellm(
    input_data: str,
    model: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
    language: str = "en",
) -> List[str]:
    prompt_template = _load_entities_prompt(language)
    user_prompt = _pick(_ENTITIES_USER_PROMPT, language).replace(
        "{input_data}", input_data
    )

    # Build schema with additionalProperties: false (required by OpenAI)
    schema = EntitiesResponse.model_json_schema()
    schema["additionalProperties"] = False

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
                "name": "entities_response",
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
    # print(response.model_dump_json(indent=2))
    parsed = EntitiesResponse.model_validate_json(response.output[-1].content[0].text)
    return parsed.entities


def get_entities(
    input_data: str,
    is_conversation: bool = False,
    use_litellm_prompt: bool = False,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
    language: str = "en",
) -> List[str]:
    if use_litellm_prompt and not is_conversation:
        return _get_entities_litellm(
            input_data,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            language=language,
        )

    EntitiesSignature = _entities_sig(is_conversation, language)
    extract = dspy.Predict(EntitiesSignature)
    result = extract(source_text=input_data)
    return result.entities
