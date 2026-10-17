"""
Language selection helpers for prompt text.

Every user-facing string that reaches the LLM lives in a table keyed by
language code, and is selected with `pick()`. English is always the
fallback, so adding a language means adding keys to the existing tables --
never editing call sites.

    _GREETING = {"en": "Hello", "es": "Hola"}
    pick(_GREETING, "es")  # -> "Hola"
    pick(_GREETING, "fr")  # -> "Hello"  (no French variant yet)
"""

from typing import Any

# text_normalization imports nothing from kg_gen, so this adds no cycle.
from kg_gen.utils.text_normalization import require_spacy_model

# Languages with a full set of translated prompts. `language` values outside
# this set still work -- `pick()` falls back to English -- but callers that
# need to warn or fail early (e.g. KGGen validating its constructor arg)
# check against this list.
SUPPORTED_LANGUAGES = ("en", "es")

DEFAULT_LANGUAGE = "en"


def validate_language(language: str) -> str:
    """Raise unless `language` is fully supported and usable here.

    Raises ValueError for a code outside SUPPORTED_LANGUAGES, and OSError,
    with the install command, when the spaCy model its deduplication needs
    is not installed. English needs none: it is singularized with inflect.

    Called when a language is first accepted (KGGen construction, or a
    per-call override) rather than where it is used, so an unusable code is
    rejected before any LLM calls are paid for. Without this the failure is
    both late and inconsistent: prompts quietly fall back to English via
    `pick()`, while `TextNormalizer` raises partway through deduplication,
    for an unknown code or a missing model alike. The model is only looked
    up here, not loaded; deduplication still loads it.
    """
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language {language!r}. "
            f"Supported: {', '.join(SUPPORTED_LANGUAGES)}. "
            "To add one, translate the prompt tables and register the code in "
            "SUPPORTED_LANGUAGES (see 'Adding a language' in the README)."
        )
    require_spacy_model(language)
    return language


def pick(table: dict[str, Any], language: str) -> Any:
    """Select the language variant from a table, falling back to English.

    Falling back rather than raising is deliberate: an unsupported language
    should degrade to English prompts, not break extraction. Note that
    `TextNormalizer` does raise for unknown languages, since there it has no
    sensible fallback -- English lemmatization rules would silently mangle
    the text.
    """
    return table.get(language, table[DEFAULT_LANGUAGE])
