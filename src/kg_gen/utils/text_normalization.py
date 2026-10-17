"""
Multilingual text normalization for KG deduplication.

Replaces the English-only `inflect` singularization with spaCy-based
lemmatization, supporting Spanish and other languages.

Usage:
    normalizer = TextNormalizer(language="es")
    normalizer.lemmatize_nouns("los libros")            # → "libro"
    normalizer.lemmatize_predicate("fue fundada por")   # → "ser fundar por"

Only the parts of speech each method targets are rewritten. Entity
lemmatization touches nouns and drops a leading article, so a trailing
adjective keeps its inflection:

    normalizer.lemmatize_nouns("las relaciones familiares")
    # → "relación familiares"   (not "la relación familiar")

That is enough for deduplication, where both sides of a comparison get the
same treatment, but it means the output is not well-formed Spanish and
should not be shown to users.
"""

import threading
import unicodedata
from pathlib import Path

import spacy

_NLP_CACHE: dict[str, spacy.language.Language] = {}
_NLP_LOCK = threading.Lock()

# Map language codes to spaCy model names.
# Add more as needed; small models suffice for lemmatization.
_SPACY_MODELS: dict[str, str] = {
    "es": "es_core_news_md",
    "en": "en_core_web_sm",
    # Add more languages here, e.g.:
    # "fr": "fr_core_news_sm",
    # "pt": "pt_core_news_sm",
    # "it": "it_core_news_sm",      

}


def _spacy_model_name(language: str) -> str:
    """Return the spaCy model configured for the given language code."""
    model_name = _SPACY_MODELS.get(language)
    if model_name is None:
        raise ValueError(
            f"No spaCy model configured for language '{language}'. "
            f"Supported: {list(_SPACY_MODELS.keys())}. "
            f"Add the model name to _SPACY_MODELS in text_normalization.py."
        )
    return model_name


def _model_not_installed(model_name: str) -> OSError:
    return OSError(
        f"spaCy model '{model_name}' not found. "
        f"Install it with: python -m spacy download {model_name}"
    )


def require_spacy_model(language: str) -> None:
    """Raise unless the spaCy model `TextNormalizer` needs is installed.

    `TextNormalizer` loads the model only at deduplication, after every
    extraction call has been made, so a missing model found there fails once
    the LLM work is already paid for. This runs the same lookup up front, for
    `validate_language` to call when a language is accepted.

    It only looks the model up, the way `spacy.load` resolves a name: an
    installed package, else a path to a model directory. Checking the
    package alone would reject a configured path that loads fine. No model
    is loaded, so the check is cheap enough to repeat on every call.

    Raises:
        ValueError: if no model is configured for `language`.
        OSError: if the configured model is not installed.
    """
    # English is singularized with inflect and never loads a spaCy model
    # (see TextNormalizer), so en_core_web_sm must not become a requirement.
    if language == "en":
        return
    model_name = _spacy_model_name(language)
    if not (spacy.util.is_package(model_name) or Path(model_name).exists()):
        raise _model_not_installed(model_name)


def _get_nlp(language: str) -> spacy.language.Language:
    """Load and cache a spaCy model for the given language code.

    The lock covers the load itself. Without it, deduplications running in
    parallel threads all miss the empty cache and each call `spacy.load` for
    the same model, repeating a slow load once per thread.
    """
    with _NLP_LOCK:
        if language not in _NLP_CACHE:
            model_name = _spacy_model_name(language)
            try:
                _NLP_CACHE[language] = spacy.load(
                    model_name, disable=["ner", "parser", "textcat"]
                )
            except OSError:
                raise _model_not_installed(model_name)
        return _NLP_CACHE[language]


class TextNormalizer:
    """
    Language-aware text normalizer for KG entity/edge deduplication.

    For English (language="en"), falls back to the original `inflect`-based
    singularization to preserve backward compatibility.

    For Spanish and other supported languages, uses spaCy lemmatization
    restricted to nouns (NOUN) so that proper nouns, verbs in edge labels,
    and adjectives are left untouched — matching the intent of the original
    `inflect.singular_noun` which only transforms plural nouns.
    """

    def __init__(self, language: str = "en"):
        self.language = language
        self._inflect_engine = None

        if language == "en":
            import inflect
            self._inflect_engine = inflect.engine()
        else:
            # Eagerly load spaCy model to fail fast if missing
            _get_nlp(language)

    def normalize(self, text: str) -> str:
        """Unicode NFKC normalization (language-agnostic)."""
        return unicodedata.normalize("NFKC", text)

    def fold(self, text: str) -> str:
        """
        Case-insensitive comparison key (language-agnostic).

        Used only to group case variants (e.g. "México" / "MÉXICO") before
        dedup, so they collapse deterministically instead of relying on the
        SemHash similarity threshold to catch them. Never used for display
        text.

        Diacritics are kept. In Spanish they distinguish words ("año" /
        "ano", "papá" / "papa", "él" / "el"), and the same mark in the same
        position also separates two spellings of one entity ("Perú" /
        "Peru"), so no character-level rule can block the first kind of
        merge and keep the second. A grouping made here is final whatever
        the threshold, so accent omission is left to the similarity-based
        stages.
        """
        # Decompose first so composed and decomposed spellings of a letter
        # ("á" vs "a" + U+0301) compare equal now that marks are part of the
        # key, and again after casefold, which can itself emit a sequence
        # that is no longer normalized. NFKD rather than NFD keeps the
        # compatibility folding ("ﬁ" -> "fi") the key always had.
        decomposed = unicodedata.normalize("NFKD", text)
        return unicodedata.normalize("NFKD", decomposed.casefold())

    def lemmatize_nouns(self, text: str) -> str:
        """
        Reduce plural nouns to their lemma form. Used for entities.

        - English: uses inflect.singular_noun (original kg-gen behavior).
        - Other languages: uses spaCy lemmatization on NOUN tokens only.

        Returns the processed text with other tokens unchanged.
        """
        if self.language == "en":
            return self._singularize_en(text)
        return self._lemmatize_spacy(
            text, pos_tags={"NOUN"}, strip_leading_det=True
        )

    def lemmatize_predicate(self, text: str) -> str:
        """
        Normalize predicate/edge text to its lemma form.

        Predicates in Spanish are usually verb phrases (e.g. "fundó",
        "fundaron", "fue fundada por"), so verb conjugation needs to be
        collapsed in addition to noun pluralization for dedup keys to
        match across tense/number variants of the same relation.

        - English: uses inflect.singular_noun (original kg-gen behavior);
          English predicates don't vary by verb conjugation the way
          Spanish ones do, so no additional handling is needed here.
        - Other languages: uses spaCy lemmatization on NOUN, VERB, and
          AUX tokens.
        """
        if self.language == "en":
            return self._singularize_en(text)
        # Predicates are never proper nouns, but spaCy's Spanish tagger
        # mistags a capitalized leading verb as PROPN (e.g. "Fundó"),
        # which skips lemmatization. Lowercasing first avoids that, and
        # is safe here since predicate casing carries no meaning.
        return self._lemmatize_spacy(text.lower(), pos_tags={"NOUN", "VERB", "AUX"})

    # --- Private helpers ---

    def _singularize_en(self, text: str) -> str:
        """Original inflect-based English singularization."""
        tokens = []
        for tok in text.split():
            sing = self._inflect_engine.singular_noun(tok)
            tokens.append(sing if isinstance(sing, str) and sing else tok)
        return " ".join(tokens).strip()

    def _lemmatize_spacy(
        self, text: str, pos_tags: set[str], strip_leading_det: bool = False
    ) -> str:
        """spaCy-based lemmatization restricted to the given POS tags."""
        nlp = _get_nlp(self.language)
        doc = nlp(text)

        tokens = list(doc)
        if (
            strip_leading_det
            and tokens
            and tokens[0].pos_ == "DET"
            # Only drop the article when the head noun is a common noun
            # ("la ciudad" -> "ciudad"). Skip it when the head is a proper
            # noun, since the article is then part of the name itself
            # ("El Salvador", "La Habana") and stripping it would change
            # the entity's meaning.
            and len(tokens) > 1
            and tokens[1].pos_ != "PROPN"
        ):
            tokens = tokens[1:]

        out = []
        for token in tokens:
            # Only lemmatize the requested POS tags (e.g. common nouns),
            # not proper nouns (PROPN), to avoid mangling entity names
            # like "Carlos" → "carlo"
            if token.pos_ in pos_tags:
                out.append(token.lemma_)
            else:
                out.append(token.text)
        return " ".join(out).strip()
