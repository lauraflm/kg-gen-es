import threading
import unicodedata
from typing import Optional

from kg_gen.models import Graph
from kg_gen.utils.text_normalization import TextNormalizer
from semhash import SemHash

# Optional: model2vec for custom SemHash encoder
try:
    from model2vec import StaticModel
except ImportError:
    StaticModel = None


def _representative_sort_key(text: str) -> tuple:
    """Preference order for the display form among variants of one item: case
    variants grouped by `TextNormalizer.fold()`, and originals that normalize
    to the same form.

    Sorting by (length, alphabetical) decides by codepoint order, which
    systematically favors the worst variant: uppercase sorts before lowercase
    ("PARIS" < "Paris") and ASCII before accented letters ("Garcia" <
    "García"), so accents were dropped and shouted forms won.

    Mixed capitalization and the higher count of diacritics come first;
    length and alphabetical order remain only as a deterministic tiebreak.
    Acronyms are unaffected: with no mixed-case variant to compare against
    ("ONU" vs "onu"), the tiebreak still picks the uppercase form.
    """
    has_upper = any(c.isupper() for c in text)
    has_lower = any(c.islower() for c in text)
    mixed_case = has_upper and has_lower

    nfkd = unicodedata.normalize("NFKD", text)
    diacritics = sum(1 for c in nfkd if unicodedata.combining(c))

    return (0 if mixed_case else 1, -diacritics, len(text), text)


class DeduplicateList:
    normalizer: TextNormalizer
    original_map: dict[str, str]
    items_map: dict[str, str]
    duplicates: dict[str, str]
    deduplicated: list[str]

    # Stats values
    total_items: int
    deduplicated_items: int
    duplicate_items: int
    reduction: float

    def __init__(
        self,
        threshold: float = 0.95,
        language: str = "en",
        semhash_model=None,
        item_type: str = "entity",
    ):
        self.threshold = threshold
        self.normalizer = TextNormalizer(language=language)
        self.semhash_model = semhash_model
        self.item_type = item_type
        self.original_map = {}
        self.items_map = {}
        self.duplicates = {}
        self.deduplicated = []

    def normalize(self, text: str) -> str:
        """
        Normalize a text.
        """
        return self.normalizer.normalize(text)

    def singularize(self, text: str) -> str:
        """
        Lemmatize text (language-aware). Entities are lemmatized on nouns
        only; edges/predicates also lemmatize verbs, since predicates are
        usually verb phrases whose conjugation should collapse for dedup.
        Kept as `singularize` for interface compatibility.
        """
        if self.item_type == "edge":
            return self.normalizer.lemmatize_predicate(text)
        return self.normalizer.lemmatize_nouns(text)

    def deduplicate(self, items: list[str]) -> list[str]:
        """
        Deduplicate a list of items using semantic hashing.
        Before deduplication, items are normalized and singularized.

        Args:
            items: List of items to deduplicate

        Returns:
            List of deduplicated items
        """
        self.total_items = len(items)

        # Normalize and singularize each string
        # Collapse case-only variants (e.g. "México" / "MÉXICO") to a single
        # canonical singular form before SemHash, so they merge
        # deterministically instead of depending on the similarity threshold.
        # Diacritics stay in the key: a merge made here ignores the threshold,
        # and "año" / "ano" or "papá" / "papa" are different words. Accent
        # omission ("Mexico" / "México") is left to SemHash and the LLM stage.
        # `items` arrives from a Graph `set`, whose iteration order depends on
        # PYTHONHASHSEED and therefore CHANGES between processes. Picking the
        # representative by arrival order -- last one wins in `items_map`,
        # first one wins in `fold_to_singular` -- makes two identical runs
        # export different forms ("contribuye a" vs "contribuyen a") for no
        # reason. Collect the candidates instead and choose with an explicit
        # rule, which also makes the result reproducible across processes.
        def _representative(candidates: set[str]) -> str:
            return min(candidates, key=_representative_sort_key)

        singular_by_item: dict[str, str] = {}
        fold_candidates: dict[str, set[str]] = {}
        for item in items:
            normalized = self.normalize(item)
            singular = self.singularize(normalized)
            singular_by_item[item] = singular
            fold_candidates.setdefault(self.normalizer.fold(singular), set()).add(
                singular
            )

        fold_to_singular = {k: _representative(v) for k, v in fold_candidates.items()}

        originals_by_form: dict[str, set[str]] = {}
        normalized_items = set()
        for item in items:
            singular = fold_to_singular[self.normalizer.fold(singular_by_item[item])]
            self.original_map[item] = singular
            originals_by_form.setdefault(singular, set()).add(item)
            normalized_items.add(singular)

        for singular, originals in originals_by_form.items():
            self.items_map[singular] = _representative(originals)

        # Deduplicate the normalized strings
        # Pass custom model if provided (e.g. multilingual potion)
        # `sorted`, not `list`: given two near-duplicates SemHash keeps the one
        # it saw first, so input order decides which survives. With a `set`
        # that order depends on PYTHONHASHSEED and changes between processes.
        semhash_kwargs = {"records": sorted(normalized_items)}
        if self.semhash_model is not None:
            semhash_kwargs["model"] = self.semhash_model

        semhash = SemHash.from_records(**semhash_kwargs, ann_backend="basic")
        deduplication_result = semhash.self_deduplicate(threshold=self.threshold)

        self.deduplicated_items = len(deduplication_result.selected)
        # Count what actually merged, not just what SemHash discarded: the
        # case-variant collapsing happens BEFORE SemHash is called, so those
        # items never show up in `filtered`. Using len(filtered) reported 0
        # duplicates in cases where a merge had in fact happened.
        self.duplicate_items = self.total_items - self.deduplicated_items
        self.reduction = (
            (self.duplicate_items / self.total_items) * 100 if self.total_items else 0.0
        )

        # Map back to original strings
        for duplicate in deduplication_result.filtered:
            original = duplicate.record
            # duplicate.duplicates is a list of (record, score) tuples
            if duplicate.duplicates:
                duplicate_value = duplicate.duplicates[0][0]
                self.items_map[original] = self.items_map[duplicate_value]
                if original not in self.duplicates:
                    self.duplicates[original] = duplicate_value

        self.deduplicated = deduplication_result.selected

    def stats(self) -> str:
        return f"Total items: {self.total_items}; Deduplicated items: {self.deduplicated_items}; Duplicate items: {self.duplicate_items}; Reduction: {self.reduction:.1f}"


# ---- Module-level cache for Model2Vec models ----
_M2V_CACHE: dict[str, object] = {}
_M2V_LOCK = threading.Lock()

# What SemHash.from_records loads when it is given no model.
_SEMHASH_DEFAULT_MODEL = "minishlab/potion-base-8M"


def _load_semhash_model(model_name: Optional[str] = None):
    """Load and cache a Model2Vec model for SemHash.

    `None` resolves to SemHash's own default rather than being passed through.
    Left to SemHash, the default model is reloaded on every `from_records`
    call, and with model2vec >= 0.8 each load opens tqdm progress bars that
    race when graphs are deduplicated from several threads at once
    (`AttributeError: type object 'tqdm' has no attribute '_lock'`). Loading it
    once here gives the same model, so the same result, without the race.
    """
    model_name = model_name or _SEMHASH_DEFAULT_MODEL
    with _M2V_LOCK:
        if model_name not in _M2V_CACHE:
            if StaticModel is None:
                raise ImportError(
                    "model2vec is required for custom SemHash models. "
                    "Install with: pip install model2vec"
                )
            _M2V_CACHE[model_name] = StaticModel.from_pretrained(model_name)
        return _M2V_CACHE[model_name]


def run_semhash_deduplication(
    graph: Graph,
    similarity_threshold: float = 0.95,
    language: str = "en",
    semhash_model_name: Optional[str] = None,
) -> Graph:
    """
    Deduplicate the graph.

    Args:
        graph: Input knowledge graph.
        similarity_threshold: SemHash cosine similarity threshold.
        language: Language code for text normalization ("en", "es", etc.).
        semhash_model_name: HuggingFace model name for SemHash encoder.
            If None, uses SemHash's default (potion-base-8M), loaded once.
            For multilingual: "minishlab/potion-multilingual-128M".
    """
    semhash_model = _load_semhash_model(semhash_model_name)

    # Deduplicate each graph components
    entities_dedup = DeduplicateList(
        similarity_threshold,
        language=language,
        semhash_model=semhash_model,
        item_type="entity",
    )
    entities_dedup.deduplicate(graph.entities)
    edges_dedup = DeduplicateList(
        similarity_threshold,
        language=language,
        semhash_model=semhash_model,
        item_type="edge",
    )
    edges_dedup.deduplicate(graph.edges)

    def _get_relation(relation: list[str]) -> list[str]:
        """
        Get the transformed relation.
        """
        # Handle case where entity might not be in original_map due to normalization
        first_entity_original = relation[0]
        if first_entity_original in entities_dedup.original_map:
            first_entity = entities_dedup.items_map[
                entities_dedup.original_map[first_entity_original]
            ]
        else:
            # If not found, use the original entity (it might have been normalized differently)
            first_entity = first_entity_original

        second_entity_original = relation[2]
        if second_entity_original in entities_dedup.original_map:
            second_entity = entities_dedup.items_map[
                entities_dedup.original_map[second_entity_original]
            ]
        else:
            # If not found, use the original entity
            second_entity = second_entity_original

        edge_original = relation[1]
        if edge_original in edges_dedup.original_map:
            edge = edges_dedup.items_map[edges_dedup.original_map[edge_original]]
        else:
            # If not found, use the original edge
            edge = edge_original

        return [first_entity, edge, second_entity]

    # Deduplicate the graph
    new_entities = [
        entities_dedup.items_map[item] for item in entities_dedup.deduplicated
    ]
    new_edges = [edges_dedup.items_map[item] for item in edges_dedup.deduplicated]
    new_relations = [_get_relation(relation) for relation in graph.relations]

    # Remove duplicate relations
    new_relations = list(set(tuple(relation) for relation in new_relations))

    # Update entity_metadata keys to match deduplicated entity names
    new_entity_metadata: dict[str, set[str]] | None = None
    if graph.entity_metadata:
        new_entity_metadata = {}
        for original_entity, metadata_set in graph.entity_metadata.items():
            if original_entity in entities_dedup.original_map:
                deduped_entity = entities_dedup.items_map[
                    entities_dedup.original_map[original_entity]
                ]
            else:
                deduped_entity = original_entity
            # Merge metadata sets when entities are deduplicated together
            if deduped_entity in new_entity_metadata:
                new_entity_metadata[deduped_entity].update(metadata_set)
            else:
                new_entity_metadata[deduped_entity] = metadata_set.copy()

    return Graph(
        entities=new_entities,
        edges=new_edges,
        relations=new_relations,
        entity_metadata=new_entity_metadata,
    )
