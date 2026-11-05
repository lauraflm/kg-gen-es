from typing import Optional

from kg_gen.models import Graph
from kg_gen.utils.deduplicate import run_semhash_deduplication
from kg_gen.utils.llm_deduplicate import LLMDeduplicate
from sentence_transformers import SentenceTransformer
import dspy
import enum


class DeduplicateMethod(enum.Enum):
    SEMHASH = "semhash"  # Deduplicate using deterministic rules and semantic hashing
    LM_BASED = (
        "lm_based"  # Deduplicate using KNN clustering + Intra cluster LM deduplication
    )
    FULL = "full"  # Deduplicate using both semantic hashing and KNN clustering + Intra cluster LM deduplication


def run_deduplication(
    lm: dspy.LM,
    graph: Graph,
    method: DeduplicateMethod = DeduplicateMethod.FULL,
    retrieval_model: SentenceTransformer | None = None,
    semhash_similarity_threshold: float = 0.95,
    language: str = "en",
    semhash_model_name: Optional[str] = None,
    linguistic_profile: Optional[str] = None,
    keep_negation_apart: bool = True,
) -> Graph:
    # Fail on the public boundary: an unknown value used to fall through the
    # branches below and end in an UnboundLocalError, or be masked by the
    # retrieval-model check. The string values ("semhash", ...) are accepted.
    try:
        method = DeduplicateMethod(method)
    except ValueError:
        raise ValueError(
            f"Unsupported deduplication method {method!r}; expected one of "
            f"{[m.value for m in DeduplicateMethod]}"
        ) from None

    if method != DeduplicateMethod.SEMHASH and retrieval_model is None:
        raise ValueError("No retrieval model provided")

    if method == DeduplicateMethod.SEMHASH:
        deduplicated_graph = run_semhash_deduplication(
            graph,
            semhash_similarity_threshold,
            language=language,
            semhash_model_name=semhash_model_name,
            linguistic_profile=linguistic_profile,
            keep_negation_apart=keep_negation_apart,
        )
    elif method == DeduplicateMethod.LM_BASED:
        llm_deduplicate = LLMDeduplicate(
            retrieval_model,
            lm,
            graph,
            language=language,
            linguistic_profile=linguistic_profile,
        )
        llm_deduplicate.cluster()
        deduplicated_graph = llm_deduplicate.deduplicate()
    elif method == DeduplicateMethod.FULL:
        deduplicated_graph = run_semhash_deduplication(
            graph,
            semhash_similarity_threshold,
            language=language,
            semhash_model_name=semhash_model_name,
            linguistic_profile=linguistic_profile,
            keep_negation_apart=keep_negation_apart,
        )
        llm_deduplicate = LLMDeduplicate(
            retrieval_model,
            lm,
            deduplicated_graph,
            language=language,
            linguistic_profile=linguistic_profile,
        )
        llm_deduplicate.cluster()
        deduplicated_graph = llm_deduplicate.deduplicate()

    return deduplicated_graph
