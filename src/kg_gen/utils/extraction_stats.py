"""Counters for where triples are lost or left dangling during extraction.

Purely additive: collecting stats never changes what the pipeline extracts.
Every hook takes an `ExtractionStats | None` and does nothing when it is None,
which is the default everywhere, so the instrumented paths behave exactly as
they did before.

Why this exists
---------------
Relation extraction has several points at which a triple can disappear or
survive in a broken form, and none of them left a trace. Without counts it is
impossible to tell "the model never proposed the relation" (a limit of the
model) from "the model proposed it and we discarded it" (a pipeline defect) --
opposite diagnoses that call for opposite responses.

The two extraction routes fail differently, and the counters reflect that
rather than papering over it:

  - The litellm route (`no_dspy=True`) constrains subject and object to the
    entity list with a JSON-schema Literal, and filters again while parsing.
    Triples whose endpoints are not exact matches are DROPPED.
  - The default dspy route does not constrain them at all: `Relation` declares
    subject and object as plain `str`. Triples whose endpoints are not in the
    entity list are KEPT, and flow into the Graph as relations pointing at
    entities the graph does not contain.

So `endpoint_missing` is a loss on one route and a dangling-reference defect
on the other. It is counted the same way on both, and `route` records which
one produced the numbers.

Chunk measurements
------------------
`chunk_chars` and `chunk_words` are recorded per chunk because chunking is
done by character count, which is not language-neutral: Spanish text runs
roughly 15-25% longer than English for the same content, so a fixed-size
window carries fewer facts in Spanish and cuts across more relations. Words
per chunk is the direct measure of that; the ratio to characters shows the
expansion itself.
"""

from __future__ import annotations

import copy
import json
import threading
from dataclasses import dataclass, field, fields
from typing import Optional


@dataclass
class ExtractionStats:
    """Thread-safe counters for one extraction run.

    Chunks are processed concurrently, so every mutation takes the lock.
    Reading the fields directly is only safe once the run has finished; use
    `snapshot()` while it is in flight.
    """

    # --- Chunking ---
    chunks: int = 0
    chunk_chars: list[int] = field(default_factory=list)
    chunk_words: list[int] = field(default_factory=list)

    # --- Step 1 ---
    entities_extracted: int = 0

    # --- Step 2: what the model returned ---
    relations_returned: int = 0

    # Triples whose subject and/or object is not an exact match to an
    # extracted entity. Dropped on the litellm route, kept (dangling) on the
    # dspy route -- see the module docstring.
    endpoint_missing_subject: int = 0
    endpoint_missing_object: int = 0
    endpoint_missing_either: int = 0

    # --- Step 2: parse failures (litellm route) ---
    # A response whose JSON could not be decoded at all. The whole chunk's
    # triples are lost and their number is unknowable, so this counts
    # responses, not triples.
    responses_json_invalid: int = 0
    # Strict Pydantic validation failed and parsing fell back to raw JSON.
    # Not a loss by itself; it is the signal that endpoints did not match.
    strict_validation_failed: int = 0
    # Items missing a subject, predicate or object field.
    items_missing_field: int = 0

    # --- Step 2: exception fallback (dspy route) ---
    # The primary extraction raised and the fallback signature ran instead.
    fallback_invoked: int = 0
    # Triples the fallback discarded because their endpoints did not match.
    fallback_dropped: int = 0

    # Which extraction route produced these numbers: "dspy" or "litellm".
    route: Optional[str] = None

    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    # ---- Recording ----

    def record_chunk(self, text: str) -> None:
        with self._lock:
            self.chunks += 1
            self.chunk_chars.append(len(text))
            self.chunk_words.append(len(text.split()))

    def record_entities(self, count: int) -> None:
        with self._lock:
            self.entities_extracted += count

    def record_route(self, route: str) -> None:
        with self._lock:
            self.route = route

    def record_relations(
        self,
        relations,
        entities,
    ) -> None:
        """Count returned triples and how many have unmatched endpoints.

        `relations` may hold tuples or objects with subject/predicate/object
        attributes, since the two routes produce different shapes.
        """
        entity_set = set(entities)
        returned = 0
        missing_subject = 0
        missing_object = 0
        missing_either = 0

        for relation in relations:
            returned += 1
            if isinstance(relation, (tuple, list)):
                subject, _, object_ = relation
            else:
                subject, object_ = relation.subject, relation.object

            subject_missing = subject not in entity_set
            object_missing = object_ not in entity_set
            missing_subject += subject_missing
            missing_object += object_missing
            missing_either += subject_missing or object_missing

        with self._lock:
            self.relations_returned += returned
            self.endpoint_missing_subject += missing_subject
            self.endpoint_missing_object += missing_object
            self.endpoint_missing_either += missing_either

    def increment(self, name: str, count: int = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + count)

    # ---- Reading ----

    def snapshot(self) -> dict:
        """A consistent copy of the counters, safe to call mid-run."""
        # Not `dataclasses.asdict`: it deep-copies every field, and a lock
        # cannot be copied, so it raised on every call.
        with self._lock:
            return {
                f.name: copy.copy(getattr(self, f.name))
                for f in fields(self)
                if f.name != "_lock"
            }

    def summary(self) -> dict:
        """Counters plus the derived rates the counters exist to produce."""
        data = self.snapshot()
        returned = data["relations_returned"]
        chars = data["chunk_chars"]
        words = data["chunk_words"]

        data["rates"] = {
            # Share of returned triples with an endpoint outside the entity
            # set. On the dspy route this is the dangling-relation rate; on
            # the litellm route it is the drop rate.
            "endpoint_missing": (
                data["endpoint_missing_either"] / returned if returned else None
            ),
            "relations_per_entity": (
                returned / data["entities_extracted"]
                if data["entities_extracted"]
                else None
            ),
            "mean_chunk_chars": (sum(chars) / len(chars) if chars else None),
            "mean_chunk_words": (sum(words) / len(words) if words else None),
            # Characters per word: the language-expansion measure. A higher
            # value means a fixed-character chunk holds fewer words.
            "chars_per_word": (sum(chars) / sum(words) if sum(words) else None),
        }
        return data

    def write(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.summary(), handle, indent=2, ensure_ascii=False)
