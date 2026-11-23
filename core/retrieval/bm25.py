"""BM25 keyword scoring for the local policy corpus."""

import math
import re
from collections import Counter

# Treat policy sections and form numbers as complete tokens.
TOKEN = re.compile(r"\b[a-z]+-\d+(?:\.\d+)*\b|\w+")


def rank_chunks_keyword(query: str, documents: dict[str, str], limit: int) -> list[str]:
    """Return positive-scoring chunk IDs, breaking ties by ID."""
    terms = set(TOKEN.findall(query.casefold()))
    if not terms or not documents:
        return []

    frequencies = {
        key: Counter(TOKEN.findall(text.casefold())) for key, text in documents.items()
    }
    lengths = {key: counts.total() for key, counts in frequencies.items()}
    size = len(documents)
    average_length = sum(lengths.values()) / size
    if not average_length:
        return []

    document_frequency = Counter(
        term for counts in frequencies.values() for term in terms.intersection(counts)
    )
    idf = {
        term: math.log1p((size - count + 0.5) / (count + 0.5))
        for term, count in sorted(document_frequency.items())
    }
    k1, b = 1.2, 0.75
    scores = {}
    for key, counts in frequencies.items():
        normalization = k1 * (1 - b + b * lengths[key] / average_length)
        score = sum(
            weight * counts[term] * (k1 + 1) / (counts[term] + normalization)
            for term, weight in idf.items()
            if term in counts
        )
        if score:
            scores[key] = score
    return sorted(scores, key=lambda key: (-scores[key], key))[:limit]
