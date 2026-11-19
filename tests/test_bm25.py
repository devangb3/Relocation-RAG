from relocation_rag.retrieval.bm25 import rank_chunks_keyword


def test_bm25_accounts_for_length_frequency_and_exact_identifiers():
    docs = {
        "short": "REL-105.1 advance",
        "long": "REL-105.1 advance for an approved transfer",
        "repeat": "REL-105.1 REL-105.1",
        "other": "REL-105.10",
    }
    assert rank_chunks_keyword("rel-105.1?", docs, 3) == ["repeat", "short", "long"]


def test_rare_terms_outweigh_common_terms_and_ties_use_chunk_id():
    docs = {"b": "transfer", "a": "transfer", "c": "transfer", "d": "carton"}
    assert rank_chunks_keyword("transfer carton", docs, 2) == ["d", "a"]
    assert rank_chunks_keyword("!!!", docs, 2) == []
    assert rank_chunks_keyword("transfer", {"empty": ""}, 2) == []
