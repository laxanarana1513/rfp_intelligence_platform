import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.app import create_app
from rfp_intel.schemas import SearchHit


def test_search_evaluation_scores_both_modes(monkeypatch):
    calls = []

    def fake_search(query, **kwargs):
        calls.append((query, kwargs))
        irrelevant = SearchHit(
            chunk_id="other",
            text="No matching passage here.",
            score=0.9,
            file_name="bid.html",
            page_number=1,
            section_heading="",
            section_id=None,
            bid_id="Bid 1",
            doc_type="bid_page",
            addendum_number=0,
        )
        relevant = SearchHit(
            chunk_id="match",
            text="Solicitation JA-42 is listed here.",
            score=0.8,
            file_name="bid.html",
            page_number=2,
            section_heading="",
            section_id=None,
            bid_id="Bid 1",
            doc_type="bid_page",
            addendum_number=0,
        )
        return [relevant] if kwargs["mode"] == "hybrid_rerank" else [irrelevant, relevant]

    monkeypatch.setattr("api.app.search_bids", fake_search)
    client = TestClient(create_app())

    response = client.post(
        "/search/evaluate",
        json={
            "query": "What is the solicitation number?",
            "passage": "JA-42",
            "bid_id": "Bid 1",
            "file_contains": "bid",
        },
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert results["hybrid"]["recall@5"] == 1.0
    assert results["hybrid"]["mrr"] == 0.5
    assert results["hybrid_rerank"]["recall@5"] == 1.0
    assert results["hybrid_rerank"]["mrr"] == 1.0
    assert [kwargs["mode"] for _query, kwargs in calls] == ["hybrid", "hybrid_rerank"]
    assert all(kwargs["top_k"] == 5 for _query, kwargs in calls)