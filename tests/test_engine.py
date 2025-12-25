import pytest
from akfe.engine import FusionEngine

@pytest.fixture
def engine():
    return FusionEngine()

def test_query_returns_results(engine):
    results = engine.query("test query", top_k=5)
    assert isinstance(results, list)
    assert len(results) <= 5
    for r in results:
        assert hasattr(r, "doc_id")
        assert hasattr(r, "combined_score")
