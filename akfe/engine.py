from typing import List
from .retrievers import BM25Retriever, FaissRetriever
from .fusion.bandit_weighting import ThompsonBandit
from .models import DocumentResult

class FusionEngine:
    def __init__(self):
        # initialise retrievers
        self.bm25 = BM25Retriever()
        self.faiss = FaissRetriever()
        # bandit knows the two sources
        self.bandit = ThompsonBandit(sources=["bm25", "faiss"])

    def _retrieve(self, query: str) -> List[DocumentResult]:
        bm25_hits = self.bm25.search(query, k=50)
        faiss_hits = self.faiss.search(query, k=50)

        # attach source label
        for r in bm25_hits:
            r.source = "bm25"
        for r in faiss_hits:
            r.source = "faiss"

        return bm25_hits + faiss_hits

    def query(self, query: str, top_k: int = 10) -> List[DocumentResult]:
        candidates = self._retrieve(query)

        # get current weight sample from bandit
        weights = self.bandit.sample_weights()

        # compute weighted score
        for doc in candidates:
            base = doc.score  # each retriever already gives a similarity score
            doc.combined_score = base * weights.get(doc.source, 0.0)

        # sort and cut
        ranked = sorted(candidates, key=lambda d: d.combined_score, reverse=True)
        top = ranked[:top_k]

        # OPTIONAL: after user feedback you would call self.bandit.update(...)
        return top
