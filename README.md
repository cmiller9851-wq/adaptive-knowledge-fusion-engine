Adaptive Knowledge Fusion Engine (AKFE)

*Bridging the gap between structured data and unstructured text, one query at a time.*

---

## What is AKFE?

AKFE is an open‑source Python framework that **fuses multiple knowledge sources**—search indexes, vector embeddings, knowledge graphs, and more—into a single, query‑able system.  
It learns on the fly which source is most trustworthy for a given question, delivering higher relevance without the need for massive labeled datasets.

---

## Key Features

| Feature | Why it matters |
|---------|----------------|
| **Adaptive weighting** (Thompson‑sampling bandit) | Dynamically balances lexical (BM25) and semantic (FAISS) signals based on real‑time feedback. |
| **Hybrid retrieval** | Combines classic IR with dense vector search, then optionally reranks with a cross‑encoder. |
| **Modular plugin architecture** | Drop‑in new encoders, retrievers, or graph back‑ends without touching core code. |
| **FastAPI service** | One‑line deployment of a REST endpoint (`POST /query`). |
| **Docker‑Compose stack** | Spin up the API + Milvus/FAISS index locally in seconds. |
| **Explainability** | Every result carries provenance (source, weight, raw score) for transparent debugging. |
| **Zero‑shot multi‑domain support** | Works out‑of‑the‑box on legal, medical, e‑commerce, or any custom corpus. |

---

## Quick Start

```bash
# Clone & install
git clone https://github.com/cmiller9851-wq/adaptive-knowledge-fusion-engine.git
cd adaptive-knowledge-fusion-engine
pip install -r requirements.txt
pip install -e .

# Index a sample corpus
python -m akfe.cli index --data data/sample_corpus.json --store ./store

# Run the API (Docker)
docker-compose up -d
# Or run locally
uvicorn akfe.api:app --reload
```

Now you can query:

```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query":"What are the health benefits of intermittent fasting?"}'
```

Or use the CLI:

```bash
akfe "What are the health benefits of intermittent fasting?" -k 5
```

---

## Project Structure

```
akfe/
│
├─ api.py                # FastAPI entry point
├─ cli.py                # Command‑line interface
├─ engine.py             # Core FusionEngine
├─ retrievers/           # BM25, FAISS, Graph, etc.
├─ encoders/             # Sentence‑transformers, OpenAI, custom TF models
├─ fusion/
│   └─ bandit_weighting.py   # Thompson‑sampling bandit
├─ models/               # Pydantic / dataclasses for results
│
tests/                   # Pytest suite
docker-compose.yml        # Local deployment stack
mkdocs.yml                # Documentation skeleton
README.md                 # You’re reading it!