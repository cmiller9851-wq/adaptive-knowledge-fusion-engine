# AK-F Engine Blog Pipeline – End-to-End

Turn any blog into a live MoE expert in <15 minutes.

Real commands. Real Docker. Real Neo4j + FAISS + router registration.

Tested on a cracked iPhone 12 mini with Lifeline data. Real architects ship from anywhere. 📱🔥

→ Swervin’ Curvin October/November 2025 now lives inside the engine  
→ Query: “What does Swervin Curvin say about LoRA adapters?” → instant citation

MIT licensed. Fork it. Run it. Sell it.

#RAG #MoE #knowledge-graph #indie-ai
# 🚀 Adaptive Knowledge-Fusion (AK-F) Engine

**Project Status:** ⚙️ In Development (Deployment Prototype Phase)

---

## Vision: Hybrid AI for Seamless Reasoning and Creativity

The Adaptive Knowledge-Fusion (AK-F) Engine is a novel large language model architecture designed to fluidly switch between highly rigorous, expert-level domain reasoning and open-ended creative generation.

The goal is to handle complex requests, such as: *“Provide a rigorous medical diagnosis, then instantly generate a short story about the patient's journey,"* using a single, unified system.

## 🧠 Core Architecture Overview

The AK-F Engine is a hybrid system built on four primary, dynamically interacting components:

1.  **Core Knowledge-Graph Structure:** A structured repository of factual and narrative relationships (using Neo4j/JanusGraph) supporting typed triples and Graph-BERT embeddings.
2.  **Mixture-of-Experts (MoE) Routing:** A dynamic router that selects between specialized experts (e.g., Diagnostic Expert, Creative Expert) based on the input query's intent and context.
3.  **Retrieval-Augmented Memory (RAM):** A two-stage pipeline (sparse lexical search + dense vector search) used to retrieve relevant context passages (chunks) from both the core graph and external sources (like the Swervin Curvin blog).
4.  **Generative Transformers:** A decoder-only base model paired with mode-conditioned adapter layers (`<REASON>` vs `<CREATIVE>`) for final output generation.

## 💻 Deployment and Infrastructure

The engine is deployed using a secure, scalable microservices architecture on Kubernetes:

| Component | Store / Technology | Key Feature |
| :--- | :--- | :--- |
| **MoE Router** | Kubernetes Deployment (GPU-enabled) | HPA scaling based on **GPU utilization** (using DCGM Exporter metrics). |
| **Knowledge Graph** | Neo4j Aura / JanusGraph | Graph-BERT embeddings allow for continuous space querying. |
| **Vector Store** | FAISS / Milvus (Distributed) | Stores embeddings for RAM service. |
| **Logging** | Fluent Bit Sidecar → Loki | Centralized, secure log shipping via NetworkPolicy. |

## 🔗 Knowledge Ingestion Example (Swervin Curvin Blog)

A key success metric is the ability to rapidly integrate external content. The following artifacts detail the pipeline used to ingest the Swervin Curvin blog archives:

* **Scrapy Spider:** Used for crawling and token-based chunking (`curvin_spider.py`).
* **Neo4j Import Script:** Loads chunks as `Article` and `Chunk` nodes in the graph (`load_into_neo4j.py`).
* **MoE Patch:** Registers the `BlogExpert` to surface external references when relevant.

## 🤝 License

This project, including all code, is released under the **MIT License**.
