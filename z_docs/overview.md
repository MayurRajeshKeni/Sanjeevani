# Project Sanjeevani: Master Overview

## The Core Concept
Project Sanjeevani is a locally-orchestrated, hybrid Retrieval-Augmented Generation (RAG) benchmarking pipeline equipped with a self-healing agentic workflow. Instead of acting as a simple Q&A bot, it rigorously evaluates three distinct data-fetching methodologies (Sparse Lexical, Dense Semantic, and Structured Knowledge Graph) and utilizes an autonomous "Critic Agent" to catch, penalize, and fix LLM hallucinations before the final output is served.

## Hardware Optimization Strategy
Engineered specifically to run efficiently on an Intel Core Ultra 7 processor with a strict 16GB RAM limit. 
* **Local Compute (CPU/RAM):** Handles orchestration (LangGraph), document parsing (Markdown-to-Graph), chunking, BM25 lexical search, in-memory vector databases (FAISS/Chroma), and lightweight dense embeddings (`all-MiniLM-L6-v2` at ~90MB).
* **Cloud Compute (Free APIs):** Offloads heavy LLM inference to prevent out-of-memory crashes. Groq provides ultra-fast generation, while Gemini 1.5 Flash provides massive context windows for evaluation.

## The End-to-End Pipeline
1. **Ingestion & Tri-Indexing:** Documents are ingested, parsed, and mapped to three distinct local engines: a keyword-based BM25 index, a semantic FAISS/Chroma vector index, and a hierarchical Open Knowledge Graph (JSON).
2. **Hybrid Retrieval:** Upon receiving a query, all three engines retrieve their top context chunks. A Reciprocal Rank Fusion (RRF) algorithm mathematically merges and re-ranks these chunks to balance exact keyword matches, conceptual relevance, and structural context.
3. **Draft Generation:** The unified context chunks and the user query are sent to Groq (`Llama-3-8b-8192`). Groq drafts an initial answer at high speed.
4. **The Critic Node (Evaluation):** Before the user sees the answer, LangGraph routes the draft and the source chunks to Gemini 1.5 Flash. Gemini acts as an impartial judge, evaluating the draft strictly for groundedness and hallucination.
5. **The Self-Healing Loop:** * **If Pass:** The answer is verified and delivered to the user.
    * **If Fail:** LangGraph blocks the output, reformulates the search query, and triggers a re-retrieval. If it fails 3 consecutive times, it safely degrades to "Information not found" rather than lying.
6. **Continuous CI/CD Benchmarking:** Any code or prompt changes trigger a GitHub Action that runs the pipeline against a 100-question "Golden Dataset." Ragas/TruLens measures hallucination rates, answer relevancy, and faithfulness, blocking PR merges if metrics regress.
7. **The Dashboard UI:** A Streamlit interface visualizes the final answer alongside critical developer metrics: end-to-end latency (ms), token usage/cost, and a breakdown of the RRF score (showing whether BM25, Vector, or Graph search contributed the most to the correct answer).