# Project Requirement Document (PRD)

## 1. Project Overview
**Name:** Project Sanjeevani (Evaluative Self-Healing RAG Pipeline)
**Description:** A locally-orchestrated, tri-modal retrieval benchmarking tool and self-healing LLM agent. It evaluates the performance of BM25, Dense Vector, and Open Knowledge Graph retrieval methods while utilizing a stateful LangGraph agent to prevent and heal hallucinations before serving answers.

## 2. Target Audience
* **Primary:** Internal AI Engineers and Data Scientists.
* **Secondary:** System Architects evaluating cost-to-performance ratios for production RAG systems.

## 3. Core Features
* **Tri-Modal Retrieval Engine:** Combines Sparse (BM25), Dense (Local MiniLM), and Structural (Knowledge Graph) search using Reciprocal Rank Fusion (RRF).
* **Stateful Orchestration:** LangGraph-based cyclic workflow for evaluation, generation, and critique.
* **Self-Healing Loop:** An internal "Critic Node" that intercepts ungrounded AI answers and forces re-retrieval.
* **LLM-as-a-Judge Evaluation:** Automated measurement of hallucination rates (faithfulness), answer relevancy, context precision, and context recall.
* **Performance Benchmarking:** Real-time logging of latency and memory footprint across different retrieval methodologies.

## 4. Success Metrics
* **Accuracy:** Hallucination rate strictly < 5% on the golden dataset.
* **Performance:** End-to-end response generation (including critique loop) under 3 seconds using cloud APIs (Groq/Gemini).
* **Resource Efficiency:** Total local memory consumption must stay comfortably below 10GB to ensure system stability on the host machine's 16GB RAM limit.