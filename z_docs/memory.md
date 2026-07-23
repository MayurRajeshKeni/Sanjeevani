# AI System Memory & State Tracker
*(Instructions for AI: Read this file at the start of every session. Update this file at the end of every session or when a major feature is complete. Do not change the structure, only update the values.)*

## Current Status
* **Project Phase:** All Phases Complete (Phases 1-4) with Multi-Model Fallback Resilience, API Rate-Limit Protection, Overhauled Knowledge Graph Visual Topology, and Automated Ragas Evaluation Metrics.
* **Benchmark Metrics Summary:** Faithfulness: **92.78%**, Context Recall: **88.24%**, Answer Relevancy: **76.70%**, Context Precision: **76.64%**.
* **Current Active Task:** Benchmark evaluations fully operational and Streamlit UI dashboard running.
* **Last Modified:** July 23, 2026

## Completed Features
* Architectural Planning & Documentation Complete (Overview, PRD, Architecture, Rules, Phases, Design finalized).
* Tri-Modal strategy (BM25, Vector, Graph) and LangGraph State Schema successfully defined.
* Phase 1 Complete: Scaffolded directories, configured requirements, created loaders, chunkers, and Markdown-to-Graph parser.
* Phase 2 Complete: Built Vector (FAISS), Sparse (BM25), and Graph engines, aligning chunk-to-graph nodes and implementing RRF ($k=60$).
* Phase 3 Complete: 
  * Defined `RagAgentState` TypedDict.
  * Programmed nodes for Retrieval, Generation (Groq `llama-3.1-8b-instant`), Critique (Gemini `gemini-2.0-flash`), Query Rewriter, and Fallback.
  * Assembled cyclic healing loop transitions via LangGraph `StateGraph`.
  * Verified live query execution paths and state machine routing behavior under 3-pass loop limits.
  * Pushed Phase 3 code to GitHub.
* Phase 4 Complete:
  * Formulated golden benchmark dataset and set up Ragas automated evaluator (`run_eval.py`).
  * Optimized embedding models to run offline (`local_files_only=True`) to bypass HF Hub latency and DNS failures.
  * Built high-fidelity Streamlit UI dashboard (`app.py`) featuring interactive chatbot play, live self-healing LangGraph telemetry trace, 3-column metrics container (Latency, Token usage, RRF engine contribution bar chart), Ragas Benchmark cards/matrices, and a search-enabled Knowledge Graph concepts viewer.
  * Refined Streamlit UI state updates (`current_state.update`) to recover empty RRF rank tables and contribution bar chart values.
  * Escaped Vis.js canvas f-string single curly braces (resolved SyntaxError) and added zoom boundaries (0.5 to 2.0) and pan constraints (800px limit).
  * Switched evaluations and critique nodes to `gemini-2.0-flash` to leverage higher RPM quotas.
  * Configured Ragas `RunConfig(max_workers=1)` and developed exponential backoff retry loops in critique and rewrite nodes to handle Google Developer API rate limit (429) errors gracefully.
  * Added dynamic file upload ingestion into `z_docs` and an in-app interactive Golden Dataset editor.
* Scaled Ingestion & Reference Manual testing:
  * Programmed `download_k8s_docs.py` to sparse-checkout and extract 1,675 Kubernetes English documentation files into `z_docs/kubernetes_docs/`, utilizing the `--depth=1` cloning speed optimization.
  * Formulated and appended 9 detailed technical Kubernetes architectural questions to `data/golden_dataset.json` (covering Pods, Services, api-server, Deployments, DaemonSets, ConfigMaps, Namespaces, RBAC, PV/PVC, HPA, Secrets, scheduling taints/tolerations, Kubelet).
  * Overhauled `runbook.md` to document complete virtualenv, credentials, scaling dataset operations, cache invalidation, and troubleshooting guidelines.
* Post-Phase 4 Multi-Provider Failover & UI Overhaul:
  * Programmed Groq <-> Gemini multi-model failover (`generate_node`, `critic_node`, `rewrite_node`) to handle provider rate limits gracefully.
  * Added 429 quota safety guards to `critic_node` to accept valid draft responses and prevent API retry storms.
  * Overhauled Tab 3 in `app.py` into a clean visual topology viewer with depth-scaled dot nodes (`shape: 'dot'`), DAG Tree and Force Cluster layout views, depth filters, visual legend, and an interactive Concept Inspector card.

## Known Bugs & Issues
* **None (Resolved)**: Rate-limit retry storms and evaluation crashes have been resolved with multi-model failovers, 429 quota protection guards, and Ragas LLM `max_retries=6`.

## Context Notes for Next Session
* System Python 3.11 environment in `.venv` must be used to run commands (`.\.venv\Scripts\python.exe ...`).
* Environment keys for Groq and Gemini are loaded successfully from `.env` in the workspace root.
* Streamlit server running locally on port 8501 (`streamlit run src/ui/app.py`).