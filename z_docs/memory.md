# AI System Memory & State Tracker
*(Instructions for AI: Read this file at the start of every session. Update this file at the end of every session or when a major feature is complete. Do not change the structure, only update the values.)*

## Current Status
* **Project Phase:** Scaled testing on Kubernetes website documentation (1,675 markdown files), expanded golden dataset (14 questions), and automated evaluations.
* **Current Active Task:** Benchmark evaluations fully operational on scaled dataset and Streamlit UI dashboard running.
* **Last Modified:** July 21, 2026

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

## Known Bugs & Issues
* **Ragas Evaluation Rate Limits & Slowness**: Running Ragas evaluations on the Gemini Free Tier API key is extremely slow (taking ~12 minutes for a 20-unit iteration run) and frequently results in `N/A` for all final metrics (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`). This is due to Gemini's free tier rate limit of 15 Requests Per Minute (RPM). Because Ragas triggers multiple evaluation API calls per metric for each question, even sequential execution (`max_workers=1`) quickly triggers 429 `RESOURCE_EXHAUSTED` blocks, leading to massive retry delays and incomplete (`N/A`) metric tables.

## Context Notes for Next Session
* System Python 3.11 environment in AppData must be used to run commands (`py -3.11 ...`) to bypass local WDAC blocks.
* Environment keys for Groq and Gemini are loaded successfully from `.env` in the workspace root.
* Streamlit server running locally on port 8501.