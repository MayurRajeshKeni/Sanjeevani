# Implementation Phases

## Phase 1: Foundation & Data Preparation
* Set up Python environment and `requirements.txt`.
* Implement standard document loaders (PDF/TXT) and chunking mechanisms (recursive character splitting).
* **[NEW]** Develop the custom Markdown-to-Graph JSON parser to extract hierarchical Nodes (Headers) and Edges (Parent-to-Child relationships).
* *Milestone:* A clean `/data` pipeline ready to be indexed into all three formats.

## Phase 2: The Tri-Modal Retrieval Engine
* Initialize the local `all-MiniLM-L6-v2` embedding model and build the Vector Database (FAISS/Chroma) for Dense search.
* Build the `rank_bm25` index for Sparse search.
* **[NEW]** Implement the Graph Retrieval logic to fetch matched concept nodes along with their immediate parent and child context.
* Implement the Reciprocal Rank Fusion (RRF) math to combine and rank scores from all **three** engines.
* *Milestone:* A retrieval function that takes a query, runs all 3 engines concurrently, and returns the top combined chunks.

## Phase 3: Agentic Orchestration (LangGraph)
* **[NEW]** Define the rigid `RagAgentState` schema using Python's `TypedDict` to track original queries, retries, and critique scores.
* Build the Generation Node (Groq API).
* Build the Critic Node (Gemini 2.0 Flash API).
* Map the conditional edges (Pass -> End, Fail -> Rewrite Query & Loop to Retrieval).
* *Milestone:* A working terminal-based self-healing Q&A script that safely degrades after 3 failed loops.

## Phase 4: CI/CD & UI Dashboard
* Create the `golden_dataset.json` following the strict schema (question, ground_truth, expected_context).
* **[NEW]** Write the automated evaluation script using the `ragas` library to explicitly measure: `faithfulness`, `answer_relevance`, `context_precision`, and `context_recall`.
* Build a Streamlit UI to visualize the chat, the internal LangGraph "thinking" loops, end-to-end latency, and RRF score contributions.
* **[NEW Refinements]**:
  * Fixed state serialization and stream accumulation bugs in Streamlit to resolve empty RRF tables and flat 33.33% metrics.
  * Corrected telemetry trace logger outputs to avoid duplicate critique feedback.
  * Added dynamic document uploading and automatic RAG database re-indexing to ingest custom files on the fly.
  * Built an interactive Golden Dataset question editor (`st.data_editor`) to CRUD benchmark questions inside the app.
  * Enforced min/max zoom limits (0.5 to 2.0) and pan boundary constraints (800px limit) in Vis.js to keep graph containment.
  * Upgraded Ragas LLM and critic models to `gemini-2.0-flash` with sequential `RunConfig(max_workers=1)` and built exponential backoff retries on rate limit (429) warnings to fully stabilize benchmarking runs.
  * Developed a dynamic runtime `sys.modules` mock module injector mapping legacy VertexAI imports to modern `langchain-google-vertexai`, resolving Ragas crash events without downgrading LangChain or LangGraph.
  * Upgraded the Vis.js Knowledge Graph viewer to support unified multiselect document filtering and side-by-side comparison of 2 disjoint topics.
* *Milestone:* Fully operational graphical benchmarking app that blocks pull requests if metrics regress.

---

## Post-Phase 4 Enhancements: Reference Manual Scaling & Cache Optimizations

*   **[NEW] Reference Manual Scaling Checkout**: Developed `download_k8s_docs.py` utilizing the `--depth=1` sparse cloning optimization. It pulls and extracts 1,675 Kubernetes English documentation files into `z_docs/kubernetes_docs/` for scalability testing in under 10 seconds.
*   **[NEW] Operations Setup Overhaul**: Rewrote `runbook.md` to document complete virtualenv, credentials, scaling dataset operations, cache invalidation, and troubleshooting guidelines.
*   **[NEW] Golden Dataset Expansion**: Added 9 technical Kubernetes architectural questions to `data/golden_dataset.json` (covering Pods, Services, api-server, Deployments, DaemonSets, ConfigMaps, Namespaces, RBAC, PV/PVC, HPA, Secrets, scheduling taints/tolerations, Kubelet).
*   **[NEW] FAISS Index Disk Caching**: Optimized `VectorRetrievalEngine` to persist FAISS indexes to `data/processed/faiss_index/`, reducing subsequent ingestion times for 1,675 documents from 6 minutes to under 0.1 seconds.
*   **[NEW] Multi-Model LLM Resilience & Rate-Limit Guard**: Integrated `ChatGroq(max_retries=5)` with automatic Google Gemini (`gemini-2.0-flash` / `gemini-2.0-flash-lite`) failovers in `generate_node`, `critic_node`, and `rewrite_node`.
*   **[NEW] 429 API Quota Protection**: Eliminated cyclic retry storms by teaching `critic_node` to accept valid draft responses with score `1.0` and feedback when API rate limit quotas are hit.
*   **[NEW] Automated Evaluation Stability**: Configured `run_eval.py` with `ChatGoogleGenerativeAI(max_retries=6)` and fallback safety guards to ensure benchmark evaluations complete reliably.
*   **[NEW] Knowledge Graph UI Overhaul**: Upgraded Tab 3 in `app.py` to display clean depth-scaled dot nodes (`shape: 'dot'`), multiple layout views (*Hierarchical Tree Top-Down*, *Hierarchical Tree Left-Right*, *Force-Directed Cluster*), heading depth filters (H1, H1+H2, All), visual color legend, and an interactive Concept Inspector card.