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
* Build the Critic Node (Gemini 1.5 Flash API).
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
  * Upgraded Ragas LLM to `gemini-2.0-flash` with sequential `RunConfig(max_workers=1)` and built exponential backoff retries on rate limit (429) warnings to fully stabilize benchmarking runs.
* *Milestone:* Fully operational graphical benchmarking app that blocks pull requests if metrics regress.