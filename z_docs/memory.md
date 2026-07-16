# AI System Memory & State Tracker
*(Instructions for AI: Read this file at the start of every session. Update this file at the end of every session or when a major feature is complete. Do not change the structure, only update the values.)*

## Current Status
* **Project Phase:** Phase 4 - CI/CD & UI Dashboard
* **Current Active Task:** Create the `golden_dataset.json` following the PRD schema, write the automated evaluation script using `ragas`, and build the Streamlit UI dashboard visualizing retriever and agent states.
* **Last Modified:** July 16, 2026

## Completed Features
* Architectural Planning & Documentation Complete (Overview, PRD, Architecture, Rules, Phases, Design finalized).
* Tri-Modal strategy (BM25, Vector, Graph) and LangGraph State Schema successfully defined.
* Phase 1 Complete: Scaffolded directories, configured requirements, created loaders, chunkers, and Markdown-to-Graph parser.
* Phase 2 Complete: Built Vector (FAISS), Sparse (BM25), and Graph engines, aligning chunk-to-graph nodes and implementing RRF ($k=60$).
* Phase 3 Complete: 
  * Defined `RagAgentState` TypedDict.
  * Programmed nodes for Retrieval, Generation (Groq `llama-3.1-8b-instant`), Critique (Gemini `gemini-flash-latest` via raw HTTP SDK client for stability), Query Rewriter, and Fallback.
  * Assembled cyclic healing loop transitions via LangGraph `StateGraph`.
  * Verified live query execution paths and state machine routing behavior under 3-pass loop limits.
  * Pushed Phase 3 code to GitHub.

## Known Bugs & Issues
* *None.*

## Context Notes for Next Session
* System Python 3.11 environment in AppData must be used to run commands (`py -3.11 ...`) to bypass local WDAC blocks.
* Environment keys for Groq and Gemini are loaded successfully from `.env` in the workspace root.
* Ready to proceed with Phase 4: CI/CD validation (Ragas) and the Streamlit UI dashboard.