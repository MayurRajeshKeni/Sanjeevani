# AI System Memory & State Tracker
*(Instructions for AI: Read this file at the start of every session. Update this file at the end of every session or when a major feature is complete. Do not change the structure, only update the values.)*

## Current Status
* **Project Phase:** Phase 3 - Agentic Orchestration (LangGraph)
* **Current Active Task:** Define the rigid `RagAgentState` TypedDict, build the Generation Node (Groq API Llama 3) and Critic Node (Gemini 1.5 Flash), map conditional edges, and create terminal Q&A script.
* **Last Modified:** July 15, 2026

## Completed Features
* Architectural Planning & Documentation Complete (Overview, PRD, Architecture, Rules, Phases, Design finalized).
* Tri-Modal strategy (BM25, Vector, Graph) and LangGraph State Schema successfully defined.
* Phase 1 Complete: Scaffolded directories, configured requirements, created loaders, chunkers, and Markdown-to-Graph parser.
* Phase 2 Complete: 
  * Initialized local all-MiniLM-L6-v2 embeddings and FAISS index for Dense retrieval.
  * Coded rank_bm25 index for Sparse retrieval.
  * Built Graph retrieval matching concept nodes semantically and traversing edges for structural parents/children.
  * Resolved chunk-to-graph mismatch using normalized substring and line-based overlap matching.
  * Implemented RRF fusion math (k=60) returning re-ranked chunks alongside score contributions.
  * Successfully verified all test queries and pushed Phase 2 code to GitHub.

## Known Bugs & Issues
* *None.*

## Context Notes for Next Session
* System Python 3.11 environment in AppData must be used to run commands (`py -3.11 ...`) to bypass OneDrive/User folder Application Control blocks.
* We are ready to execute Phase 3: Agentic Orchestration.