import os
import json
import re
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from google import genai
from dotenv import load_dotenv

from src.retrieval.fusion import HybridRetriever
from .state import RagAgentState

# Load environment variables (for API keys)
load_dotenv()

class AgentNodes:
    """Implements the node logic for our stateful LangGraph Q&A pipeline."""
    
    def __init__(self, retriever: HybridRetriever) -> None:
        """Initializes the Agent Nodes with the RRF Hybrid Retriever and API clients.
        
        Args:
            retriever: The fused tri-modal retriever engine.
        """
        self.retriever = retriever
        
        # Enforce API Keys
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
            
        # Initialize Groq interface (temperature 0.0 for deterministic answer drafting)
        self.generator_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
        
        # Initialize direct Google GenAI client to bypass LangChain gRPC hanging/connection issues
        self.genai_client = genai.Client()
        self.gemini_model = "gemini-2.0-flash"
        
    def retrieve_node(self, state: RagAgentState) -> Dict[str, Any]:
        """Retrieves and packages document chunks from the tri-modal hybrid search engine.
        
        Args:
            state: The current LangGraph agent state.
            
        Returns:
            Dict containing the retrieved_chunks update.
        """
        query = state.get("current_search_query") or state.get("original_query")
        print(f"\n[Node: Retrieve] Fetching chunks for query: '{query}'")
        docs = self.retriever.retrieve(query)
        
        # Convert Document list to serialized list of dicts for Graph State portability
        chunks_data = []
        for doc in docs:
            chunks_data.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
            
        print(f"[Node: Retrieve] Found {len(chunks_data)} re-ranked chunks.")
        return {"retrieved_chunks": chunks_data}
        
    def generate_node(self, state: RagAgentState) -> Dict[str, Any]:
        """Generates a draft answer using Llama 3.1 on Groq based strictly on context.
        
        Args:
            state: The current LangGraph agent state.
            
        Returns:
            Dict containing the draft_answer update.
        """
        print("\n[Node: Generate] Drafting answer using Llama-3.1-8b-instant...")
        original_query = state["original_query"]
        retrieved_chunks = state.get("retrieved_chunks", [])
        
        # Assemble context string with source annotations
        context_parts = []
        for idx, chunk in enumerate(retrieved_chunks):
            source = os.path.basename(chunk["metadata"].get("source", "Unknown"))
            context_parts.append(f"--- Chunk {idx+1} [Source: {source}] ---\n{chunk['page_content']}")
        context_str = "\n\n".join(context_parts)
        
        system_prompt = (
            "You are a helpful, factual AI assistant for Project Sanjeevani.\n"
            "Your task is to answer the user's question based strictly on the retrieved context below.\n"
            "Follow these rules:\n"
            "1. Rely ONLY on clear facts directly mentioned in the context.\n"
            "2. DO NOT assume, extrapolate, or bring in outside knowledge.\n"
            "3. If the context does not contain the answer, draft a response stating the information is not fully available in the context but do not make up facts.\n\n"
            f"=== RETRIEVED CONTEXT ===\n{context_str}"
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=original_query)
        ]
        
        response = self.generator_llm.invoke(messages)
        draft = response.content
        print(f"[Node: Generate] Draft answer generated ({len(draft)} characters).")
        return {"draft_answer": draft}
        
    def critic_node(self, state: RagAgentState) -> Dict[str, Any]:
        """Evaluates draft groundedness and checks for hallucinations using Gemini 2.0 Flash.
        
        Args:
            state: The current LangGraph agent state.
            
        Returns:
            Dict containing the critique_score and critique_feedback updates.
        """
        print("\n[Node: Critic] Evaluating draft groundedness using Gemini 2.0 Flash...")
        draft = state["draft_answer"]
        retrieved_chunks = state.get("retrieved_chunks", [])
        
        # Assemble context string
        context_parts = []
        for idx, chunk in enumerate(retrieved_chunks):
            source = os.path.basename(chunk["metadata"].get("source", "Unknown"))
            context_parts.append(f"--- Chunk {idx+1} [Source: {source}] ---\n{chunk['page_content']}")
        context_str = "\n\n".join(context_parts)
        
        prompt = (
            "You are an objective Critic and Hallucination Detector for a RAG pipeline.\n"
            "Evaluate whether the generated draft answer is strictly grounded in and supported by the retrieved context chunks.\n"
            "Analyze the draft sentence-by-sentence against the context.\n"
            "Provide your response ONLY as a valid JSON object with these keys:\n"
            "- \"score\": a float value between 0.0 and 1.0 (1.0 means perfectly grounded; < 0.7 means hallucinations or ungrounded assertions exist)\n"
            "- \"feedback\": a string explaining the reason for the score, outlining any ungrounded facts, hallucinations, or why it is fully grounded.\n\n"
            "Do not include markdown formatting (such as ```json) or any conversational text. Return only the raw JSON.\n\n"
            f"=== RETRIEVED CONTEXT ===\n{context_str}\n\n"
            f"=== DRAFT ANSWER ===\n{draft}"
        )
        
        import time
        max_retries = 3
        backoff = 2
        raw_content = ""
        for attempt in range(max_retries):
            try:
                response = self.genai_client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt
                )
                raw_content = response.text.strip()
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    sleep_time = backoff ** (attempt + 1)
                    print(f"[Node: Critic] Rate limited (429). Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    print(f"[Node: Critic] Warning - Failed to query critique model: {e}. Defaulting to score 0.0.")
                    score = 0.0
                    feedback = f"Critique engine failure: {str(e)}"
                    return {
                        "critique_score": score,
                        "critique_feedback": feedback
                    }
                    
        try:
            # Clean markdown code blocks if the model generated them
            json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
            if json_match:
                raw_content = json_match.group(0)
                
            critique_data = json.loads(raw_content)
            score = float(critique_data.get("score", 0.0))
            feedback = str(critique_data.get("feedback", "No feedback provided."))
            
        except Exception as e:
            print(f"[Node: Critic] Warning - Failed to parse critique output: {e}. Defaulting to score 0.0.")
            score = 0.0
            feedback = f"Critique engine parse failure: {str(e)}"
            
        # Safe ASCII prints for Windows console
        safe_feedback = feedback.encode('ascii', errors='replace').decode('ascii')
        print(f"[Node: Critic] Groundedness Score: {score:.2f}")
        print(f"[Node: Critic] Feedback: {safe_feedback}")
        
        return {
            "critique_score": score,
            "critique_feedback": feedback
        }
        
    def rewrite_node(self, state: RagAgentState) -> Dict[str, Any]:
        """Rewrites and optimizes the search query based on feedback using Gemini.
        
        Args:
            state: The current LangGraph agent state.
            
        Returns:
            Dict containing rewritten current_search_query, incremented loop_count, and cleared draft_answer.
        """
        original_query = state["original_query"]
        current_query = state.get("current_search_query") or original_query
        feedback = state.get("critique_feedback", "")
        loop_count = state.get("loop_count", 0)
        
        print(f"\n[Node: Rewrite] Optimizing search query (Healing Loop Retry {loop_count+1}/3)...")
        
        prompt = (
            "You are an expert search engine query optimizer.\n"
            "Your task is to take the original user question, the previous search query, and the critique feedback explaining what was missing or hallucinated, "
            "and construct an improved keyword and semantic search query to fetch the correct context.\n"
            "Return ONLY the new query string. Do not include any quotes, explanations, or introductory text.\n\n"
            f"Original Question: {original_query}\n"
            f"Previous Query: {current_query}\n"
            f"Critique Feedback: {feedback}"
        )
        
        import time
        max_retries = 3
        backoff = 2
        new_query = original_query
        for attempt in range(max_retries):
            try:
                response = self.genai_client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt
                )
                new_query = response.text.strip().replace('"', '').replace("'", "")
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    sleep_time = backoff ** (attempt + 1)
                    print(f"[Node: Rewrite] Rate limited (429). Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    print(f"[Node: Rewrite] Warning - Query rewrite failed: {e}. Using original query.")
                    new_query = original_query
            
        print(f"[Node: Rewrite] Optimized search query generated: '{new_query}'")
        
        return {
            "current_search_query": new_query,
            "loop_count": loop_count + 1,
            "draft_answer": ""  # Reset draft so a fresh answer is generated using the new context
        }
        
    def fallback_node(self, state: RagAgentState) -> Dict[str, Any]:
        """Sets a clean safety fallback message when maximum healing loop retries are exhausted.
        
        Args:
            state: The current LangGraph agent state.
            
        Returns:
            Dict containing fallback draft_answer.
        """
        print("\n[Node: Fallback] Maximum retries reached. Setting safe output fallback.")
        return {"draft_answer": "Information not found."}
