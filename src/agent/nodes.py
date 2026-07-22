import os
import json
import re
import time
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
    """Implements resilient, stateful node logic for our LangGraph Q&A pipeline with multi-provider failover."""
    
    def __init__(self, retriever: HybridRetriever) -> None:
        """Initializes the Agent Nodes with the RRF Hybrid Retriever and API clients.
        
        Args:
            retriever: The fused tri-modal retriever engine.
        """
        self.retriever = retriever
        
        # Enforce API Keys
        if not os.getenv("GROQ_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("At least one of GROQ_API_KEY or GOOGLE_API_KEY must be set.")
            
        # Initialize Groq interface if key is present
        self.generator_llm = None
        if os.getenv("GROQ_API_KEY"):
            try:
                self.generator_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0, max_retries=5)
            except Exception as e:
                print(f"[AgentNodes] Warning: Could not initialize ChatGroq: {e}")
        
        # Initialize Google GenAI client if key is present
        self.genai_client = None
        if os.getenv("GOOGLE_API_KEY"):
            try:
                self.genai_client = genai.Client()
            except Exception as e:
                print(f"[AgentNodes] Warning: Could not initialize Google GenAI Client: {e}")
                
        self.gemini_models = ["gemini-2.0-flash", "gemini-2.0-flash-lite"]
        
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
        
    def _generate_with_gemini(self, system_prompt: str, original_query: str) -> str:
        """Fallback method to generate a draft using Google Gemini when Groq is unavailable/limited."""
        if not self.genai_client:
            return ""
            
        prompt = f"{system_prompt}\n\nUser Question: {original_query}"
        for model_name in self.gemini_models:
            for attempt in range(2):
                try:
                    res = self.genai_client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    if res.text and res.text.strip():
                        print(f"[Node: Generate] Successfully generated answer via Gemini fallback ({model_name}).")
                        return res.text.strip()
                except Exception as e:
                    print(f"[Node: Generate] Gemini ({model_name}) attempt {attempt+1} error: {e}")
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        time.sleep(3 * (attempt + 1))
        return ""
        
    def generate_node(self, state: RagAgentState) -> Dict[str, Any]:
        """Generates a draft answer using Llama 3.1 on Groq with seamless Gemini failover.
        
        Args:
            state: The current LangGraph agent state.
            
        Returns:
            Dict containing the draft_answer update.
        """
        print("\n[Node: Generate] Drafting answer using Llama-3.1-8b-instant (with Gemini failover)...")
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
        
        draft = ""
        # 1. Try Groq generator first if available
        if self.generator_llm:
            max_retries = 3
            backoff = 2
            for attempt in range(max_retries):
                try:
                    response = self.generator_llm.invoke(messages)
                    draft = response.content
                    if draft and draft.strip():
                        break
                except Exception as e:
                    print(f"[Node: Generate] Groq attempt {attempt+1}/{max_retries} error: {e}")
                    if attempt < max_retries - 1:
                        sleep_time = backoff ** (attempt + 1)
                        time.sleep(sleep_time)
                        
        # 2. Fallback to Gemini if Groq did not return a valid draft
        if not draft or not draft.strip():
            print("[Node: Generate] Falling back to Gemini for answer generation...")
            draft = self._generate_with_gemini(system_prompt, original_query)
            
        if not draft or not draft.strip():
            draft = "Information not found due to generation model failure."
            
        print(f"[Node: Generate] Draft answer generated ({len(draft)} characters).")
        return {"draft_answer": draft}
        
    def critic_node(self, state: RagAgentState) -> Dict[str, Any]:
        """Evaluates draft groundedness and checks for hallucinations using Gemini with rate-limit safety.
        
        Args:
            state: The current LangGraph agent state.
            
        Returns:
            Dict containing the critique_score and critique_feedback updates.
        """
        print("\n[Node: Critic] Evaluating draft groundedness using Gemini (with rate limit safety)...")
        draft = state["draft_answer"]
        retrieved_chunks = state.get("retrieved_chunks", [])
        
        # If draft indicates generation failure, don't waste critique calls
        if "generation model failure" in draft.lower():
            return {
                "critique_score": 0.0,
                "critique_feedback": "Generation model failed to produce a draft."
            }
            
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
        
        raw_content = ""
        is_rate_limited = False
        
        if self.genai_client:
            for model_name in self.gemini_models:
                max_retries = 2
                for attempt in range(max_retries):
                    try:
                        response = self.genai_client.models.generate_content(
                            model=model_name,
                            contents=prompt
                        )
                        if response.text and response.text.strip():
                            raw_content = response.text.strip()
                            break
                    except Exception as e:
                        err_str = str(e).lower()
                        print(f"[Node: Critic] Model {model_name} attempt {attempt+1} error: {e}")
                        if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                            is_rate_limited = True
                            time.sleep(3 * (attempt + 1))
                if raw_content:
                    break
                    
        # Safety Guard: If Critic API is rate limited, DO NOT set score=0.0 (which triggers an API retry storm!).
        # Accept the draft answer with score=1.0 and a descriptive feedback note.
        if not raw_content:
            if is_rate_limited:
                print("[Node: Critic] Rate limit hit on Critic API. Accepting draft answer to prevent retry storm.")
                return {
                    "critique_score": 1.0,
                    "critique_feedback": "Critique model rate limited (429); accepted draft answer to prevent retry loop."
                }
            else:
                print("[Node: Critic] Warning - Failed to query critique model. Passing draft answer.")
                return {
                    "critique_score": 1.0,
                    "critique_feedback": "Critique engine unavailable; passing draft answer."
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
            print(f"[Node: Critic] Warning - Failed to parse critique output: {e}. Defaulting to score 0.8.")
            score = 0.8
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
        
        new_query = original_query
        if self.genai_client:
            for model_name in self.gemini_models:
                try:
                    response = self.genai_client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    if response.text and response.text.strip():
                        new_query = response.text.strip().replace('"', '').replace("'", "")
                        break
                except Exception as e:
                    print(f"[Node: Rewrite] Model {model_name} error: {e}")
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        time.sleep(3)
                        
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

