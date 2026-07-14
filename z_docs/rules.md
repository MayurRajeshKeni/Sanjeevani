# AI Development Rules & Boundaries

## 1. Technology Constraints
* **Memory Limit Strictly Enforced:** The host machine has a hard cap of 16GB shared RAM. **DO NOT** attempt to load large LLMs (e.g., Llama 7B, Mistral) locally via Ollama or HuggingFace. 
* **Local Compute Mapping:** Confine local compute strictly to:
    * Python execution and LangGraph state management.
    * In-memory vector operations (FAISS/Chroma).
    * Extremely small embedding models (`all-MiniLM-L6-v2` at ~90MB).
* **Cloud API Delegation:** All generation, reasoning, and critique tasks **MUST** be routed to remote APIs (Groq and Gemini).

## 2. Code Quality & Formatting
* **Language:** Python 3.11+.
* **Typing:** Strict type hinting is mandatory for all function definitions (`def retrieve(query: str) -> list[Document]:`).
* **Async:** Use `asyncio` for API calls to Groq and Gemini to prevent blocking the LangGraph state machine.
* **Error Handling:** Implement graceful fallbacks. If an API rate limit is hit, implement exponential backoff. If the agent loop fails 3 times, output a clean "Information not found" string, do not throw a raw exception to the UI.

## 3. Security Boundaries
* Never hardcode API keys in the source files. Always use `os.getenv()` and `python-dotenv`.
* Exclude the `.env` file from git tracking via `.gitignore`.