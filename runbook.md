# Sanjeevani Runbook

This runbook contains the concise and concrete steps required to set up, configure, and run **Project Sanjeevani**—a Tri-Modal Self-Healing Retrieval-Augmented Generation (RAG) system.

---

## 🛠️ Step 1: Environment Setup

The project includes an existing virtual environment directory (`.venv`). Follow these commands to activate the virtual environment and install the required dependencies.

### 1. Activate the Virtual Environment
Depending on your terminal and operating system, run one of the following commands:

* **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
* **Windows (Command Prompt):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
* **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 2. Install Dependencies
Once the virtual environment is active, install the packages defined in [requirements.txt](file:///c:/Users/asus/OneDrive/Documents/Projects/Sanjeevani/requirements.txt):
```bash
pip install -r requirements.txt
```

> [!NOTE]
> The codebase uses the Google GenAI SDK directly in [nodes.py](file:///c:/Users/asus/OneDrive/Documents/Projects/Sanjeevani/src/agent/nodes.py#L7). If you encounter a `ModuleNotFoundError` for `google.genai`, install it manually:
> ```bash
> pip install google-genai
> ```

---

## 🔑 Step 2: Environment Variables Configuration

The project relies on external API keys configured in the [.env](file:///c:/Users/asus/OneDrive/Documents/Projects/Sanjeevani/.env) file located in the root directory. Ensure the following keys are populated:

```env
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here
```

* **GROQ_API_KEY**: Used for the answer generator node running `llama-3.1-8b-instant`.
* **GOOGLE_API_KEY**: Used for the hallucination/groundedness check and query rewriting running `gemini-2.0-flash`.

---

## 🚀 Step 3: Running the Application

There are three key entry points for executing and evaluating the RAG pipeline.

### Option A: Run the Standalone Pipeline (Console Test)
To execute a quick, end-to-end check of the document chunking, graph parsing, hybrid retrieval, and LangGraph workflow:
```bash
python main.py
```
This runs test queries defined in [main.py](file:///c:/Users/asus/OneDrive/Documents/Projects/Sanjeevani/main.py) and prints the self-healing telemetry and final response directly in your terminal.

### Option B: Launch the Interactive Dashboard (Streamlit UI)
To launch the interactive, premium web dashboard that provides a chat interface, real-time Reciprocal Rank Fusion (RRF) metrics, and dynamic knowledge graph rendering:
```bash
streamlit run src/ui/app.py
```
After executing, Streamlit will open the dashboard in your default web browser (typically at `http://localhost:8501`).

### Option C: Run Ragas Evaluation Benchmarks
To run the automated validation suite comparing system outputs against the golden dataset truths:
```bash
python src/evaluation/run_eval.py
```
This generates and saves metrics (Groundedness/Faithfulness, Answer Relevancy, Context Precision, and Context Recall) to `data/processed/eval_results.json`.

---

## 📂 Project Structure Map

* [main.py](file:///c:/Users/asus/OneDrive/Documents/Projects/Sanjeevani/main.py) - Entry point for the console test pipeline.
* [src/ui/app.py](file:///c:/Users/asus/OneDrive/Documents/Projects/Sanjeevani/src/ui/app.py) - Streamlit dashboard application script.
* [src/evaluation/run_eval.py](file:///c:/Users/asus/OneDrive/Documents/Projects/Sanjeevani/src/evaluation/run_eval.py) - Ragas automated benchmarking test runner.
* [requirements.txt](file:///c:/Users/asus/OneDrive/Documents/Projects/Sanjeevani/requirements.txt) - List of application requirements.
* [.env](file:///c:/Users/asus/OneDrive/Documents/Projects/Sanjeevani/.env) - Credentials configuration file.
