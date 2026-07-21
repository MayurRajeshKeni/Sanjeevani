# Project Sanjeevani: Operations & Setup Runbook

This document provides complete, step-by-step instructions to initialize, configure, execute, and benchmark **Project Sanjeevani**—the hybrid Tri-Modal Self-Healing Retrieval-Augmented Generation (RAG) pipeline.

---

## 🛠️ Step 1: Environment Activation & Dependency Setup

The workspace contains a pre-configured Python virtual environment directory (`.venv`). Follow these instructions to activate the environment and install dependencies.

### 1. Activate the Virtual Environment
Execute the command corresponding to your terminal and operating system:

*   **Windows (PowerShell):**
    ```powershell
    .venv\Scripts\Activate.ps1
    ```
*   **Windows (Command Prompt):**
    ```cmd
    .venv\Scripts\activate.bat
    ```
*   **macOS / Linux:**
    ```bash
    source .venv/bin/activate
    ```

*Note: On Windows systems with strict WDAC (Windows Defender Application Control) policies, run python scripts explicitly using system-wide Python executors (e.g. `py -3.11 <script_name>`).*

### 2. Install Project Dependencies
Once the virtual environment is active, install the required packages:
```bash
pip install -r requirements.txt
```

> [!NOTE]
> Sanjeevani calls the Google GenAI SDK directly in the self-healing critique logic (`src/agent/nodes.py`). If you experience a `ModuleNotFoundError` for the `google.genai` namespace, install it manually:
> ```bash
> pip install google-genai
> ```

---

## 🔑 Step 2: Environment Credentials Configuration

Copy or rename the template credentials file to `.env` in the root folder, and populate the following remote cloud API keys:

```env
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here
```

*   **`GROQ_API_KEY`**: Authenticates the high-speed draft answer generator node running `llama-3.1-8b-instant`.
*   **`GOOGLE_API_KEY`**: Authenticates the self-healing critic node and rewriter query optimization steps running `gemini-2.0-flash`.

---

## 📂 Step 3: Scaling & Data Ingestion (Kubernetes Website Docs)

To test the retrieval latency, RRF ranking weights, and database scaling capabilities of the Tri-Modal pipeline with real-world technical reference manuals, you can load external datasets.

### 1. Download the Kubernetes English Documentation
Run the helper download script. It performs a sparse git clone to download ONLY the markdown source directory from the Kubernetes website repository, saving it in a flat-name layout to avoid nested directories:
```bash
python download_k8s_docs.py
```
This clones all Kubernetes English markdown guides to `z_docs/kubernetes_docs/`.

### 2. Force Index Invalidation (Re-indexing New Files)
The dashboard and console indexers cache database files to improve startup speeds. If you have downloaded new manual documents or edited files in `z_docs/` and want to force the pipeline to fully rebuild the Vector, Lexical, and Graph indexes, delete the processed database folder:

*   **Windows (cmd):**
    ```cmd
    rmdir /s /q data\processed
    ```
*   **Windows (PowerShell):**
    ```powershell
    Remove-Item -Recurse -Force data\processed
    ```
*   **macOS / Linux:**
    ```bash
    rm -rf data/processed
    ```
The next time you start the app, the pipeline will automatically re-ingest all files in `z_docs/` and build clean databases.

---

## 🚀 Step 4: Running the Application

There are three key entry points for running the Sanjeevani pipeline.

### Option A: Run the Standalone Pipeline (Console Diagnostic)
To run a fast, end-to-end check of the ingestion loaders, indexing, RRF retrieval, and LangGraph self-healing routing state machine in the console:
```bash
python main.py
```
This runs pre-defined queries and outputs live node trace telemetry directly in your terminal.

### Option B: Launch the Interactive Dashboard (Streamlit UI)
To launch the interactive, dark-themed Streamlit dashboard with real-time chatbot playgrounds, live LangGraph trace logging, telemetry bar charts, and interactive Vis.js knowledge graphs:
```bash
streamlit run src/ui/app.py
```
Streamlit will launch and bind to `http://localhost:8501`.

### Option C: Run Ragas Evaluation Benchmarks
To execute the automated evaluation benchmark suite checking system accuracy metrics against the Golden Dataset:
```bash
python src/evaluation/run_eval.py
```
This runs 20 metric runs across the test queries and saves results to `data/processed/eval_results.json`.

---

## 🩺 Step 5: Troubleshooting & Diagnostics

### 1. Quota Exhaustion (HTTP 429 Errors)
If you hit Gemini or Groq rate limits, the Critic node automatically defaults to a Groundedness score of `0.00` and displays:
`[RESOURCE LIMIT] API rate limits exceeded. State machine routing query for query optimization and healing.`
*Solution:* Wait 60 seconds for the token-per-minute (TPM) or request-per-minute (RPM) windows to reset.

### 2. Port Conflict (Port 8501 Already in Use)
If launching Streamlit fails because port 8501 is taken:
```bash
streamlit run src/ui/app.py --server.port 8502
```

### 3. Local RAM Footprint Constraints
The local vector and sparse search engines run entirely in memory. To ensure overall system stability on 16GB host RAM limit systems, the local index footprint is mapped below 10GB. Verify local memory usage in the Streamlit Sidebar diagnostics panel.
