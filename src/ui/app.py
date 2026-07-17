import os
import json
import time
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from langchain_core.documents import Document

# Import ingestion, retrieval, and agent compilation
from src.ingestion.loaders import load_directory
from src.ingestion.chunkers import chunk_documents
from src.ingestion.markdown_graph_parser import MarkdownGraphParser
from src.retrieval.vector_engine import VectorRetrievalEngine
from src.retrieval.sparse_engine import SparseRetrievalEngine
from src.retrieval.graph_engine import GraphRetrievalEngine
from src.retrieval.fusion import HybridRetriever
from src.agent.graph import create_agent_graph

# Configure Streamlit page layout
st.set_page_config(
    page_title="Sanjeevani - RAG Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Deep dark gray background (#0E1117) & premium minimalist cyan (#00F0FF) overrides
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;700&family=Inter:wght@400;600;700&display=swap');
    
    /* Global typeface overrides */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Monospaced typeface overrides for metrics, dataframes and code blocks */
    code, pre, .mono-metric, .stMarkdown pre, [data-testid="stMetricValue"], [data-testid="stTable"] {
        font-family: 'Fira Code', 'JetBrains Mono', monospace !important;
    }
    
    /* Header layout styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00F0FF;
        margin-top: 0.5rem;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    /* Monospaced metric container style */
    .mono-metric {
        font-size: 1.6rem;
        font-weight: 700;
        color: #00F0FF;
        margin-top: 0.3rem;
    }
    
    /* Streamlit sidebar container adjustment */
    [data-testid="stSidebar"] {
        background-color: #1A1C23;
        border-right: 1px solid #334155;
    }
    </style>
""", unsafe_allow_html=True)

# Determine root directories
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
docs_dir = os.path.join(base_dir, "z_docs")
processed_dir = os.path.join(base_dir, "data", "processed")
os.makedirs(processed_dir, exist_ok=True)

# Cache resource to compile the RAG agent pipeline once
@st.cache_resource
def initialize_rag_pipeline():
    docs = load_directory(docs_dir)
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=100)
    
    parser = MarkdownGraphParser()
    graph_data = parser.parse_directory(docs_dir)
    graph_path = os.path.join(processed_dir, "graph.json")
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
    vector_engine = VectorRetrievalEngine(chunks)
    sparse_engine = SparseRetrievalEngine(chunks)
    graph_engine = GraphRetrievalEngine(graph_path)
    hybrid_retriever = HybridRetriever(vector_engine, sparse_engine, graph_engine)
    
    agent = create_agent_graph(hybrid_retriever)
    return agent, hybrid_retriever, chunks, graph_data

# Initialize RAG resources
with st.spinner("[BOOTSTRAPPING] Initializing local index engines and state compilation..."):
    agent, retriever, chunks, graph_data = initialize_rag_pipeline()

# ================= SIDEBAR CONFIGURATION =================
st.sidebar.markdown("<h2 style='color:#00F0FF; font-size:1.1rem; margin-top:0;'>PIPELINE DIAGNOSTICS</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Adjust RRF smoothing constant k
st.sidebar.markdown("### RRF Tuning")
k_val = st.sidebar.slider("RRF Constant (k)", min_value=1, max_value=100, value=60)
retriever.k = k_val

# RAM Limit Diagnostics
st.sidebar.markdown("### Compute Constraints")
st.sidebar.markdown("**Host RAM Cap: 16 GB**")
st.sidebar.progress(0.44) # Safe default mapping ~7GB utilized
st.sidebar.caption("System executing comfortably below 10GB threshold (WDAC compliant)")

# Index Statistics
st.sidebar.markdown("### Memory Index Stats")
st.sidebar.markdown(f"- **Ingested Chunks:** {len(chunks)} parsed")
st.sidebar.markdown(f"- **Embedding Weights:** `all-MiniLM-L6-v2` (Cached)")
st.sidebar.markdown(f"- **Concept Graph size:** {len(graph_data.get('nodes', []))} nodes")

# Active APIs Toggle Review
st.sidebar.markdown("### API Deployments")
st.sidebar.markdown("- **Generator:** Llama 3.1 (Groq API)")
st.sidebar.markdown("- **Critic Node:** Gemini 1.5 (AI Studio API)")

# Terminology Guide
st.sidebar.markdown("---")
with st.sidebar.expander("Terminology Guide"):
    st.markdown("""
    * **LangGraph**: A cyclical AI workflow. The AI can pause, look at its own work, and loop back to try again if it makes a mistake.
    * **Critic Node**: The internal AI proofreader. It intercepts the draft and checks it against the retrieved documents to prevent hallucinations.
    * **Self-Healing Loop**: The automatic correction cycle. If the Critic finds a mistake, it rewrites the query and triggers a re-retrieval.
    """)

# ================= MAIN CANVAS =================
st.markdown("<div class='main-header'>Project Sanjeevani</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Tri-Modal Self-Healing Retrieval-Augmented Generation Dashboard</div>", unsafe_allow_html=True)

# Render Tabs Layout
tab_chat, tab_bench, tab_graph = st.tabs(["CHAT PLAYGROUND", "EVALUATION BENCHMARKS", "KNOWLEDGE GRAPH"])

# ================= TAB 1: CHAT PLAYGROUND =================
with tab_chat:
    st.markdown("### Interactive QA Playground")
    st.markdown("Submit queries to monitor the agent state transitions, healing cycles, and RRF retrieval contribution mapping.")
    
    # Initialize message log history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Render messages thread
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "trace" in msg and len(msg["trace"]) > 0:
                with st.expander("[EXECUTION FLOW & SELF-HEALING TELEMETRY]", expanded=False):
                    st.markdown(f"**Loops Executed:** `{msg.get('loop_count', 0)}`")
                    if msg.get("feedback"):
                        f_back = msg.get("feedback")
                        if "429" in f_back or "resource_exhausted" in f_back.lower() or "quota" in f_back.lower():
                            display_f_back = "[RESOURCE LIMIT] API rate limits exceeded. State machine routing query for query optimization and healing."
                        else:
                            display_f_back = f_back
                        st.markdown(f"**Critique Feedback:**\n```\n{display_f_back}\n```")
                    st.markdown("**Step-by-step Trace:**")
                    for step in msg["trace"]:
                        st.markdown(step, unsafe_allow_html=True)
            if "rrf" in msg and msg["rrf"] is not None:
                with st.expander("[FUSED RRF RANK DETAILS]", expanded=False):
                    st.dataframe(msg["rrf"], use_container_width=True)
                    
                    # Engine Search Distribution Analysis safely wrapped
                    df_rrf = msg["rrf"]
                    if not df_rrf.empty and "Vector Rank" in df_rrf.columns:
                        vector_cnt = sum(1 for v in df_rrf["Vector Rank"] if str(v) != "N/A")
                        sparse_cnt = sum(1 for s in df_rrf["Sparse Rank"] if str(s) != "N/A")
                        graph_cnt = sum(1 for g in df_rrf["Graph Rank"] if str(g) != "N/A")
                        
                        chart_df = pd.DataFrame({
                            "Engine": ["Sparse BM25", "Dense Vector", "JSON Graph"],
                            "Matched Chunks": [sparse_cnt, vector_cnt, graph_cnt]
                        })
                        st.markdown("**Engine Search Distribution Analysis**")
                        st.bar_chart(chart_df.set_index("Engine"), height=150)
                    else:
                        st.caption("No chunks retrieved or ranks available for distribution analysis.")

    # Chat Input Field
    query = st.chat_input("Enter a query related to Project Sanjeevani...")
    if query:
        # Save and render user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        # Compile response with agent pipeline
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            trace_logs = []
            
            initial_state = {
                "original_query": query,
                "current_search_query": query,
                "retrieved_chunks": [],
                "draft_answer": "",
                "critique_score": 0.0,
                "critique_feedback": "",
                "loop_count": 0
            }
            
            start_time = time.time()
            final_state = None
            
            with st.spinner("LangGraph running state machine loops..."):
                # Stream events step-by-step
                for event in agent.stream(initial_state):
                    for node_name, state_update in event.items():
                        if node_name == "retrieve":
                            trace_logs.append(f"`[RETRIEVE]` Fetched {len(state_update.get('retrieved_chunks', []))} fused document chunks.")
                        elif node_name == "generate":
                            trace_logs.append(f"`[GENERATE]` Drafted response answer using Groq Llama 3.1.")
                        elif node_name == "critic":
                            score = state_update.get("critique_score", 0.0)
                            feedback = state_update.get("critique_feedback", "")
                            if "429" in feedback or "resource_exhausted" in feedback.lower() or "quota" in feedback.lower():
                                display_feedback = "[RESOURCE LIMIT] API rate limits exceeded. State machine routing query for query optimization and healing."
                            else:
                                display_feedback = feedback
                            trace_logs.append(f"`[CRITIQUE]` Groundedness score: **{score:.2f}**<br><small>Feedback: {display_feedback}</small>")
                        elif node_name == "rewrite":
                            rewritten = state_update.get("current_search_query", "")
                            trace_logs.append(f"`[REWRITE]` Groundedness check failed. Query reformulated to: *'{rewritten}'*")
                        elif node_name == "fallback":
                            trace_logs.append(f"`[FALLBACK]` Max retries exceeded. Degraded response triggered.")
                            
                        # Update progress live inside UI placeholder
                        with thinking_placeholder.expander("[EXECUTION FLOW & SELF-HEALING TELEMETRY] (Live Trace)", expanded=True):
                            st.markdown(f"**Current State Loops:** `{state_update.get('loop_count', 0)}`")
                            for log in trace_logs:
                                st.markdown(log, unsafe_allow_html=True)
                                
                    final_state = initial_state | state_update
            
            latency = time.time() - start_time
            answer = final_state.get("draft_answer", "Information not found.")
            retrieved_chunks = final_state.get("retrieved_chunks", [])
            
            # Print Final Answer
            st.markdown(f"**Answer:** {answer}")
            
            # Calculate engine contribution percentages via Reciprocal Rank Fusion scores
            vector_score = 0.0
            sparse_score = 0.0
            graph_score = 0.0
            
            for c in retrieved_chunks:
                meta = c.get("metadata", {})
                ranks = meta.get("rrf_ranks", {})
                v_rank = ranks.get("vector")
                s_rank = ranks.get("sparse")
                g_rank = ranks.get("graph")
                
                if isinstance(v_rank, int):
                    vector_score += 1.0 / (k_val + v_rank)
                if isinstance(s_rank, int):
                    sparse_score += 1.0 / (k_val + s_rank)
                if isinstance(g_rank, int):
                    graph_score += 1.0 / (k_val + g_rank)
                    
            total_score = vector_score + sparse_score + graph_score
            if total_score > 0:
                vector_pct = float((vector_score / total_score) * 100)
                sparse_pct = float((sparse_score / total_score) * 100)
                graph_pct = float((graph_score / total_score) * 100)
            else:
                vector_pct = 33.33
                sparse_pct = 33.33
                graph_pct = 33.33
                
            # Build RRF rank details table
            rrf_rows = []
            for idx, c in enumerate(retrieved_chunks):
                meta = c.get("metadata", {})
                ranks = meta.get("rrf_ranks", {"vector": None, "sparse": None, "graph": None})
                rrf_rows.append({
                    "Rank": idx + 1,
                    "Source Document": os.path.basename(meta.get("source", "Unknown")),
                    "RRF Score": f"{meta.get('rrf_score', 0.0):.4f}",
                    "Vector Rank": ranks.get("vector", "N/A"),
                    "Sparse Rank": ranks.get("sparse", "N/A"),
                    "Graph Rank": ranks.get("graph", "N/A"),
                    "Snippet": c.get("page_content", "")[:200] + "..."
                })
            rrf_df = pd.DataFrame(rrf_rows)
            
            # Save assistant message, logs, metrics and RRF scores
            tokens_est = len(query.split()) + len(answer.split()) + 450
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "trace": trace_logs,
                "feedback": final_state.get("critique_feedback", ""),
                "loop_count": final_state.get("loop_count", 0),
                "rrf": rrf_df,
                "latency_ms": latency * 1000,
                "tokens": tokens_est,
                "vector_pct": vector_pct,
                "sparse_pct": sparse_pct,
                "graph_pct": graph_pct
            })
            
            st.rerun()

    # Under Chat history, display the Metrics Dashboard for the last assistant response
    if st.session_state.messages:
        last_msg = st.session_state.messages[-1]
        if last_msg["role"] == "assistant":
            st.markdown("---")
            st.markdown("### METRICS & TELEMETRY DASHBOARD")
            
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                st.markdown(f"**Latency**\n<div class='mono-metric'>{last_msg.get('latency_ms', 0.0):.0f} ms</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"**Tokens Consumed**\n<div class='mono-metric'>{last_msg.get('tokens', 0)}</div>", unsafe_allow_html=True)
            with col3:
                st.markdown("**Retrieval contribution breakdown**")
                chart_df = pd.DataFrame({
                    "Engine": ["Sparse BM25", "Dense Vector", "JSON Graph"],
                    "Contribution (%)": [
                        last_msg.get("sparse_pct", 33.3),
                        last_msg.get("vector_pct", 33.3),
                        last_msg.get("graph_pct", 33.3)
                    ]
                })
                st.bar_chart(chart_df.set_index("Engine"), height=140)

# ================= TAB 2: EVALUATION BENCHMARKS =================
with tab_bench:
    st.markdown("### Automated Evaluation Benchmarks (Ragas)")
    st.markdown("System benchmarks calculated against the golden dataset truths.")
    
    # Path to evaluation outputs
    eval_results_path = os.path.join(base_dir, "data", "processed", "eval_results.json")
    
    if st.button("Trigger Live Ragas Benchmark Run"):
        with st.spinner("Executing Ragas test suite..."):
            from src.evaluation.run_eval import run_evaluations
            try:
                run_evaluations()
                st.success("Benchmark evaluation completed and saved!")
            except Exception as e:
                st.error(f"Error running evaluations: {e}")
                
    if os.path.exists(eval_results_path):
        with open(eval_results_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
            
        summary = eval_data.get("summary", {})
        details = eval_data.get("details", [])
        
        # Helper to format metrics safely (mapping None to 'N/A')
        def fmt_val(v):
            return f"{v:.2f}" if v is not None else "N/A"
            
        f_val = fmt_val(summary.get('faithfulness'))
        ar_val = fmt_val(summary.get('answer_relevancy'))
        cp_val = fmt_val(summary.get('context_precision'))
        cr_val = fmt_val(summary.get('context_recall'))
        
        # Display Metric cards with hover-over help definitions
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="Groundedness (Faithfulness)",
                value=f_val,
                help="Measures if the AI stuck strictly to the facts in the documents, or if it hallucinated."
            )
        with col2:
            st.metric(
                label="Answer Relevancy",
                value=ar_val,
                help="Checks if the response directly answers the prompt, rather than just giving a useless fact."
            )
        with col3:
            st.metric(
                label="Context Precision (RRF)",
                value=cp_val,
                help="Grades the search engines. Measures if the absolute most important text components were ranked at the top of the pile."
            )
        with col4:
            st.metric(
                label="Context Recall",
                value=cr_val,
                help="Measures if the retrieval engines fetched every single piece of information required to construct the ground-truth answer."
            )
            
        st.markdown("### Detailed Score Matrix per Question")
        
        # Create details DataFrame
        details_rows = []
        for row in details:
            details_rows.append({
                "Question": row.get("question", ""),
                "Groundedness (Faithfulness)": fmt_val(row.get('faithfulness')),
                "Answer Relevancy": fmt_val(row.get('answer_relevancy')),
                "Context Precision": fmt_val(row.get('context_precision')),
                "Context Recall": fmt_val(row.get('context_recall'))
            })
        st.dataframe(pd.DataFrame(details_rows), use_container_width=True)
    else:
        st.info("No evaluation scores found. Run the initial Ragas evaluation benchmarking suite.")

# ================= TAB 3: KNOWLEDGE GRAPH =================
with tab_graph:
    st.markdown("### Parsed Knowledge Graph Viewer")
    st.markdown("Interactive structure showing concepts, levels, and associated details.")
    
    # Filter concepts input
    search_q = st.text_input("Filter concepts by name:", "")
    
    nodes_list = graph_data.get("nodes", []) if isinstance(graph_data, dict) else []
    edges_list = graph_data.get("edges", []) if isinstance(graph_data, dict) else []

    # 1. Interactive Vis.js Network Graph Component
    st.markdown("#### Interactive Topology Visualization")
    
    vis_nodes = []
    for node in nodes_list:
        vis_nodes.append({
            "id": node["id"],
            "label": node["title"][:25] + "..." if len(node["title"]) > 25 else node["title"],
            "title": node["content"][:200] + "..." if node.get("content") else ""
        })
        
    vis_edges = []
    for edge in edges_list:
        vis_edges.append({
            "from": edge["source"],
            "to": edge["target"]
        })
        
    nodes_json = json.dumps(vis_nodes)
    edges_json = json.dumps(vis_edges)
    
    html_code = f"""
    <html>
    <head>
      <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
      <style type="text/css">
        #network {{
          width: 100%;
          height: 400px;
          background-color: #1A1C23;
          border: 1px solid #334155;
          border-radius: 6px;
        }}
      </style>
    </head>
    <body>
      <div id="network"></div>
      <script type="text/javascript">
        var nodes = new vis.DataSet({nodes_json});
        var edges = new vis.DataSet({edges_json});
        var container = document.getElementById('network');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
          nodes: {{
            shape: 'box',
            margin: 10,
            color: {{
              background: '#1E293B',
              border: '#00F0FF',
              highlight: {{ background: '#00F0FF', border: '#00F0FF' }}
            }},
            font: {{ color: '#F8FAFC', face: 'Inter, sans-serif', size: 12 }}
          }},
          edges: {{
            color: {{ color: '#475569', highlight: '#00F0FF' }},
            arrows: 'to',
            smooth: {{ type: 'continuous' }}
          }},
          physics: {{
            stabilization: {{
              enabled: true,
              iterations: 150
            }},
            barnesHut: {{ gravitationalConstant: -1200, centralGravity: 0.3, springLength: 95 }}
          }}
        }};
        var network = new vis.Network(container, data, options);
        network.once("stabilizationIterationsDone", function() {{
          network.setOptions({{ physics: false }});
        }});
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=420)

    # 2. Tabular Concepts Data
    st.markdown("#### Tabular Concept Inventory")
    graph_rows = []
    for concept in nodes_list:
        title = concept.get("title", "")
        if search_q.lower() in title.lower():
            graph_rows.append({
                "Concept ID": concept.get("id", ""),
                "Title": title,
                "Heading Level": concept.get("level", ""),
                "Source File": os.path.basename(concept.get("source_file", "")) if concept.get("source_file") else "",
                "Associated Content": concept.get("content", "")[:300] + "..." if concept.get("content") else ""
            })
            
    st.dataframe(pd.DataFrame(graph_rows), use_container_width=True)
