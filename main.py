import os
import json
from dotenv import load_dotenv
from src.ingestion.loaders import load_directory, load_document
from src.ingestion.chunkers import chunk_documents
from src.ingestion.markdown_graph_parser import MarkdownGraphParser
from src.retrieval.vector_engine import VectorRetrievalEngine
from src.retrieval.sparse_engine import SparseRetrievalEngine
from src.retrieval.graph_engine import GraphRetrievalEngine
from src.retrieval.fusion import HybridRetriever
from src.agent.graph import create_agent_graph

# Load environment variables (such as API keys) from .env
load_dotenv()

def run_pipeline() -> None:
    # Determine base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    print("\n--- Phase 1: Ingestion, Chunking & Graph Parsing ---")
    
    # 1. Load markdown files from the z_docs directory
    docs_dir = os.path.join(base_dir, "z_docs")
    print(f"Loading documents from: {docs_dir}")
    docs = load_directory(docs_dir)
    print(f"Loaded {len(docs)} document pages/files.")
    
    # 2. Chunk documents using RecursiveCharacterTextSplitter
    print("Chunking documents...")
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=100)
    print(f"Created {len(chunks)} chunks.")
    
    # Save chunks
    chunks_path = os.path.join(processed_dir, "chunks.json")
    chunks_data = [
        {"page_content": chunk.page_content, "metadata": chunk.metadata}
        for chunk in chunks
    ]
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    print(f"Saved chunks to {chunks_path}")
    
    # 3. Parse markdown documents to Graph JSON structure
    print("Parsing markdown files to graph...")
    parser = MarkdownGraphParser()
    graph_data = parser.parse_directory(docs_dir)
    
    # Save graph
    graph_path = os.path.join(processed_dir, "graph.json")
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    print(f"Saved graph JSON to {graph_path}")
    print(f"Graph stats: {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges.")
    
    print("\n--- Phase 2 & 3: Tri-Modal Retrieval & LangGraph Orchestration ---")
    
    # Initialize the three retrieval engines
    vector_engine = VectorRetrievalEngine(chunks)
    sparse_engine = SparseRetrievalEngine(chunks)
    graph_engine = GraphRetrievalEngine(graph_path)
    
    # Initialize the RRF Hybrid Retriever
    hybrid_retriever = HybridRetriever(vector_engine, sparse_engine, graph_engine)
    
    # Compile the LangGraph agent graph
    print("Compiling self-healing LangGraph agent...")
    agent = create_agent_graph(hybrid_retriever)
    print("Agent compiled successfully!")
    
    # Run test queries to verify agent workflow
    test_queries = [
        "What is the exact memory limitation of the local host machine?",
        "What is the capital of France?"  # Groundedness critique will fail, triggering query rewrite, loop retries, and fallback
    ]
    
    for query in test_queries:
        print(f"\n==================================================")
        print(f"RUNNING AGENT FOR QUERY: '{query}'")
        print(f"==================================================")
        
        initial_state = {
            "original_query": query,
            "current_search_query": query,
            "retrieved_chunks": [],
            "draft_answer": "",
            "critique_score": 0.0,
            "critique_feedback": "",
            "loop_count": 0
        }
        
        # Execute the compiled graph agent
        final_state = agent.invoke(initial_state)
        
        print(f"\n--------------------------------------------------")
        print(f"AGENT EXECUTION SUMMARY:")
        print(f"--------------------------------------------------")
        print(f"Original Query: {final_state['original_query']}")
        print(f"Final Search Query: {final_state['current_search_query']}")
        print(f"Groundedness Score: {final_state['critique_score']:.2f}")
        print(f"Loop Count: {final_state['loop_count']}")
        print(f"Retrieved Chunks Count: {len(final_state['retrieved_chunks'])}")
        print(f"\nFINAL ANSWER:\n{final_state['draft_answer']}")
        print(f"--------------------------------------------------")

if __name__ == "__main__":
    run_pipeline()
