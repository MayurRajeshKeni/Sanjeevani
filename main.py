import os
import json
from src.ingestion.loaders import load_directory, load_document
from src.ingestion.chunkers import chunk_documents
from src.ingestion.markdown_graph_parser import MarkdownGraphParser
from src.retrieval.vector_engine import VectorRetrievalEngine
from src.retrieval.sparse_engine import SparseRetrievalEngine
from src.retrieval.graph_engine import GraphRetrievalEngine
from src.retrieval.fusion import HybridRetriever

def run_pipeline() -> None:
    # Determine base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    print("\n--- Phase 1: Ingestion, Chunking & Graph Parsing ---")
    
    # 1. Load markdown files from the workspace root (specifically z_docs/)
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
    
    print("\n--- Phase 2: Tri-Modal Retrieval & Reciprocal Rank Fusion (RRF) ---")
    
    # Initialize the three retrieval engines
    vector_engine = VectorRetrievalEngine(chunks)
    sparse_engine = SparseRetrievalEngine(chunks)
    graph_engine = GraphRetrievalEngine(graph_path)
    
    # Initialize the RRF Hybrid Retriever
    hybrid_retriever = HybridRetriever(vector_engine, sparse_engine, graph_engine)
    
    # Run test queries to verify fusion ranking and breakdowns
    test_queries = [
        "What is the exact memory limitation of the local host machine?",
        "Explain the Reciprocal Rank Fusion RRF mathematical formula.",
        "What frontend framework is chosen and what is the color palette?"
    ]
    
    for query in test_queries:
        print(f"\n==================================================")
        print(f"QUERY: '{query}'")
        print(f"==================================================")
        
        retrieved_docs = hybrid_retriever.retrieve(query, top_k=3)
        
        for idx, doc in enumerate(retrieved_docs):
            source_file = os.path.basename(doc.metadata.get("source", "Unknown"))
            rrf_score = doc.metadata.get("rrf_score", 0.0)
            ranks = doc.metadata.get("rrf_ranks", {})
            print(f"\n[HIT {idx+1}] Source: {source_file}")
            print(f"RRF Score: {rrf_score:.6f}")
            print(f"Rank Contributions -> Vector: {ranks.get('vector')}, Sparse: {ranks.get('sparse')}, Graph: {ranks.get('graph')}")
            content_snippet = doc.page_content[:200].replace('\n', ' ')
            print(f"Snippet: {content_snippet}...")

if __name__ == "__main__":
    run_pipeline()
