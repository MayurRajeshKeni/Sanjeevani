import os
import json
from src.ingestion.loaders import load_directory, load_document
from src.ingestion.chunkers import chunk_documents
from src.ingestion.markdown_graph_parser import MarkdownGraphParser

def run_pipeline() -> None:
    # Determine base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    print("--- Phase 1: Ingestion, Chunking & Graph Parsing ---")
    
    # 1. Load markdown files from the workspace root (where Overview.md etc. are located)
    print("Loading documents from workspace root...")
    docs = load_directory(base_dir)
    print(f"Loaded {len(docs)} document pages/files.")
    
    # 2. Chunk documents using RecursiveCharacterTextSplitter
    print("Chunking documents...")
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=100)
    print(f"Created {len(chunks)} chunks.")
    
    # Save chunks to data/processed/chunks.json for validation
    chunks_path = os.path.join(processed_dir, "chunks.json")
    chunks_data = [
        {
            "page_content": chunk.page_content,
            "metadata": chunk.metadata
        }
        for chunk in chunks
    ]
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    print(f"Saved chunks to {chunks_path}")
    
    # 3. Parse markdown documents to Graph JSON structure
    print("Parsing markdown files to graph...")
    parser = MarkdownGraphParser()
    graph_data = parser.parse_directory(base_dir)
    
    # Save graph to data/processed/graph.json for validation
    graph_path = os.path.join(processed_dir, "graph.json")
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    print(f"Saved graph JSON to {graph_path}")
    print(f"Graph stats: {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges.")
    
if __name__ == "__main__":
    run_pipeline()
