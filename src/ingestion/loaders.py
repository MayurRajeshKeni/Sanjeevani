import os
from typing import List
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader

def load_document(file_path: str) -> List[Document]:
    """Loads a single document based on its extension (PDF, TXT, MD).
    
    Args:
        file_path: Path to the target document.
        
    Returns:
        List of LangChain Document objects.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            return loader.load()
        elif ext in (".txt", ".md"):
            loader = TextLoader(file_path, encoding="utf-8")
            return loader.load()
        else:
            # Attempt to load as text anyway
            loader = TextLoader(file_path, encoding="utf-8")
            return loader.load()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def load_directory(directory_path: str) -> List[Document]:
    """Recursively loads all PDF, TXT, and MD files from a directory, skipping virtual envs and system folders.
    
    Args:
        directory_path: Path to the target directory.
        
    Returns:
        List of LangChain Document objects.
    """
    documents = []
    if not os.path.exists(directory_path):
        return []
        
    for root, dirs, files in os.walk(directory_path):
        # Prune virtual environments and hidden directories in-place to avoid scanning them
        dirs[:] = [d for d in dirs if d not in ('.git', '.venv', 'venv', '.gemini', '.system_generated', '__pycache__')]
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in (".pdf", ".txt", ".md"):
                file_path = os.path.join(root, file)
                documents.extend(load_document(file_path))
    return documents
