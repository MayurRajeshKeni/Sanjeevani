import os
import re
from typing import List, Dict, Any, Tuple

class MarkdownGraphParser:
    """Parses Markdown files into a hierarchical graph representation with Nodes and Edges."""
    
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """Parses a single Markdown file into nodes and edges.
        
        Args:
            file_path: Absolute or relative path to the markdown file.
            
        Returns:
            A dictionary with "nodes" and "edges" lists.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        filename = os.path.basename(file_path)
        
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        
        # Create a virtual root node for the file
        root_id = f"file:{filename}"
        root_node = {
            "id": root_id,
            "title": filename,
            "level": 0,
            "content": "",
            "source_file": file_path
        }
        nodes.append(root_node)
        
        # Keep track of active headers (level, title, node_id)
        # Root is level 0
        active_headers: List[Tuple[int, str, str]] = [(0, filename, root_id)]
        current_node = root_node
        
        in_code_block = False
        
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        for line in lines:
            stripped_line = line.strip()
            
            # Check for code blocks to prevent parsing comments/headers inside them
            if stripped_line.startswith("```"):
                in_code_block = not in_code_block
                current_node["content"] += line
                continue
                
            if in_code_block:
                current_node["content"] += line
                continue
                
            # Check if line is a header (1 to 6 hash signs)
            header_match = re.match(r"^(#{1,6})\s+(.*)$", line)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                # Resolve parent node
                # Pop headers until the top of the stack is at a higher hierarchical level (smaller integer level)
                while len(active_headers) > 0 and active_headers[-1][0] >= level:
                    active_headers.pop()
                    
                parent_id = active_headers[-1][2] if active_headers else root_id
                
                # Generate unique path-based node ID to prevent duplicate name collisions
                path_prefix = "#".join([h[1] for h in active_headers if h[0] > 0])
                if path_prefix:
                    node_id = f"{filename}#{path_prefix}#{title}"
                else:
                    node_id = f"{filename}#{title}"
                    
                # Create and track the new node
                new_node = {
                    "id": node_id,
                    "title": title,
                    "level": level,
                    "content": "",
                    "source_file": file_path
                }
                nodes.append(new_node)
                
                # Form edge from immediate parent to current header
                edges.append({
                    "source": parent_id,
                    "target": node_id,
                    "relation": "parent-to-child"
                })
                
                # Push current header to stack
                active_headers.append((level, title, node_id))
                current_node = new_node
            else:
                # Add line to content of the current node
                current_node["content"] += line
                
        # Clean up leading and trailing whitespace for all node contents
        for node in nodes:
            node["content"] = node["content"].strip()
            
        return {"nodes": nodes, "edges": edges}

    def parse_directory(self, directory_path: str) -> Dict[str, Any]:
        """Parses all Markdown files in a directory and merges their graphs.
        
        Args:
            directory_path: Path to the target directory.
            
        Returns:
            A dictionary with merged "nodes" and "edges" lists.
        """
        all_nodes: List[Dict[str, Any]] = []
        all_edges: List[Dict[str, Any]] = []
        
        if not os.path.exists(directory_path):
            return {"nodes": [], "edges": []}
            
        for root, dirs, files in os.walk(directory_path):
            # Prune virtual environments and hidden directories in-place to avoid scanning them
            dirs[:] = [d for d in dirs if d not in ('.git', '.venv', 'venv', '.gemini', '.system_generated', '__pycache__')]
            for file in files:
                if file.lower().endswith(".md"):
                    file_path = os.path.join(root, file)
                    graph = self.parse_file(file_path)
                    all_nodes.extend(graph["nodes"])
                    all_edges.extend(graph["edges"])
                    
        return {"nodes": all_nodes, "edges": all_edges}
