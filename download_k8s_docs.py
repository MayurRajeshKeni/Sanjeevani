import os
import shutil
import subprocess
from pathlib import Path

# --- Configuration ---
REPO_URL = "https://github.com/kubernetes/website.git"
SPARSE_DIR = "content/en/docs"
TARGET_DIR = os.path.join(os.getcwd(), "z_docs", "kubernetes_docs")
TEMP_DIR = os.path.join(os.getcwd(), "temp_k8s_clone")

def run_command(cmd, cwd=None):
    """Executes a shell command safely."""
    subprocess.run(cmd, cwd=cwd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def extract_markdown():
    print("🚀 Initializing Sparse Checkout of Kubernetes Repository...")
    
    # 1. Clean up old runs
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TARGET_DIR, exist_ok=True)

    try:
        # 2. Clone ONLY the repository structure (no file contents yet)
        print("📦 Cloning repository metadata (this takes a few seconds)...")
        run_command(f"git clone --filter=blob:none --sparse {REPO_URL} {TEMP_DIR}")

        # 3. Tell Git to download ONLY the documentation folder
        print(f"🎯 Fetching specific directory: {SPARSE_DIR}...")
        run_command(f"git sparse-checkout set {SPARSE_DIR}", cwd=TEMP_DIR)

        # 4. Walk the directory and extract ONLY .md files
        print("📂 Moving Markdown files to z_docs...")
        md_count = 0
        source_path = os.path.join(TEMP_DIR, *SPARSE_DIR.split('/'))
        
        for root, _, files in os.walk(source_path):
            for file in files:
                if file.endswith(".md"):
                    source_file = os.path.join(root, file)
                    # Create a flat filename structure so we don't have deeply nested folders
                    relative_path = os.path.relpath(source_file, source_path)
                    flat_filename = relative_path.replace(os.sep, "_")
                    target_file = os.path.join(TARGET_DIR, flat_filename)
                    
                    shutil.copy2(source_file, target_file)
                    md_count += 1

        print(f"✅ Success! Extracted {md_count} Markdown files to {TARGET_DIR}")

    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        
    finally:
        # 5. Clean up the massive temp git folder
        if os.path.exists(TEMP_DIR):
            print("🧹 Cleaning up temporary git files...")
            # On Windows, git objects can be read-only, requiring a special remove handler
            def remove_readonly(func, path, _):
                os.chmod(path, 0o777)
                func(path)
            shutil.rmtree(TEMP_DIR, onerror=remove_readonly)
            print("✨ Done!")

if __name__ == "__main__":
    extract_markdown()
