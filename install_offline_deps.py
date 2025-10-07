#!/usr/bin/env python3
"""
Installation script for offline recommendation system dependencies.
Run this to enable sentence-BERT and RAG capabilities.
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    print("🚀 Installing offline recommendation system dependencies...")
    print("=" * 60)
    
    packages = [
        "langchain>=0.2.16",
        "langchain-community>=0.2.16", 
        "llama-index-core>=0.14.3,<0.15",
        "llama-index-llms-huggingface>=0.1.5",
        "llama-index-embeddings-huggingface>=0.1.5",
        "sentence-transformers>=2.6.0",
        "faiss-cpu>=1.7.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "torch>=2.0.0"
    ]
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print("=" * 60)
    print(f"📊 Installation Summary: {success_count}/{len(packages)} packages installed successfully")
    
    if success_count == len(packages):
        print("🎉 All dependencies installed! Your app now has full offline capabilities.")
        print("Run: streamlit run recommendation_app.py")
    else:
        print("⚠️  Some packages failed to install. The app will work with basic TF-IDF recommendations.")
        print("You can still run: streamlit run recommendation_app.py")

if __name__ == "__main__":
    main()

