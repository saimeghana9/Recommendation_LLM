# Enhanced Cross-Domain Recommendation System

A Streamlit app that provides recommendations across movies, TV shows, music, books, and food using multiple AI approaches - all running offline without API keys!

## Features

### 🎯 Multiple Recommendation Methods
- **Sentence-BERT**: Semantic similarity using local embeddings
- **TF-IDF**: Traditional text-based matching (fallback)
- **RAG**: Cross-domain conversational queries

### 🔒 Fully Offline
- No API keys required
- Uses local HuggingFace models
- Works without internet connection (after initial model download)

### 🎬 Multi-Domain Support
- Movies & TV Shows
- Music & Books  
- Food & Recipes
- Cross-domain queries

## Quick Start

### Option 1: Basic Installation (TF-IDF only)
```bash
pip install streamlit pandas numpy scikit-learn
streamlit run recommendation_app.py
```

### Option 2: Full Offline Capabilities
```bash
# Install all dependencies
python install_offline_deps.py

# Or manually:
pip install -r requirements_enhanced.txt

# Run the app
streamlit run recommendation_app.py
```

## Usage Examples

### Domain-Specific Queries
- "Suggest romantic movies with great plots"
- "Find energetic pop songs for working out"
- "Recommend easy vegetarian dinner recipes"
- "What are some classic mystery books?"

### Cross-Domain Queries (requires offline packages)
- "Great comfort recommendations"
- "Iconic classics to explore"
- "Nostalgic picks from the 2000s"

## How It Works

1. **Query Processing**: Detects domain and enhances query with related terms
2. **Sentence-BERT**: Tries semantic similarity first (if available)
3. **TF-IDF Fallback**: Uses traditional text matching if needed
4. **RAG System**: Handles cross-domain conversational queries
5. **Smart Filtering**: Avoids duplicate recommendations

## Data Sources

The app loads data from:
1. GitHub repository (primary)
2. Local CSV files (fallback)
3. Sample data (if neither available)

## Requirements

### Basic (TF-IDF only)
- Python 3.8+
- streamlit
- pandas
- numpy
- scikit-learn

### Enhanced (Full offline capabilities)
- All basic requirements
- langchain
- llama-index-core
- sentence-transformers
- faiss-cpu
- transformers
- torch

## Troubleshooting

### "Offline capabilities not available"
- Install the enhanced dependencies: `python install_offline_deps.py`
- Or install manually: `pip install -r requirements_enhanced.txt`

### "StreamlitAPIException: set_page_config()"
- This is fixed in the current version
- Make sure you're running the latest code

### Slow first load
- Models download on first use (~100MB)
- Subsequent runs are much faster

## Development

The app gracefully degrades:
- With offline packages: Full sentence-BERT + RAG capabilities
- Without offline packages: TF-IDF only (still fully functional)

## License

Developed by Sai Meghana Boyapati

