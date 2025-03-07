# CSV-based RAG System

A Retrieval-Augmented Generation (RAG) system that answers questions about product data using local LLMs. This project demonstrates how to build an interactive question-answering system using LangChain, Ollama, and Gradio.

## Features

- 🔍 **Interactive Web Interface**: A modern, user-friendly Gradio interface for asking questions
- 💬 **Command-line Demo**: A colorful CLI interface with multiple demo modes
- 🧠 **Hybrid Approach**: Combines RAG with direct data analysis for accurate numerical queries
- 📊 **Data Visualization**: Preview and explore your CSV data
- 🔄 **Real-time Processing**: Get answers in seconds using local LLMs

## Prerequisites

- Python 3.8+
- Ollama installed and running (https://ollama.com)
- Required Python packages (see Installation)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/loftwah/langchain-csv.git
cd langchain-csv
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment with uv
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
# Install with uv
uv pip install langchain langchain_community langchain_ollama faiss-cpu colorama gradio pandas
```

4. Ensure Ollama is installed and running:
```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server
ollama serve
```

5. Pull the necessary models:
```bash
ollama pull llama3.2
```

## Usage

### Web Interface

Run the Gradio demo:
```bash
uv run gradio_demo.py
```

The web interface will be available at `http://localhost:7860`.

### Command-line Demo

Run the CLI demo:
```bash
uv run rag_demo.py
```

The CLI demo offers three modes:
1. **Sample Queries**: See the system answer preset questions
2. **Interactive Mode**: Ask your own questions
3. **Behind the Scenes**: Learn how RAG works with a step-by-step walkthrough

## How It Works

This system uses a hybrid approach to answer questions about product data:

1. **Vector Embeddings**: The system creates mathematical representations of your product data
2. **Semantic Search**: When you ask a question, it finds the most relevant products
3. **LLM Generation**: It uses a language model to create a natural language answer

### Special Handling for Numerical Queries

For specific types of queries, the system uses direct data analysis instead of relying solely on the LLM:

- Price comparisons (e.g., "What's the cheapest product?")
- Numerical rankings (e.g., "Which product has the highest rating?")
- Price range queries (e.g., "What products cost less than $50?")

This hybrid approach ensures accurate numerical answers while maintaining the flexibility of natural language understanding for other types of queries.

## Known Limitations

1. **Numerical Analysis**:
   - Pure RAG approaches may struggle with precise numerical comparisons
   - The system uses direct data analysis for price-related queries to ensure accuracy
   - Complex multi-step numerical reasoning may still be challenging

2. **LLM Limitations**:
   - Local LLMs may have limited context window sizes
   - Complex reasoning tasks may produce incorrect results
   - The quality of answers depends on the richness of your product data

3. **Data Requirements**:
   - The system works best with structured CSV data
   - Missing or inconsistent data may affect answer quality
   - Large datasets may require more processing time

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request to the [GitHub repository](https://github.com/loftwah/langchain-csv).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
