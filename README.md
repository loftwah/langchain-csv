# Product Catalog RAG System

A Retrieval-Augmented Generation (RAG) system that answers natural language questions about product data using local LLMs. This project demonstrates how to build an interactive product catalog explorer using LangChain, Ollama, and Gradio.

## Features

- 🔍 **Interactive Web Interface**: A modern, user-friendly Gradio interface for exploring product data
- 💬 **Command-line Demo**: A colorful CLI interface with multiple demo modes
- 🧠 **Hybrid Analysis**: Combines RAG with direct data analysis for precise queries
- 📊 **Rich Data Support**: Handles complex product data including prices, categories, ratings, stock levels, release dates, discounts, and features
- 🔄 **Real-time Processing**: Get answers in seconds using local LLMs
- 🎯 **Specialized Queries**: Advanced handling for price comparisons, discount detection, feature searches, and inventory status

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

Sample questions you can ask:

- "What's the cheapest laptop?"
- "Show me products under $100"
- "Which products have discounts?"
- "What are the newest products?"
- "Which headphones have noise cancellation?"
- "Compare Apple and Samsung products"

### Command-line Demo

Run the CLI demo:

```bash
uv run rag_demo.py
```

The CLI demo offers three modes:

1. **Sample Queries**: See the system answer preset questions in categories like Price & Value, Brands & Categories, Ratings & Features, and Availability & Release
2. **Interactive Mode**: Ask your own questions about the product catalog
3. **Behind the Scenes**: Learn how RAG works with a step-by-step walkthrough

## How It Works

This system uses a hybrid approach to answer questions about product data:

1. **Vector Embeddings**: The system creates mathematical representations of your product data
2. **Semantic Search**: When you ask a question, it finds the most relevant products
3. **LLM Generation**: It uses a language model to create a natural language answer

### Smart Direct Data Analysis

For specific types of queries, the system uses direct data analysis instead of relying solely on the LLM:

- **Price Queries**: Finds cheapest/most expensive products, products in specific price ranges
- **Rating Analysis**: Identifies highest-rated products by category or overall
- **Discount Detection**: Analyzes products on sale with discount percentages
- **Inventory Status**: Reports on stock levels and availability
- **Release Date Analysis**: Identifies newest products or products from specific time periods
- **Feature Search**: Finds products with specific features mentioned in the query

This hybrid approach ensures accurate, detailed answers while maintaining the flexibility of natural language understanding.

## Product Data Format

The system works best with CSV files containing product information. The enhanced version supports the following fields:

- **Basic Fields**: name, price, description, category, brand, rating
- **Advanced Fields**: stock, release_date, discount_percent, features

Example row:

```
"MacBook Pro M3",1799,"Powerful laptop with M3 chip...",Laptop,Apple,4.9,15,2023-11-07,0,"AI-optimized,Thunderbolt 4,120Hz display"
```

## Known Limitations

1. **Numerical Analysis**:

   - Pure RAG approaches may struggle with precise numerical comparisons
   - The system uses direct data analysis for data-specific queries to ensure accuracy
   - Complex multi-step numerical reasoning may still be challenging

2. **LLM Limitations**:

   - Local LLMs may have limited context window sizes
   - The quality of answers depends on the richness of your product data

3. **Data Requirements**:
   - The system works best with structured CSV data
   - Missing or inconsistent data may affect answer quality
   - Large datasets may require more processing time

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request to the [GitHub repository](https://github.com/loftwah/langchain-csv).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
