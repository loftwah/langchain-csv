# CSV-based RAG System

This project implements a Retrieval-Augmented Generation (RAG) system that answers questions about data stored in CSV files using local LLMs through Ollama.

## What It's Doing

1. **Vector Embedding Creation**:

   - The system reads your CSV product data
   - For each product, it creates a mathematical representation (vector embedding) using Ollama's language model
   - These embeddings capture the semantic meaning of each product's attributes

2. **Vector Database Construction**:

   - It stores these embeddings in a FAISS vector database
   - FAISS is optimized for fast similarity searching

3. **Semantic Retrieval**:

   - When a user asks a question like "What's the cheapest product?"
   - The query is converted to the same vector space
   - FAISS finds products whose embeddings are most similar to the query

4. **LLM-Based Answer Generation**:
   - The system feeds the relevant product information to the LLM (Ollama)
   - The LLM generates a natural language answer based on the retrieved context

## What It Achieves

The system enables natural language querying of structured data. For example:

- "What's the cheapest product?" → The system understands this requires comparing prices and returns "Wireless Mouse at $25"
- "Tell me about electronics products" → It can identify relevant products and summarize them
- "Which laptop has the best rating?" → It compares ratings specifically among laptops
- "What products are made by Apple?" → It filters by brand

## Why This Matters

1. **Natural Language Interface**: Users can ask questions in plain English instead of using SQL or filters.

2. **Semantic Understanding**: Unlike traditional keyword search:

   - "Affordable computing options" would still find "Budget Laptop" even though "affordable" isn't in the data
   - It understands that "earphones" relates to "Bluetooth Earbuds"

3. **Contextual Answers**: Instead of just returning data rows, it generates human-readable responses.

4. **Local Privacy**: By using Ollama, all processing happens on your machine with no data sent to external APIs.

5. **Flexibility**: This approach works with any tabular data, not just products.

## Why Use LangChain Instead of Direct Context Insertion

When working with structured data like CSV files, you might wonder why you should use LangChain's RAG approach instead of simply inserting the entire CSV content into an LLM's context window. Here's why the LangChain approach is superior:

### Scalability for Large Datasets

- **Context Window Limitations**: LLMs have fixed context windows (typically 4K-128K tokens). A modest 10,000-row CSV would exceed most context limits.
- **Vector Search Efficiency**: LangChain's vector database approach can handle millions of records while only retrieving relevant ones for each query.

### Precision and Relevance

- **Semantic Retrieval**: Only the most relevant documents are provided to the LLM, reducing "hallucinations" and improving answer relevance.
- **Less Noise**: Without retrieval, the model might get distracted by irrelevant data or struggle to find the right information in a large context.

### Computational Efficiency

- **Reduced Token Usage**: Processing only relevant chunks uses fewer tokens than analyzing the entire dataset for every query.
- **Lower Latency**: Retrieving only what's needed results in faster responses, especially for large datasets.

### How LangChain Achieves This

1. **Document Processing**: LangChain breaks your CSV into manageable chunks (documents), preserving row context.

2. **Vectorization Pipeline**: It handles the complex process of converting text to vectors and managing the vector database.

3. **Retrieval Chain Orchestration**: It combines:

   - Query understanding
   - Similarity search
   - Context formatting
   - LLM prompting

4. **Result Management**: It processes the LLM's response and extracts metadata about sources.

This approach allows you to build sophisticated data query systems that would otherwise require complex custom code to handle the retrieval, chunking, and prompt engineering aspects.

## Prerequisites

- Python 3.8+
- Ollama installed and running (https://ollama.com)
- Required Python packages (see Installation)

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/loftwah/langchain-csv.git
   cd langchain-csv
   ```

2. Create and activate a virtual environment:

   ```
   # Create virtual environment with uv
   uv venv

   # Activate the virtual environment
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:

   ```
   # Install with uv
   uv pip install langchain langchain_community langchain_ollama faiss-cpu colorama gradio pandas
   ```

4. Ensure Ollama is installed and running:

   ```
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.com/install.sh | sh

   # Start Ollama server
   ollama serve
   ```

5. Pull the necessary models:
   ```
   ollama pull llama3.2
   ```

## Usage

### Command Line Demo (rag_demo.py)

The `rag_demo.py` script provides a colorful command-line interface for the RAG system.

1. Run the script:

   ```
   uv run rag_demo.py
   ```

2. The script will:
   - Verify Ollama server connection and display available models
   - Load and display the CSV data
   - Create vector embeddings with visual examples
   - Set up a question-answering chain
   - Process sample queries with step-by-step explanations
   - Display the results with source documents

### Web Interface Demo (gradio_demo.py)

The `gradio_demo.py` script provides a user-friendly web interface using Gradio.

1. Run the Gradio web interface:

   ```
   uv run gradio_demo.py
   ```

2. Access the web interface in your browser (typically at http://127.0.0.1:7860).

3. Using the interface:
   - Step 1: Initialize the RAG system by specifying the CSV file path and Ollama model
   - Step 2: Enter your questions or select from sample questions
   - View the answers and the source documents used to generate them

## Customization

### Using Different Models

To use a different Ollama model, you can:

- In `rag_demo.py`: Modify the model name in the `setup_vector_store` and `setup_qa_chain` functions
- In `gradio_demo.py`: Select a different model from the dropdown menu

```python
# Example for rag_demo.py
vector_store = setup_vector_store(docs, model_name="mistral")
qa_chain = setup_qa_chain(retriever, model_name="mistral")
```

### Different CSV Files

To use a different CSV file:

- In `rag_demo.py`: Change the file path in the `load_documents` function call
- In `gradio_demo.py`: Enter the path to your CSV file in the text field

```python
# Example for rag_demo.py
docs = load_documents("your_data.csv")
```

### Custom Queries

- In `rag_demo.py`: Modify the `sample_queries` list in the `main` function
- In `gradio_demo.py`: Enter your queries directly in the web interface

## Troubleshooting

### "Broken Pipe" Error

If you encounter a "broken pipe" error from Ollama, try:

1. Restart the Ollama server:

   ```
   killall ollama
   ollama serve
   ```

2. Use a smaller model if memory is an issue:

   ```
   ollama pull tinyllama
   ```

   Then update the model name in the code or select it from the Gradio dropdown.

3. Check Ollama logs for more details:
   ```
   journalctl -u ollama
   ```

### Memory Issues

If you're experiencing memory issues with large CSV files:

1. Process the CSV in batches
2. Use a more memory-efficient embedding model
3. Consider using disk-based vector stores for larger datasets

## Advanced Features

### Interactive Mode

You can modify the `rag_demo.py` script to accept user queries interactively:

```python
def interactive_mode(qa_chain, retriever):
    print("Enter your questions (type 'exit' to quit):")
    while True:
        query = input("> ")
        if query.lower() in ["exit", "quit"]:
            break
        process_query(qa_chain, query, retriever)

# Add to main function
interactive_mode(qa_chain, retriever)
```

### API Integration

The system can be easily integrated with FastAPI to create a modern, high-performance REST API:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="CSV RAG API")

# Initialize components
docs = load_documents("products.csv")
vector_store = setup_vector_store(docs)
retriever = vector_store.as_retriever()
qa_chain = setup_qa_chain(retriever)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def process_user_query(request: QueryRequest):
    try:
        answer, sources = process_query(qa_chain, request.query, retriever)
        return {
            "answer": answer,
            "sources": [
                {"content": s.page_content, "metadata": s.metadata}
                for s in sources
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn api:app --reload
```

This creates a FastAPI application with automatic OpenAPI documentation and better performance than Flask.

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request to the [GitHub repository](https://github.com/loftwah/langchain-csv).
