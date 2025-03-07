from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import requests
import logging
import os
import time
import pandas as pd
from colorama import Fore, Style, init

# Initialize colorama
init()

# Setup logging with colors
class ColorFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: Fore.CYAN + "%(message)s" + Style.RESET_ALL,
        logging.INFO: "%(message)s",
        logging.WARNING: Fore.YELLOW + "%(message)s" + Style.RESET_ALL,
        logging.ERROR: Fore.RED + "%(message)s" + Style.RESET_ALL,
        logging.CRITICAL: Fore.RED + Style.BRIGHT + "%(message)s" + Style.RESET_ALL
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Setup custom logger
logger = logging.getLogger("rag_demo")
logger.setLevel(logging.INFO)

# Remove existing handlers
for handler in logger.handlers:
    logger.removeHandler(handler)

# Create console handler with custom formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColorFormatter())
logger.addHandler(console_handler)

def print_header(text):
    """Print a formatted header"""
    print(f"\n{Fore.GREEN}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'=' * 3}{Style.RESET_ALL} {Fore.CYAN}{text}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'=' * 80}{Style.RESET_ALL}\n")

def print_step(step_num, description):
    """Print a step in the process"""
    print(f"{Fore.YELLOW}[Step {step_num}]{Style.RESET_ALL} {description}")

def check_ollama_server():
    """Check if Ollama server is running and return available models"""
    print_step(1, "Checking Ollama server status")
    
    try:
        response = requests.get("http://localhost:11434/api/version")
        if response.status_code == 200:
            logger.info(f"✓ Connected to Ollama version: {response.json().get('version')}")
            
            # Get available models
            models_response = requests.get("http://localhost:11434/api/tags")
            if models_response.status_code == 200:
                models = models_response.json().get("models", [])
                if models:
                    model_names = [m.get('name') for m in models]
                    logger.info(f"✓ Available models: {', '.join(model_names)}")
            return True
        else:
            logger.warning(f"✗ Ollama server responded with status code: {response.status_code}")
    except Exception as e:
        logger.error(f"✗ Could not connect to Ollama server: {e}")
        logger.info("  Make sure Ollama is running with 'ollama serve'")
    return False

def load_documents(file_path):
    """Load documents from CSV file"""
    print_step(2, f"Loading data from {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"✗ File not found: {file_path}")
        return None
    
    # First show the raw CSV data for clarity
    try:
        df = pd.read_csv(file_path)
        logger.info(f"✓ CSV contains {len(df)} rows and {len(df.columns)} columns")
        print(f"\n{Fore.CYAN}Preview of CSV data:{Style.RESET_ALL}")
        print(df.head().to_string())
        print()
    except Exception as e:
        logger.warning(f"Could not preview CSV data: {e}")
    
    # Now load as LangChain documents
    loader = CSVLoader(file_path=file_path)
    docs = loader.load()
    logger.info(f"✓ Transformed into {len(docs)} LangChain documents")
    
    # Display an example document
    if docs:
        print(f"\n{Fore.CYAN}Example of a LangChain document:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{docs[0].page_content}{Style.RESET_ALL}\n")
    
    return docs

def setup_vector_store(docs, model_name="llama3.2"):
    """Create vector store using Ollama embeddings"""
    print_step(3, f"Creating vector embeddings with {model_name}")
    
    start_time = time.time()
    logger.info("→ Initializing embedding model...")
    embedding_model = OllamaEmbeddings(model=model_name)
    
    # Create a test embedding to show what it looks like
    logger.info("→ Generating sample embedding...")
    sample_text = "This is a sample product"
    sample_embedding = embedding_model.embed_query(sample_text)
    
    print(f"\n{Fore.CYAN}Example of vector embedding:{Style.RESET_ALL}")
    print(f"Text: '{sample_text}'")
    print(f"Vector dimensions: {len(sample_embedding)}")
    print(f"First 5 values: {sample_embedding[:5]}\n")
    
    logger.info("→ Creating FAISS vector store...")
    vector_store = FAISS.from_documents(docs, embedding=embedding_model)
    
    elapsed_time = time.time() - start_time
    logger.info(f"✓ Vector store created in {elapsed_time:.2f} seconds")
    
    return vector_store

def setup_qa_chain(retriever, model_name="llama3.2"):
    """Set up question-answering chain"""
    print_step(4, f"Setting up QA chain with {model_name}")
    
    logger.info("→ Initializing LLM...")
    llm = OllamaLLM(
        model=model_name,
        temperature=0.1
    )
    
    logger.info("→ Creating RetrievalQA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    
    logger.info(f"✓ QA chain ready")
    return qa_chain

def process_query(qa_chain, query, retriever):
    """Process a single query and return results"""
    print(f"\n{Fore.MAGENTA}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}QUERY: {query}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'=' * 80}{Style.RESET_ALL}\n")
    
    # Step 1: Show what documents are retrieved
    print(f"{Fore.YELLOW}[Retrieval Step]{Style.RESET_ALL} Finding relevant documents...")
    relevant_docs = retriever.get_relevant_documents(query)
    
    print(f"\n{Fore.CYAN}Retrieved {len(relevant_docs)} relevant documents:{Style.RESET_ALL}")
    for i, doc in enumerate(relevant_docs):
        print(f"\n{Fore.WHITE}Document {i+1}:{Style.RESET_ALL}")
        print(f"{doc.page_content}")
    
    # Step 2: Generate the answer
    print(f"\n{Fore.YELLOW}[Generation Step]{Style.RESET_ALL} Generating answer from context...")
    start_time = time.time()
    result = qa_chain.invoke({"query": query})
    
    # Extract answer and sources
    if isinstance(result, dict) and "result" in result:
        answer = result["result"]
        sources = result.get("source_documents", [])
    else:
        answer = str(result)
        sources = []
    
    elapsed_time = time.time() - start_time
    
    # Step 3: Show the final answer
    print(f"\n{Fore.CYAN}Final answer (generated in {elapsed_time:.2f} seconds):{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{answer}{Style.RESET_ALL}\n")
    
    return answer, sources

def main():
    print_header("CSV-based RAG System Demo")
    print("This demo shows how LangChain processes CSV data for question answering.\n")
    
    # Check Ollama server
    if not check_ollama_server():
        return
    
    # Load documents
    docs = load_documents("products.csv")
    if not docs:
        return
    
    # Setup vector store and retriever
    vector_store = setup_vector_store(docs)
    retriever = vector_store.as_retriever()
    
    # Setup QA chain
    qa_chain = setup_qa_chain(retriever)
    
    print_header("Demonstration Queries")
    print("The following queries will demonstrate the RAG system's capabilities.\n")
    
    # Process sample queries
    sample_queries = [
        "What's the cheapest product?",
        "Which laptop has the best rating?",
        "What products are made by Apple?"
    ]
    
    for i, query in enumerate(sample_queries):
        try:
            process_query(qa_chain, query, retriever)
            if i < len(sample_queries) - 1:
                input(f"\n{Fore.YELLOW}Press Enter to continue to the next query...{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")

if __name__ == "__main__":
    main()