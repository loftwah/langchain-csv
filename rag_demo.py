from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import requests
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_ollama_server():
    """Check if Ollama server is running and return available models"""
    try:
        response = requests.get("http://localhost:11434/api/version")
        if response.status_code == 200:
            logger.info(f"Connected to Ollama version: {response.json().get('version')}")
            return True
        else:
            logger.warning(f"Ollama server responded with status code: {response.status_code}")
    except Exception as e:
        logger.error(f"Could not connect to Ollama server: {e}")
        logger.info("Make sure Ollama is running with 'ollama serve'")
    return False

def load_documents(file_path):
    """Load documents from CSV file"""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
        
    loader = CSVLoader(file_path=file_path)
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} documents from {file_path}")
    return docs

def setup_vector_store(docs, model_name="llama3.2"):
    """Create vector store using Ollama embeddings"""
    embedding_model = OllamaEmbeddings(model=model_name)
    vector_store = FAISS.from_documents(docs, embedding=embedding_model)
    logger.info(f"Vector store created with {model_name} embeddings")
    return vector_store

def setup_qa_chain(retriever, model_name="llama3.2"):
    """Set up question-answering chain"""
    llm = OllamaLLM(
        model=model_name,
        temperature=0.1
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    
    logger.info(f"QA chain initialized with {model_name}")
    return qa_chain

def process_query(qa_chain, query):
    """Process a single query and return results"""
    logger.info(f"Processing query: {query}")
    
    result = qa_chain.invoke({"query": query})
    
    # Extract answer and sources
    if isinstance(result, dict) and "result" in result:
        answer = result["result"]
        sources = result.get("source_documents", [])
    else:
        answer = str(result)
        sources = []
    
    # Print results
    logger.info(f"Query: {query}")
    logger.info(f"Response: {answer}")
    
    if sources:
        logger.info("Sources:")
        for i, doc in enumerate(sources[:3]):
            logger.info(f"  Source {i+1}: {doc.page_content[:100]}...")
    
    return answer, sources

def main():
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
    
    # Process sample queries
    sample_queries = [
        "What's the cheapest product?",
        "Tell me about electronics products",
        "Which laptop has the best rating?",
        "What products are made by Apple?"
    ]
    
    for query in sample_queries:
        try:
            process_query(qa_chain, query)
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")

if __name__ == "__main__":
    main()