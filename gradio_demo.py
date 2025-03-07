import gradio as gr
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import requests
import logging
import os
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables to store initialized components
qa_chain = None
csv_data = None

def check_ollama_server():
    """Check if Ollama server is running"""
    try:
        response = requests.get("http://localhost:11434/api/version")
        if response.status_code == 200:
            version = response.json().get('version')
            return True, f"Connected to Ollama version: {version}"
        else:
            return False, f"Ollama server responded with status code: {response.status_code}"
    except Exception as e:
        return False, f"Could not connect to Ollama server: {e}"

def load_documents(file_path):
    """Load documents from CSV file"""
    if not os.path.exists(file_path):
        return None, f"File not found: {file_path}"
        
    loader = CSVLoader(file_path=file_path)
    docs = loader.load()
    return docs, f"Loaded {len(docs)} documents from {file_path}"

def setup_system(csv_file_path, model_name="llama3.2"):
    """Initialize the RAG system"""
    global qa_chain, csv_data
    
    status_messages = []
    
    # Check Ollama server
    ollama_running, ollama_msg = check_ollama_server()
    status_messages.append(ollama_msg)
    if not ollama_running:
        return "\n".join(status_messages), None
    
    # Load CSV data for display
    try:
        csv_data = pd.read_csv(csv_file_path)
        status_messages.append(f"CSV data loaded successfully: {len(csv_data)} rows")
        
        # Display a preview of the CSV
        csv_preview = csv_data.head().to_string()
        status_messages.append(f"\nCSV Preview:\n{csv_preview}")
    except Exception as e:
        status_messages.append(f"Error loading CSV for display: {e}")
    
    # Load documents
    docs, load_msg = load_documents(csv_file_path)
    status_messages.append(load_msg)
    if docs is None:
        return "\n".join(status_messages), None
    
    # Setup vector store and embeddings
    try:
        status_messages.append(f"Setting up embeddings with model: {model_name}")
        embedding_start = time.time()
        embedding_model = OllamaEmbeddings(model=model_name)
        vector_store = FAISS.from_documents(docs, embedding=embedding_model)
        embedding_time = time.time() - embedding_start
        status_messages.append(f"Vector store created in {embedding_time:.2f} seconds")
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        status_messages.append("Retriever configured")
    except Exception as e:
        status_messages.append(f"Error setting up vector store: {e}")
        return "\n".join(status_messages), None
    
    # Setup QA chain
    try:
        status_messages.append(f"Initializing LLM with model: {model_name}")
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
        
        status_messages.append("RAG system initialized successfully!")
    except Exception as e:
        status_messages.append(f"Error setting up QA chain: {e}")
        return "\n".join(status_messages), None
    
    return "\n".join(status_messages), csv_data

def process_query(query):
    """Process a user query and return formatted results"""
    global qa_chain, csv_data
    
    if not query.strip():
        return "Please enter a query", "No query provided"
    
    if qa_chain is None:
        return "RAG system not initialized. Please initialize first.", "System not initialized"
    
    try:
        # Process the query
        start_time = time.time()
        result = qa_chain.invoke({"query": query})
        
        # Extract answer and sources
        if isinstance(result, dict) and "result" in result:
            answer = result["result"]
            sources = result.get("source_documents", [])
        else:
            answer = str(result)
            sources = []
        
        processing_time = time.time() - start_time
        
        # Format source information
        source_text = ""
        
        for i, doc in enumerate(sources):
            source_text += f"Source {i+1}:\n{doc.page_content}\n\n"
            
        result_text = f"Answer (processed in {processing_time:.2f} seconds):\n{answer}"
        
        return result_text, source_text
    except Exception as e:
        return f"Error processing query: {e}", "Error occurred"

# Define the Gradio interface with a top-down flow
with gr.Blocks(title="CSV RAG System Demo", theme=gr.themes.Base()) as demo:
    gr.Markdown("# CSV-based RAG System Demo")
    gr.Markdown("This demo showcases a Retrieval-Augmented Generation system that answers questions about product data using local LLMs.")
    
    # Step 1: Initialize the system
    gr.Markdown("## Step 1: Initialize RAG System")
    
    # Setup inputs in a row
    with gr.Row():
        csv_file = gr.Textbox(value="products.csv", label="CSV File Path")
        model_dropdown = gr.Dropdown(
            ["llama3.2", "everythinglm", "mistral"], 
            value="llama3.2", 
            label="Ollama Model"
        )
        init_button = gr.Button("Initialize", variant="primary")
    
    # Status output and CSV data display
    status_output = gr.Textbox(label="Initialization Status", lines=5)
    csv_display = gr.Dataframe(label="CSV Data Preview", visible=True)
    
    # Connect the initialization button
    init_button.click(
        setup_system, 
        inputs=[csv_file, model_dropdown], 
        outputs=[status_output, csv_display]
    )
    
    # Divider
    gr.Markdown("---")
    
    # Step 2: Ask questions
    gr.Markdown("## Step 2: Ask Questions")
    
    # Query input and button
    query_input = gr.Textbox(
        placeholder="Type your question here or select from the samples above", 
        label="Your Question",
        lines=2
    )
    query_button = gr.Button("Ask Question", variant="primary")
    
    # Results - These will be used
    result_output = gr.Textbox(label="Answer", lines=5)
    sources_output = gr.Textbox(label="Source Documents Used", lines=8)
    
    # Sample questions
    with gr.Accordion("Sample Questions", open=True):
        sample_questions = [
            "What's the cheapest product?",
            "Tell me about electronics products",
            "Which laptop has the best rating?",
            "What products are made by Apple?",
            "Which accessories cost less than $50?",
            "Compare the gaming products"
        ]
        
        sample_btns = []
        for q in sample_questions:
            sample_btn = gr.Button(q)
            sample_btn.click(lambda qst=q: qst, outputs=query_input)
            sample_btns.append(sample_btn)
    
    # Connect the query button
    query_button.click(
        process_query, 
        inputs=query_input, 
        outputs=[result_output, sources_output]
    )

if __name__ == "__main__":
    demo.launch()