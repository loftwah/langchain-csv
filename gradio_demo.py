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
        return "\n".join(status_messages), None, "⚠️ Ollama server not available"
    
    # Load CSV data for display
    try:
        csv_data = pd.read_csv(csv_file_path)
        status_messages.append(f"✅ CSV data loaded successfully: {len(csv_data)} rows")
        
        # Display a preview of the CSV
        csv_preview = csv_data.head().to_string()
        status_messages.append(f"\nCSV Preview:\n{csv_preview}")
    except Exception as e:
        status_messages.append(f"❌ Error loading CSV for display: {e}")
    
    # Load documents
    docs, load_msg = load_documents(csv_file_path)
    status_messages.append(load_msg)
    if docs is None:
        return "\n".join(status_messages), None, "❌ Failed to load documents"
    
    # Setup vector store and embeddings
    try:
        status_messages.append(f"Setting up embeddings with model: {model_name}")
        embedding_start = time.time()
        embedding_model = OllamaEmbeddings(model=model_name)
        vector_store = FAISS.from_documents(docs, embedding=embedding_model)
        embedding_time = time.time() - embedding_start
        status_messages.append(f"✅ Vector store created in {embedding_time:.2f} seconds")
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        status_messages.append("✅ Retriever configured")
    except Exception as e:
        status_messages.append(f"❌ Error setting up vector store: {e}")
        return "\n".join(status_messages), None, "❌ Failed to setup vector store"
    
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
        
        status_messages.append("✅ RAG system initialized successfully!")
    except Exception as e:
        status_messages.append(f"❌ Error setting up QA chain: {e}")
        return "\n".join(status_messages), None, "❌ Failed to setup QA chain"
    
    return "\n".join(status_messages), csv_data, "✅ System ready!"

def process_query(query):
    """Process a user query and return formatted results"""
    global qa_chain, csv_data
    
    if not query.strip():
        return "Please enter a query", "No query provided"
    
    if qa_chain is None:
        return "RAG system not initialized. Please initialize first.", "System not initialized"
    
    try:
        # Special handling for specific query types that require precise numerical analysis
        query_lower = query.lower()
        if csv_data is not None and any(term in query_lower for term in ["cheapest", "lowest price", "least expensive"]):
            # Direct data processing for price queries
            if 'price' in csv_data.columns:
                cheapest_product = csv_data.loc[csv_data['price'].idxmin()]
                answer = f"The cheapest product is {cheapest_product['name']} at ${cheapest_product['price']}."
                result_text = f"Answer (processed with direct data analysis):\n{answer}"
                source_text = f"Source: Direct numerical analysis of price data\n\nProduct details:\n{cheapest_product.to_string()}"
                return result_text, source_text
        
        # Process the query with RAG for all other queries
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
with gr.Blocks(title="CSV RAG System Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Interactive CSV-based RAG System")
    gr.Markdown("Ask questions about products in natural language and get AI-powered answers.")
    
    # System status indicator
    with gr.Row():
        system_status = gr.Textbox(value="System not initialized", label="Status", interactive=False)
    
    # Setup inputs in tabs for better organization
    with gr.Tabs():
        with gr.TabItem("🛠️ Setup & Data"):
            # Step 1: Initialize the system
            gr.Markdown("## Initialize RAG System")
            
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
            status_output = gr.Textbox(label="Initialization Log", lines=5)
            
            with gr.Accordion("📊 Data Preview", open=True):
                csv_display = gr.Dataframe(label="CSV Data", visible=True)
        
        with gr.TabItem("❓ Ask Questions"):
            # Step 2: Ask questions
            gr.Markdown("## Ask Questions About Your Products")
            
            # Interactive elements
            with gr.Row():
                with gr.Column(scale=3):
                    # Query input
                    query_input = gr.Textbox(
                        placeholder="Type your question here or select from the samples above", 
                        label="Your Question",
                        lines=2
                    )
                    
                    # Query button with icon
                    query_button = gr.Button("🔍 Ask Question", variant="primary")
                
                with gr.Column(scale=1):
                    # Sample questions
                    gr.Markdown("### Sample Questions")
                    
                    sample_questions = [
                        "What's the cheapest product?",
                        "Tell me about electronics products",
                        "Which laptop has the best rating?",
                        "What products are made by Apple?",
                        "Which accessories cost less than $50?",
                        "Compare the gaming products"
                    ]
                    
                    for q in sample_questions:
                        sample_btn = gr.Button(q)
                        sample_btn.click(lambda qst=q: qst, outputs=query_input)
            
            # Add a chat history for a more conversational feel
            chat_history = gr.Chatbot(label="Conversation History", height=400)
            
            # Results
            with gr.Accordion("🔍 Answer Details", open=True):
                result_output = gr.Textbox(label="Answer", lines=4)
                sources_output = gr.Textbox(label="Sources Used", lines=6)
    
    with gr.Accordion("ℹ️ About This Demo", open=False):
        gr.Markdown("""
        ## How It Works
        
        This system uses **RAG (Retrieval-Augmented Generation)** to answer questions about product data:
        
        1. **Vector Embeddings**: The system creates mathematical representations of your product data
        2. **Semantic Search**: When you ask a question, it finds the most relevant products
        3. **LLM Generation**: It uses a language model to create a natural language answer
        
        ### Known Limitations
        
        - For numerical queries (like finding the cheapest item), the system uses direct data analysis to ensure accuracy
        - Complex multi-step reasoning may sometimes produce incorrect results
        - The quality of answers depends on the richness of your product data
        """)
    
    # Connect the initialization button
    init_button.click(
        setup_system, 
        inputs=[csv_file, model_dropdown], 
        outputs=[status_output, csv_display, system_status]
    )
    
    # Define a function to handle the conversation
    def add_to_conversation(query, result, sources):
        # Add user query to chat
        chat_history = [[query, None]]
        
        # Add system response to chat
        if "Error" in result:
            chat_history[0][1] = f"❌ {result}"
        else:
            # Extract just the answer part without the "processed in X seconds" prefix
            answer_text = result.split(":\n", 1)[1] if ":\n" in result else result
            chat_history[0][1] = answer_text
        
        return chat_history, result, sources
    
    # Connect the query button with updated function
    query_button.click(
        process_query, 
        inputs=query_input, 
        outputs=[result_output, sources_output]
    ).then(
        add_to_conversation,
        inputs=[query_input, result_output, sources_output],
        outputs=[chat_history, result_output, sources_output]
    )

if __name__ == "__main__":
    demo.launch()