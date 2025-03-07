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
        
        # Format the data summary for better display
        categories = csv_data['category'].unique()
        brands = csv_data['brand'].unique()
        price_range = f"${csv_data['price'].min()} - ${csv_data['price'].max()}"
        
        # Data summary
        status_messages.append(f"\n📊 Data Summary:")
        status_messages.append(f"• Products: {len(csv_data)}")
        status_messages.append(f"• Categories: {len(categories)} ({', '.join(categories) if len(categories) <= 5 else ', '.join(categories[:5]) + '...'})")
        status_messages.append(f"• Brands: {len(brands)} ({', '.join(brands) if len(brands) <= 5 else ', '.join(brands[:5]) + '...'})")
        status_messages.append(f"• Price Range: {price_range}")
        
        # Column info
        columns_info = []
        for col in csv_data.columns:
            columns_info.append(f"• {col}")
        status_messages.append(f"\n📋 Available Fields:\n" + "\n".join(columns_info))
        
        # Display a preview of the CSV
        csv_preview = csv_data.head(3).to_string()
        status_messages.append(f"\n👁️ Data Preview (first 3 rows):\n{csv_preview}")
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
        
        # Direct data analysis for specific numerical queries
        if csv_data is not None:
            # Price-related queries
            if any(term in query_lower for term in ["cheapest", "lowest price", "least expensive"]):
                if 'price' in csv_data.columns:
                    # Filter by category if specified
                    if any(cat in query_lower for cat in csv_data['category'].str.lower().unique()):
                        # Find which category is mentioned
                        for category in csv_data['category'].unique():
                            if category.lower() in query_lower:
                                category_data = csv_data[csv_data['category'] == category]
                                if not category_data.empty:
                                    cheapest = category_data.loc[category_data['price'].idxmin()]
                                    answer = f"The cheapest {category.lower()} is {cheapest['name']} at ${cheapest['price']}."
                                    result_text = f"Answer (processed with direct data analysis):\n{answer}"
                                    source_text = f"Source: Direct numerical analysis of price data\n\nProduct details:\n{cheapest.to_string()}"
                                    return result_text, source_text
                    else:
                        # Overall cheapest
                        cheapest_product = csv_data.loc[csv_data['price'].idxmin()]
                        answer = f"The cheapest product is {cheapest_product['name']} at ${cheapest_product['price']}."
                        result_text = f"Answer (processed with direct data analysis):\n{answer}"
                        source_text = f"Source: Direct numerical analysis of price data\n\nProduct details:\n{cheapest_product.to_string()}"
                        return result_text, source_text
                    
            # Most expensive queries
            elif any(term in query_lower for term in ["most expensive", "highest price", "costliest"]):
                if 'price' in csv_data.columns:
                    # Check for category qualifiers
                    for category in csv_data['category'].unique():
                        if category.lower() in query_lower:
                            category_data = csv_data[csv_data['category'] == category]
                            if not category_data.empty:
                                most_expensive = category_data.loc[category_data['price'].idxmax()]
                                answer = f"The most expensive {category.lower()} is {most_expensive['name']} at ${most_expensive['price']}."
                                result_text = f"Answer (processed with direct data analysis):\n{answer}"
                                source_text = f"Source: Direct numerical analysis of price data\n\nProduct details:\n{most_expensive.to_string()}"
                                return result_text, source_text
                    
                    # If no category specified, return overall most expensive
                    most_expensive = csv_data.loc[csv_data['price'].idxmax()]
                    answer = f"The most expensive product is {most_expensive['name']} at ${most_expensive['price']}."
                    result_text = f"Answer (processed with direct data analysis):\n{answer}"
                    source_text = f"Source: Direct numerical analysis of price data\n\nProduct details:\n{most_expensive.to_string()}"
                    return result_text, source_text
            
            # Products under a certain price
            elif "under" in query_lower and any(f"${i}" in query_lower for i in range(1, 2000)):
                price_limit = None
                # Extract price limit (e.g., "under $100")
                for i in range(1, 2000):
                    if f"${i}" in query_lower:
                        price_limit = i
                        break
                
                if price_limit and 'price' in csv_data.columns:
                    affordable_products = csv_data[csv_data['price'] < price_limit]
                    if not affordable_products.empty:
                        count = len(affordable_products)
                        avg_price = affordable_products['price'].mean()
                        cheap_products = affordable_products.sort_values('price').head(3)
                        
                        product_list = ""
                        for _, product in cheap_products.iterrows():
                            product_list += f"• {product['name']} (${product['price']}): {product['description'][:50]}...\n"
                        
                        answer = f"Found {count} products under ${price_limit}. The average price is ${avg_price:.2f}.\n\nHere are the cheapest options:\n{product_list}"
                        result_text = f"Answer (processed with direct data analysis):\n{answer}"
                        source_text = f"Source: Direct numerical analysis of price data\n\nFiltered products: {count} items under ${price_limit}"
                        return result_text, source_text
            
            # Newest products query
            elif any(term in query_lower for term in ["newest", "latest", "recent"]) and 'release_date' in csv_data.columns:
                # Sort by release date descending
                newest_products = csv_data.sort_values('release_date', ascending=False).head(5)
                
                if not newest_products.empty:
                    product_list = ""
                    for _, product in newest_products.iterrows():
                        release_date = pd.to_datetime(product['release_date']).strftime('%b %d, %Y')
                        product_list += f"• {product['name']} (${product['price']}) - Released: {release_date}\n"
                    
                    answer = f"The newest products in our inventory are:\n{product_list}"
                    result_text = f"Answer (processed with direct data analysis):\n{answer}"
                    source_text = f"Source: Direct analysis of release date data"
                    return result_text, source_text
            
            # Products with discounts
            elif any(term in query_lower for term in ["discount", "sale", "deals"]) and 'discount_percent' in csv_data.columns:
                discounted = csv_data[csv_data['discount_percent'] > 0].sort_values('discount_percent', ascending=False)
                
                if not discounted.empty:
                    top_deals = discounted.head(5)
                    deal_list = ""
                    for _, product in top_deals.iterrows():
                        original_price = product['price'] / (1 - product['discount_percent']/100)
                        savings = original_price - product['price']
                        deal_list += f"• {product['name']}: {product['discount_percent']}% off (Save ${savings:.2f})\n"
                    
                    answer = f"Found {len(discounted)} products with active discounts. Here are the best deals:\n{deal_list}"
                    result_text = f"Answer (processed with direct data analysis):\n{answer}"
                    source_text = f"Source: Direct analysis of discount data"
                    return result_text, source_text
            
            # Stock level queries
            elif any(term in query_lower for term in ["stock", "inventory", "available"]) and 'stock' in csv_data.columns:
                if "low" in query_lower:
                    low_stock = csv_data[csv_data['stock'] < 10].sort_values('stock')
                    
                    if not low_stock.empty:
                        stock_list = ""
                        for _, product in low_stock.iterrows():
                            stock_list += f"• {product['name']}: Only {product['stock']} left in stock\n"
                        
                        answer = f"Found {len(low_stock)} products with low stock (less than 10 units):\n{stock_list}"
                        result_text = f"Answer (processed with direct data analysis):\n{answer}"
                        source_text = f"Source: Direct analysis of inventory data"
                        return result_text, source_text
                else:
                    # General stock overview
                    total_items = csv_data['stock'].sum()
                    avg_stock = csv_data['stock'].mean()
                    out_of_stock = len(csv_data[csv_data['stock'] == 0])
                    low_stock = len(csv_data[csv_data['stock'] < 10])
                    
                    answer = f"Current inventory status:\n• Total items in stock: {total_items}\n• Average stock per product: {avg_stock:.1f}\n• Products out of stock: {out_of_stock}\n• Products with low stock: {low_stock}"
                    result_text = f"Answer (processed with direct data analysis):\n{answer}"
                    source_text = f"Source: Direct analysis of inventory data"
                    return result_text, source_text
                    
            # Highest rating queries
            elif any(term in query_lower for term in ["highest rating", "best rated", "top rated"]):
                if 'rating' in csv_data.columns:
                    # Check for category qualifiers
                    for category in csv_data['category'].unique():
                        if category.lower() in query_lower:
                            category_data = csv_data[csv_data['category'] == category]
                            if not category_data.empty:
                                best_rated = category_data.loc[category_data['rating'].idxmax()]
                                answer = f"The highest rated {category.lower()} is {best_rated['name']} with a rating of {best_rated['rating']} out of 5."
                                result_text = f"Answer (processed with direct data analysis):\n{answer}"
                                source_text = f"Source: Direct numerical analysis of rating data\n\nProduct details:\n{best_rated.to_string()}"
                                return result_text, source_text
                    
                    # If no category specified, return overall top rated
                    best_rated = csv_data.sort_values('rating', ascending=False).head(3)
                    
                    product_list = ""
                    for _, product in best_rated.iterrows():
                        product_list += f"• {product['name']} ({product['category']}): {product['rating']} stars\n"
                    
                    answer = f"The highest rated products are:\n{product_list}"
                    result_text = f"Answer (processed with direct data analysis):\n{answer}"
                    source_text = f"Source: Direct numerical analysis of rating data"
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

# Define the Gradio interface with improved layout and dark mode compatibility
with gr.Blocks(title="Product Catalog Explorer", theme=gr.themes.Default()) as demo:
    # Enhanced header with product catalog theme
    with gr.Row(elem_classes="header"):
        gr.Markdown("""
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px; padding: 10px; border-radius: 8px; position: relative; overflow: hidden;">
            <div style="font-size: 30px; z-index: 1;">🛍️</div>
            <div style="z-index: 1;">
                <h1 style="margin: 0; font-size: 24px;">Product Catalog Explorer</h1>
                <p style="margin: 5px 0 0 0;">Ask natural language questions about your product inventory</p>
            </div>
        </div>
        """)
    
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
            # Improved layout for Q&A section
            gr.Markdown("## Ask Questions About Your Products")
            
            # Enhanced query input with better styling
            with gr.Row():
                query_input = gr.Textbox(
                    placeholder="e.g., 'Show me the best-rated laptops' or 'What products are on sale?'", 
                    label="Your Question",
                    lines=2,
                    scale=4
                )
                query_button = gr.Button("🔍 Search Catalog", variant="primary", scale=1)
            
            # Sample questions organized in categories with tabbed interface
            with gr.Tabs() as question_tabs:
                with gr.TabItem("💰 Price & Value"):
                    with gr.Row():
                        price_questions = [
                            "What's the cheapest product?",
                            "What's the most expensive laptop?",
                            "Show me products under $100",
                            "Which products have discounts?"
                        ]
                        for q in price_questions:
                            sample_btn = gr.Button(q, size="sm")
                            sample_btn.click(lambda qst=q: qst, outputs=query_input)
                    
                    with gr.Row():
                        more_price_questions = [
                            "Compare the prices of gaming accessories",
                            "What's the average price of audio products?",
                            "Which brands offer the most affordable products?",
                            "What's the price range for monitors?"
                        ]
                        for q in more_price_questions:
                            sample_btn = gr.Button(q, size="sm")
                            sample_btn.click(lambda qst=q: qst, outputs=query_input)
                
                with gr.TabItem("🏷️ Brands & Categories"):
                    with gr.Row():
                        brand_questions = [
                            "What are all the Apple products?",
                            "Which brand has the most products?", 
                            "List all product categories",
                            "What Razer gaming products are available?"
                        ]
                        for q in brand_questions:
                            sample_btn = gr.Button(q, size="sm")
                            sample_btn.click(lambda qst=q: qst, outputs=query_input)
                    
                    with gr.Row():
                        more_brand_questions = [
                            "Compare Apple and Samsung products",
                            "What types of audio products are available?",
                            "Which brands make monitors?",
                            "Show me all the wearable devices"
                        ]
                        for q in more_brand_questions:
                            sample_btn = gr.Button(q, size="sm")
                            sample_btn.click(lambda qst=q: qst, outputs=query_input)
                
                with gr.TabItem("⭐ Ratings & Features"):
                    with gr.Row():
                        rating_questions = [
                            "Which products have the highest ratings?",
                            "Compare the top-rated laptops",
                            "Show me products with a rating over 4.7",
                            "What features do gaming laptops have?"
                        ]
                        for q in rating_questions:
                            sample_btn = gr.Button(q, size="sm")
                            sample_btn.click(lambda qst=q: qst, outputs=query_input)
                    
                    with gr.Row():
                        more_feature_questions = [
                            "Which products support Thunderbolt?",
                            "Compare noise cancellation in headphones",
                            "What products have RGB lighting?",
                            "Find products with wireless connectivity"
                        ]
                        for q in more_feature_questions:
                            sample_btn = gr.Button(q, size="sm")
                            sample_btn.click(lambda qst=q: qst, outputs=query_input)
                
                with gr.TabItem("📆 Availability & Release"):
                    with gr.Row():
                        availability_questions = [
                            "What are the newest products?",
                            "Which products are low in stock?",
                            "Show me products released in 2023",
                            "Which Apple products were released this year?"
                        ]
                        for q in availability_questions:
                            sample_btn = gr.Button(q, size="sm")
                            sample_btn.click(lambda qst=q: qst, outputs=query_input)
                    
                    with gr.Row():
                        more_availability_questions = [
                            "Which products are currently in stock?",
                            "Compare products released in the last 6 months",
                            "When was the MacBook Pro released?",
                            "What's the oldest product in the inventory?"
                        ]
                        for q in more_availability_questions:
                            sample_btn = gr.Button(q, size="sm")
                            sample_btn.click(lambda qst=q: qst, outputs=query_input)
            
            # Results in a separate section from conversation history
            gr.Markdown("### Results")
            with gr.Row():
                with gr.Column(scale=2):
                    # Results section with more space for answers
                    result_output = gr.Textbox(label="Answer", lines=5)
                    with gr.Accordion("Sources & Data", open=False):
                        sources_output = gr.Textbox(label="Sources Used", lines=6)
                
                with gr.Column(scale=1):
                    # Optional chat history in a separate column with better styling
                    gr.Markdown("#### Previous Questions")
                    chat_history = gr.Textbox(label="", lines=10)
                    with gr.Row():
                        clear_history_btn = gr.Button("Clear History")
                        copy_history_btn = gr.Button("📋 Copy")
    
    # Add footer with attribution (simplified)
    with gr.Row(elem_classes="footer"):
        gr.Markdown("""
        <div style="text-align: center; margin-top: 20px; padding: 10px;">
            <p style="margin: 0; font-size: 14px;">📦 <b>Powered by LangChain & Ollama</b></p>
        </div>
        """)
    
    # Add custom CSS for better UI with dark mode compatibility and product catalog styling
    gr.Markdown("""
    <style>
    .gradio-container {
        max-width: 90% !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    
    /* Dark mode detection */
    @media (prefers-color-scheme: dark) {
        body {
            color-scheme: dark;
        }
        
        .header div, .footer div {
            background: rgba(30, 30, 30, 0.2) !important;
            color: #eee !important;
        }
        
        .header h1, .header p {
            color: #eee !important;
        }
        
        /* Make buttons more visible in dark mode */
        button.secondary {
            background-color: rgba(60, 60, 60, 0.8) !important;
            color: #ddd !important;
            border: 1px solid #555 !important;
        }
        
        button.primary {
            background-color: #2a6099 !important;
            border: 1px solid #3a70a9 !important;
        }
        
        /* Dark mode result box styling */
        textarea[label="Answer"] {
            background-color: rgba(40, 40, 40, 0.3) !important;
            border-left: 4px solid #2a6099 !important;
        }
    }
    
    /* Light mode styles */
    @media (prefers-color-scheme: light) {
        .header div {
            background: linear-gradient(to right, #f0f7ff, #e6f0fb);
        }
        
        /* Light mode result box styling */
        textarea[label="Answer"] {
            background-color: #f8fbff !important;
            border-left: 4px solid #2a6099 !important;
        }
    }
    
    /* General improvements */
    .tabs > .tab-nav > button {
        font-size: 1em !important;
        font-weight: 600 !important;
    }
    
    /* Better button styling */
    button {
        border-radius: 4px !important;
        margin: 0 3px !important;
        transition: all 0.2s ease !important;
    }
    
    button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    button.primary {
        font-weight: 600 !important;
    }
    
    /* Make sample questions buttons compact but readable */
    button[size="sm"] {
        padding: 3px 10px !important;
        font-size: 0.85em !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        max-width: 180px !important;
        border-radius: 20px !important;
    }
    
    /* Enhanced input styling */
    textarea[label="Your Question"] {
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    textarea[label="Your Question"]:focus {
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3) !important;
    }
    
    /* Improved tab styling */
    .tabs > .tab-nav {
        border-bottom: 2px solid #e5e7eb !important;
        padding-bottom: 0 !important;
    }
    
    .tabs > .tab-nav > button {
        border-radius: 8px 8px 0 0 !important;
        padding: 8px 16px !important;
        margin: 0 2px !important;
    }
    
    .tabs > .tab-nav > button.selected {
        font-weight: bold !important;
        color: #2a6099 !important;
        border-bottom: 3px solid #2a6099 !important;
    }
    
    /* More responsive layout */
    @media (max-width: 768px) {
        .gradio-container {
            max-width: 100% !important;
        }
        
        button[size="sm"] {
            max-width: 120px !important;
        }
    }
    </style>
    """)
    
    # Connect the initialization button
    init_button.click(
        setup_system, 
        inputs=[csv_file, model_dropdown], 
        outputs=[status_output, csv_display, system_status]
    )
    
    # Define a function to update the chat history text (simplified from the chat widget)
    def update_chat_history(query, result, history):
        if history is None or history.strip() == "":
            history = ""
        
        # Extract just the answer part without the "processed in X seconds" prefix
        answer_text = result.split(":\n", 1)[1] if ":\n" in result else result
        
        # Add timestamp for better history tracking
        timestamp = time.strftime("%H:%M:%S")
        
        # Add the new Q&A to the history
        new_history = f"{history}[{timestamp}] Q: {query}\nA: {answer_text}\n{'-'*40}\n"
        return new_history
    
    # Function to copy history to clipboard
    def copy_to_clipboard():
        return "History copied to clipboard!"
    
    # Connect the query button
    query_button.click(
        process_query, 
        inputs=query_input, 
        outputs=[result_output, sources_output]
    ).then(
        update_chat_history,
        inputs=[query_input, result_output, chat_history],
        outputs=chat_history
    )
    
    # Support for pressing Enter to submit
    query_input.submit(
        process_query, 
        inputs=query_input, 
        outputs=[result_output, sources_output]
    ).then(
        update_chat_history,
        inputs=[query_input, result_output, chat_history],
        outputs=chat_history
    )
    
    # Clear history functionality
    clear_history_btn.click(
        lambda: "",
        outputs=chat_history
    )
    
    # Copy history functionality (shows a message when clicked)
    copy_history_btn.click(
        copy_to_clipboard,
        outputs=gr.Textbox(visible=False)
    )

if __name__ == "__main__":
    demo.launch(
        height=700,  # Slightly reduced height
    )