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
from datetime import datetime

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

def print_category_header(text):
    """Print a category header"""
    print(f"\n{Fore.BLUE}{'·' * 60}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}· {text}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'·' * 60}{Style.RESET_ALL}\n")

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

def analyze_csv_stats(df):
    """Analyze and display statistics about the CSV data"""
    categories = df['category'].unique()
    brands = df['brand'].unique()
    price_range = f"${df['price'].min()} - ${df['price'].max()}"
    total_stock = df['stock'].sum() if 'stock' in df.columns else "N/A"
    avg_rating = round(df['rating'].mean(), 2) if 'rating' in df.columns else "N/A"
    
    print(f"\n{Fore.CYAN}Dataset Statistics:{Style.RESET_ALL}")
    print(f"• {Fore.WHITE}Products:{Style.RESET_ALL} {len(df)}")
    print(f"• {Fore.WHITE}Categories:{Style.RESET_ALL} {len(categories)} ({', '.join(categories) if len(categories) <= 5 else ', '.join(categories[:5]) + '...'})")
    print(f"• {Fore.WHITE}Brands:{Style.RESET_ALL} {len(brands)} ({', '.join(brands) if len(brands) <= 5 else ', '.join(brands[:5]) + '...'})")
    print(f"• {Fore.WHITE}Price Range:{Style.RESET_ALL} {price_range}")
    print(f"• {Fore.WHITE}Total Items in Stock:{Style.RESET_ALL} {total_stock}")
    print(f"• {Fore.WHITE}Average Rating:{Style.RESET_ALL} {avg_rating}")
    
    # Display release date range if available
    if 'release_date' in df.columns:
        try:
            df['release_date'] = pd.to_datetime(df['release_date'])
            min_date = df['release_date'].min().strftime('%B %Y')
            max_date = df['release_date'].max().strftime('%B %Y')
            print(f"• {Fore.WHITE}Release Date Range:{Style.RESET_ALL} {min_date} to {max_date}")
        except:
            pass
    
    # Display discount information if available
    if 'discount_percent' in df.columns:
        discounted_count = len(df[df['discount_percent'] > 0])
        if discounted_count > 0:
            avg_discount = round(df[df['discount_percent'] > 0]['discount_percent'].mean(), 1)
            print(f"• {Fore.WHITE}Products on Sale:{Style.RESET_ALL} {discounted_count} ({avg_discount}% average discount)")
    
    # Show field information
    print(f"\n{Fore.CYAN}Available Fields:{Style.RESET_ALL}")
    for col in df.columns:
        print(f"• {col}")

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
        
        # Analyze the data
        analyze_csv_stats(df)
        
        print(f"\n{Fore.CYAN}Preview of CSV data:{Style.RESET_ALL}")
        print(df.head(3).to_string())
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

def direct_data_analysis(query, csv_data):
    """Perform direct data analysis for specific query types"""
    query_lower = query.lower()
    
    # Cheapest product query
    if any(term in query_lower for term in ["cheapest", "lowest price", "least expensive"]):
        print(f"{Fore.YELLOW}[Direct Analysis]{Style.RESET_ALL} Detecting price comparison query - using direct data analysis")
        
        if 'price' in csv_data.columns:
            # Check for category qualifiers
            for category in csv_data['category'].unique():
                if category.lower() in query_lower:
                    category_data = csv_data[csv_data['category'] == category]
                    if not category_data.empty:
                        cheapest_product = category_data.loc[category_data['price'].idxmin()]
                        print(f"\n{Fore.CYAN}Direct data analysis results:{Style.RESET_ALL}")
                        print(f"Cheapest {category}: {Fore.GREEN}{cheapest_product['name']} at ${cheapest_product['price']}{Style.RESET_ALL}")
                        
                        answer = f"The cheapest {category.lower()} is {cheapest_product['name']} at ${cheapest_product['price']}."
                        return answer, [pd.Series.to_frame(cheapest_product)]
            
            # No category specified, return overall cheapest
            cheapest_product = csv_data.loc[csv_data['price'].idxmin()]
            print(f"\n{Fore.CYAN}Direct data analysis results:{Style.RESET_ALL}")
            print(f"Cheapest product: {Fore.GREEN}{cheapest_product['name']} at ${cheapest_product['price']}{Style.RESET_ALL}")
            
            answer = f"The cheapest product is {cheapest_product['name']} at ${cheapest_product['price']}."
            return answer, [pd.Series.to_frame(cheapest_product)]
    
    # Most expensive product query
    elif any(term in query_lower for term in ["most expensive", "highest price", "costliest"]):
        print(f"{Fore.YELLOW}[Direct Analysis]{Style.RESET_ALL} Detecting price comparison query - using direct data analysis")
        
        if 'price' in csv_data.columns:
            # Check for category qualifiers
            for category in csv_data['category'].unique():
                if category.lower() in query_lower:
                    category_data = csv_data[csv_data['category'] == category]
                    if not category_data.empty:
                        most_expensive = category_data.loc[category_data['price'].idxmax()]
                        print(f"\n{Fore.CYAN}Direct data analysis results:{Style.RESET_ALL}")
                        print(f"Most expensive {category}: {Fore.GREEN}{most_expensive['name']} at ${most_expensive['price']}{Style.RESET_ALL}")
                        
                        answer = f"The most expensive {category.lower()} is {most_expensive['name']} at ${most_expensive['price']}."
                        return answer, [pd.Series.to_frame(most_expensive)]
            
            # No category specified, return overall most expensive
            most_expensive = csv_data.loc[csv_data['price'].idxmax()]
            print(f"\n{Fore.CYAN}Direct data analysis results:{Style.RESET_ALL}")
            print(f"Most expensive product: {Fore.GREEN}{most_expensive['name']} at ${most_expensive['price']}{Style.RESET_ALL}")
            
            answer = f"The most expensive product is {most_expensive['name']} at ${most_expensive['price']}."
            return answer, [pd.Series.to_frame(most_expensive)]
    
    # Best rated product query
    elif any(term in query_lower for term in ["highest rating", "best rated", "top rated"]):
        print(f"{Fore.YELLOW}[Direct Analysis]{Style.RESET_ALL} Detecting rating query - using direct data analysis")
        
        if 'rating' in csv_data.columns:
            # Check for category qualifiers
            for category in csv_data['category'].unique():
                if category.lower() in query_lower:
                    category_data = csv_data[csv_data['category'] == category]
                    if not category_data.empty:
                        best_rated = category_data.loc[category_data['rating'].idxmax()]
                        print(f"\n{Fore.CYAN}Direct data analysis results:{Style.RESET_ALL}")
                        print(f"Highest rated {category}: {Fore.GREEN}{best_rated['name']} with {best_rated['rating']} stars{Style.RESET_ALL}")
                        
                        answer = f"The highest rated {category.lower()} is {best_rated['name']} with a rating of {best_rated['rating']} out of 5."
                        return answer, [pd.Series.to_frame(best_rated)]
            
            # If no category specified, return overall top rated products
            best_rated = csv_data.sort_values('rating', ascending=False).head(3)
            
            print(f"\n{Fore.CYAN}Direct data analysis results:{Style.RESET_ALL}")
            print(f"Top rated products:")
            
            product_list = ""
            for i, (_, product) in enumerate(best_rated.iterrows()):
                print(f"{i+1}. {Fore.GREEN}{product['name']} ({product['category']}) - {product['rating']} stars{Style.RESET_ALL}")
                product_list += f"{product['name']} ({product['category']}) with {product['rating']} stars, "
            
            answer = f"The highest rated products are: {product_list[:-2]}."
            return answer, [best_rated]
    
    # Products under a certain price
    elif "under" in query_lower and any(f"${i}" in query_lower for i in range(1, 2000)):
        print(f"{Fore.YELLOW}[Direct Analysis]{Style.RESET_ALL} Detecting price range query - using direct data analysis")
        
        # Extract price limit (e.g., "under $100")
        price_limit = None
        for i in range(1, 2000):
            if f"${i}" in query_lower:
                price_limit = i
                break
        
        if price_limit and 'price' in csv_data.columns:
            # Filter products by the price limit
            affordable_products = csv_data[csv_data['price'] < price_limit]
            
            if not affordable_products.empty:
                count = len(affordable_products)
                avg_price = affordable_products['price'].mean()
                
                print(f"\n{Fore.CYAN}Direct data analysis results:{Style.RESET_ALL}")
                print(f"Found {Fore.GREEN}{count} products under ${price_limit}{Style.RESET_ALL}")
                print(f"Average price: ${avg_price:.2f}")
                
                # Show top 3 cheapest products
                cheapest = affordable_products.sort_values('price').head(3)
                print(f"\nCheapest options:")
                
                product_list = ""
                for i, (_, product) in enumerate(cheapest.iterrows()):
                    print(f"{i+1}. {Fore.GREEN}{product['name']} (${product['price']}){Style.RESET_ALL}")
                    product_list += f"{product['name']} (${product['price']}), "
                
                answer = f"Found {count} products under ${price_limit}. The average price is ${avg_price:.2f}. "
                answer += f"Some of the most affordable options include: {product_list[:-2]}."
                return answer, [affordable_products.head()]
    
    # Products on sale
    elif any(term in query_lower for term in ["discount", "sale", "deals"]):
        print(f"{Fore.YELLOW}[Direct Analysis]{Style.RESET_ALL} Detecting discount query - using direct data analysis")
        
        if 'discount_percent' in csv_data.columns:
            # Make sure discount_percent is numeric
            try:
                # Convert column to numeric, forcing errors to become NaN
                csv_data['discount_percent'] = pd.to_numeric(csv_data['discount_percent'], errors='coerce')
                
                # Filter out NaN values and zeroes
                discounted = csv_data[csv_data['discount_percent'] > 0].dropna(subset=['discount_percent']).sort_values('discount_percent', ascending=False)
                
                if not discounted.empty:
                    count = len(discounted)
                    avg_discount = discounted['discount_percent'].mean()
                    
                    print(f"\n{Fore.CYAN}Direct data analysis results:{Style.RESET_ALL}")
                    print(f"Found {Fore.GREEN}{count} products on sale{Style.RESET_ALL}")
                    print(f"Average discount: {avg_discount:.1f}%")
                    
                    # Show top deals
                    best_deals = discounted.head(3)
                    print(f"\nBest deals:")
                    
                    deal_list = ""
                    for i, (_, product) in enumerate(best_deals.iterrows()):
                        try:
                            # Ensure values are properly converted to float for calculation
                            price = float(product['price'])
                            discount = float(product['discount_percent'])
                            
                            # Calculate original price and savings
                            original_price = price / (1 - discount/100) if discount < 100 else price * 2
                            savings = original_price - price
                            
                            print(f"{i+1}. {Fore.GREEN}{product['name']} - {discount:.1f}% off (Save ${savings:.2f}){Style.RESET_ALL}")
                            deal_list += f"{product['name']} ({discount:.1f}% off), "
                        except (ValueError, TypeError) as e:
                            # Skip this product if there's an error in calculation
                            print(f"{Fore.YELLOW}Warning: Could not calculate discount for {product['name']}: {e}{Style.RESET_ALL}")
                            continue
                    
                    # Only add deal list to answer if we have deals
                    if deal_list:
                        answer = f"Found {count} products with active discounts. The average discount is {avg_discount:.1f}%. "
                        answer += f"The best deals include: {deal_list[:-2]}."
                    else:
                        answer = f"Found {count} products with active discounts. The average discount is {avg_discount:.1f}%."
                    
                    return answer, [discounted.head()]
            except Exception as e:
                print(f"{Fore.RED}Error processing discount data: {e}{Style.RESET_ALL}")
                # Fall back to RAG-based answer instead of direct analysis
    
    # Newest products query
    elif any(term in query_lower for term in ["newest", "latest", "recent"]) and 'release_date' in csv_data.columns:
        print(f"{Fore.YELLOW}[Direct Analysis]{Style.RESET_ALL} Detecting release date query - using direct data analysis")
        
        # Convert release_date to datetime
        try:
            csv_data['release_date'] = pd.to_datetime(csv_data['release_date'])
            newest_products = csv_data.sort_values('release_date', ascending=False).head(5)
            
            if not newest_products.empty:
                print(f"\n{Fore.CYAN}Direct data analysis results:{Style.RESET_ALL}")
                print(f"Newest products:")
                
                product_list = ""
                for i, (_, product) in enumerate(newest_products.iterrows()):
                    release_date = product['release_date'].strftime('%B %d, %Y')
                    print(f"{i+1}. {Fore.GREEN}{product['name']} - Released: {release_date}{Style.RESET_ALL}")
                    product_list += f"{product['name']} (released on {release_date}), "
                
                answer = f"The newest products in our inventory are: {product_list[:-2]}."
                return answer, [newest_products]
        except Exception as e:
            print(f"Error processing release dates: {e}")
    
    # Stock level queries
    elif any(term in query_lower for term in ["in stock", "low stock", "available"]) and 'stock' in csv_data.columns:
        print(f"{Fore.YELLOW}[Direct Analysis]{Style.RESET_ALL} Detecting inventory query - using direct data analysis")
        
        if "low" in query_lower:
            # Find products with low stock (less than 10)
            low_stock_products = csv_data[csv_data['stock'] < 10].sort_values('stock')
            
            if not low_stock_products.empty:
                count = len(low_stock_products)
                
                print(f"\n{Fore.CYAN}Direct data analysis results:{Style.RESET_ALL}")
                print(f"Found {Fore.GREEN}{count} products with low stock{Style.RESET_ALL}")
                
                # Show products with lowest stock
                critical_stock = low_stock_products.head(3)
                print(f"\nProducts with critical stock levels:")
                
                product_list = ""
                for i, (_, product) in enumerate(critical_stock.iterrows()):
                    print(f"{i+1}. {Fore.GREEN}{product['name']} - Only {product['stock']} left{Style.RESET_ALL}")
                    product_list += f"{product['name']} (only {product['stock']} remaining), "
                
                answer = f"Found {count} products with low stock (less than 10 units). "
                answer += f"The most critical are: {product_list[:-2]}."
                return answer, [low_stock_products.head()]
        else:
            # General stock overview
            total_items = csv_data['stock'].sum()
            avg_stock = csv_data['stock'].mean()
            out_of_stock = len(csv_data[csv_data['stock'] == 0])
            low_stock = len(csv_data[csv_data['stock'] < 10])
            
            print(f"\n{Fore.CYAN}Direct data analysis results:{Style.RESET_ALL}")
            print(f"Stock overview:")
            print(f"• Total items in stock: {Fore.GREEN}{total_items}{Style.RESET_ALL}")
            print(f"• Average stock per product: {avg_stock:.1f}")
            print(f"• Products out of stock: {out_of_stock}")
            print(f"• Products with low stock: {low_stock}")
            
            answer = f"Current inventory status: We have {total_items} total items in stock, "
            answer += f"with an average of {avg_stock:.1f} units per product. "
            answer += f"There are {out_of_stock} products out of stock and {low_stock} products with low stock (less than 10 units)."
            return answer, []
    
    # Products with specific features
    elif any(term in query_lower for term in ["feature", "support", "has", "with"]) and 'features' in csv_data.columns:
        # Look for feature keywords in the query
        feature_keywords = ["thunderbolt", "usb-c", "wireless", "bluetooth", "noise cancellation", 
                           "oled", "amoled", "ips", "led", "120hz", "144hz", "240hz", "rgb", 
                           "mechanical", "optical", "water resistant", "waterproof"]
        
        found_features = []
        for feature in feature_keywords:
            if feature in query_lower:
                found_features.append(feature)
        
        if found_features:
            print(f"{Fore.YELLOW}[Direct Analysis]{Style.RESET_ALL} Detecting feature query - using direct data analysis")
            print(f"Looking for products with feature: {', '.join(found_features)}")
            
            # Convert features column to lists for easier searching
            feature_matches = []
            
            for idx, row in csv_data.iterrows():
                if isinstance(row['features'], str):
                    row_features = row['features'].lower()
                    if any(feature in row_features for feature in found_features):
                        feature_matches.append(row)
            
            if feature_matches:
                count = len(feature_matches)
                features_df = pd.DataFrame(feature_matches)
                
                print(f"\n{Fore.CYAN}Direct data analysis results:{Style.RESET_ALL}")
                print(f"Found {Fore.GREEN}{count} products with {'/'.join(found_features)}{Style.RESET_ALL}")
                
                # Show matching products
                top_matches = features_df.head(3)
                print(f"\nMatching products:")
                
                product_list = ""
                for i, product in top_matches.iterrows():
                    print(f"{i+1}. {Fore.GREEN}{product['name']} ({product['category']}){Style.RESET_ALL}")
                    product_list += f"{product['name']} ({product['category']}), "
                
                feature_text = '/'.join(found_features)
                answer = f"Found {count} products with {feature_text} feature. "
                answer += f"Some examples include: {product_list[:-2]}."
                return answer, [features_df.head()]
    
    # No direct analysis was performed
    return None, None

def process_query(qa_chain, query, retriever, csv_data=None):
    """Process a single query and return results"""
    print(f"\n{Fore.MAGENTA}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}QUERY: {query}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'=' * 80}{Style.RESET_ALL}\n")
    
    # Check if we can perform direct data analysis
    if csv_data is not None:
        direct_answer, direct_sources = direct_data_analysis(query, csv_data)
        if direct_answer:
            return direct_answer, direct_sources
    
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

def get_sample_queries():
    """Return categorized sample queries"""
    queries = {
        "Price & Value": [
            "What's the cheapest product?",
            "What's the most expensive laptop?",
            "Show me products under $100",
            "Which products have discounts?"
        ],
        "Brands & Categories": [
            "What Apple products do you have?",
            "Which brands make monitors?",
            "Compare Apple and Samsung products",
            "What types of audio products are available?"
        ],
        "Ratings & Features": [
            "Which products have the highest ratings?",
            "What features do gaming laptops have?",
            "Which products support Thunderbolt?",
            "Find products with wireless connectivity"
        ],
        "Availability & Release": [
            "What are the newest products?",
            "Which products are low in stock?",
            "Show me products released in 2023",
            "When was the MacBook Pro released?"
        ]
    }
    return queries

def main():
    print_header("🛍️ INTERACTIVE PRODUCT CATALOG EXPLORER")
    print("This demo shows how LangChain processes product data for question answering.\n")
    
    # Check Ollama server
    if not check_ollama_server():
        return
    
    # Load documents
    docs = load_documents("products.csv")
    if not docs:
        return
    
    # Load raw CSV data for special queries
    try:
        csv_data = pd.read_csv("products.csv")
    except:
        csv_data = None
        logger.warning("Could not load raw CSV data for special query handling")
    
    # Setup vector store and retriever
    vector_store = setup_vector_store(docs)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Setup QA chain
    qa_chain = setup_qa_chain(retriever)
    
    # Offer choice of demo mode
    print(f"\n{Fore.CYAN}Choose a demo mode:{Style.RESET_ALL}")
    print(f"1. {Fore.YELLOW}Sample Queries{Style.RESET_ALL} - See the system answer preset questions")
    print(f"2. {Fore.YELLOW}Interactive Mode{Style.RESET_ALL} - Ask your own questions")
    print(f"3. {Fore.YELLOW}Behind the Scenes{Style.RESET_ALL} - See how RAG works with a deep dive")
    
    choice = input(f"\n{Fore.GREEN}Enter your choice (1-3):{Style.RESET_ALL} ")
    
    if choice == "1":
        # Process sample queries
        print_header("Sample Queries Demonstration")
        
        sample_categories = get_sample_queries()
        
        for category, queries in sample_categories.items():
            print_category_header(category)
            
            for i, query in enumerate(queries):
                try:
                    process_query(qa_chain, query, retriever, csv_data)
                    if i < len(queries) - 1:
                        input(f"\n{Fore.YELLOW}Press Enter to continue to the next query...{Style.RESET_ALL}")
                except Exception as e:
                    logger.error(f"Error processing query '{query}': {e}")
            
            if category != list(sample_categories.keys())[-1]:  # Not the last category
                input(f"\n{Fore.BLUE}Press Enter to see the next category...{Style.RESET_ALL}")
    
    elif choice == "2":
        # Interactive mode
        print_header("Interactive Query Mode")
        print(f"{Fore.CYAN}Ask any question about the product data (type 'exit' to quit):{Style.RESET_ALL}\n")
        
        # Show example question categories
        sample_categories = get_sample_queries()
        print(f"{Fore.CYAN}Example question categories:{Style.RESET_ALL}")
        for category, queries in sample_categories.items():
            print(f"{Fore.YELLOW}{category}:{Style.RESET_ALL} {queries[0]}")
        print()
        
        while True:
            query = input(f"{Fore.GREEN}Your question:{Style.RESET_ALL} ")
            if query.lower() in ["exit", "quit", "q"]:
                break
            
            try:
                process_query(qa_chain, query, retriever, csv_data)
            except Exception as e:
                logger.error(f"Error processing query: {e}")
    
    elif choice == "3":
        # Behind the scenes demo
        print_header("Behind the Scenes: How RAG Works")
        
        # Educational walkthrough
        print(f"{Fore.CYAN}This demonstration will show you exactly how RAG processes a question about our product catalog.{Style.RESET_ALL}\n")
        
        query = "What's the best laptop for gaming?"
        print(f"{Fore.GREEN}Sample question:{Style.RESET_ALL} {query}\n")
        
        # Step 1: Query embedding
        print(f"{Fore.YELLOW}STEP 1: Converting the question to a vector embedding{Style.RESET_ALL}")
        print("The system converts your natural language question into a mathematical vector.")
        embedding_model = OllamaEmbeddings(model="llama3.2")
        query_embedding = embedding_model.embed_query(query)
        print(f"Vector dimensions: {len(query_embedding)}")
        print(f"First 5 values: {query_embedding[:5]}\n")
        input(f"{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")
        
        # Step 2: Vector similarity search
        print(f"\n{Fore.YELLOW}STEP 2: Finding the most similar documents{Style.RESET_ALL}")
        print("The system compares your query vector with all document vectors to find relevant matches.")
        relevant_docs = retriever.get_relevant_documents(query)
        print(f"Found {len(relevant_docs)} relevant documents:\n")
        for i, doc in enumerate(relevant_docs):
            print(f"{Fore.WHITE}Document {i+1}:{Style.RESET_ALL}")
            print(f"{doc.page_content}\n")
        input(f"{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")
        
        # Step 3: Prompt construction
        print(f"\n{Fore.YELLOW}STEP 3: Constructing the prompt for the LLM{Style.RESET_ALL}")
        print("The system creates a prompt combining your question with the retrieved context:")
        prompt = f"""Use the following pieces of context to answer the question at the end.

Context:
"""
        for doc in relevant_docs:
            prompt += f"- {doc.page_content}\n"
        
        prompt += f"\nQuestion: {query}\nAnswer: "
        print(f"{Fore.WHITE}{prompt}{Style.RESET_ALL}\n")
        input(f"{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")
        
        # Step 4: LLM generation
        print(f"\n{Fore.YELLOW}STEP 4: Generating the answer with the LLM{Style.RESET_ALL}")
        print("The LLM generates a natural language answer based on the prompt:")
        llm = OllamaLLM(model="llama3.2", temperature=0.1)
        start_time = time.time()
        answer = llm.invoke(prompt)
        elapsed_time = time.time() - start_time
        print(f"{Fore.GREEN}Answer (generated in {elapsed_time:.2f} seconds):{Style.RESET_ALL}")
        print(f"{answer}\n")
        
        # Show direct data analysis approach
        print(f"\n{Fore.YELLOW}BONUS: Direct Data Analysis{Style.RESET_ALL}")
        print("For certain query types, the system can perform direct numerical analysis:")
        
        direct_query = "What are the products with the highest discounts?"
        print(f"{Fore.GREEN}Example direct analysis query:{Style.RESET_ALL} {direct_query}\n")
        
        direct_answer, _ = direct_data_analysis(direct_query, csv_data)
        if direct_answer:
            print(f"\n{Fore.GREEN}Final direct analysis answer:{Style.RESET_ALL}")
            print(f"{direct_answer}\n")
        
        print(f"{Fore.CYAN}That's how our product catalog RAG system works! This process allows for flexible and accurate answers about your product inventory.{Style.RESET_ALL}")
    
    else:
        print(f"{Fore.RED}Invalid choice. Exiting.{Style.RESET_ALL}")

if __name__ == "__main__":
    main()