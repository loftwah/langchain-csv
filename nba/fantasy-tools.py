#!/usr/bin/env python3
"""
Fantasy Basketball Tools - A suite of interactive tools for fantasy basketball managers
Powered by NBA API, LangChain, and Ollama with Llama 3.2
"""

from nba_api.stats.static import players
import time
import sys
import os
import threading
import subprocess

# Import from our refactored package
from src.ui import create_interface

def print_with_animation(text, delay=0.05):
    """Print text with a typing animation effect"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def check_ollama_availability():
    """Check if Ollama is available and try to start it if needed"""
    try:
        # Try to import Ollama from langchain_ollama
        from langchain_ollama import OllamaLLM
        
        # Try to initialize Ollama with a simple request
        ollama = OllamaLLM(model="llama3.2")
        response = ollama.invoke("Say hello")
        
        print_with_animation("✅ Successfully connected to Ollama")
        print_with_animation(f"   LLM response: {response[:50]}...")
        return True
    except Exception as e:
        print_with_animation(f"⚠️ Ollama not available: {e}")
        
        # Check if user wants to try starting Ollama
        if os.environ.get("NBA_AUTO_START_OLLAMA", "0").lower() in ("1", "true", "yes"):
            print_with_animation("🔄 Attempting to start Ollama service...")
            try:
                # Try to start Ollama in the background
                subprocess.Popen(
                    ["ollama", "run", "llama3.2"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                print_with_animation("Started Ollama in background. AI features might become available shortly.")
            except Exception as start_error:
                print_with_animation(f"❌ Failed to start Ollama: {start_error}")
                print_with_animation("To use AI features, please start Ollama manually: ollama run llama3.2")
        else:
            print_with_animation("ℹ️ To use AI features, please start Ollama manually: ollama run llama3.2")
            print_with_animation("   Or set NBA_AUTO_START_OLLAMA=1 to attempt automatic startup")
        
        return False

if __name__ == "__main__":
    print_with_animation("🏀 Starting Fantasy Basketball Tools...", 0.03)
    print_with_animation("Loading NBA data - this may take a moment...", 0.03)
    
    print("Connecting to NBA API", end="")
    for _ in range(5):
        time.sleep(0.3)
        print(".", end="", flush=True)
    print()
    
    # Test API connectivity
    try:
        test_players = players.get_players()
        print_with_animation(f"✅ Successfully connected to NBA API. Found {len(test_players)} players in static data.")
    except Exception as e:
        print_with_animation(f"⚠️ Warning: Could not load static player data: {e}")
        print_with_animation("Will attempt to use API endpoints directly.")
    
    # Check Ollama in a separate thread to not block startup
    threading.Thread(target=check_ollama_availability, daemon=True).start()
    
    print_with_animation("Setting up interface...", 0.03)
    demo = create_interface()
    print_with_animation("🚀 Launching interface... Opening in your browser!", 0.03)
    demo.launch(share=False, inbrowser=True) 