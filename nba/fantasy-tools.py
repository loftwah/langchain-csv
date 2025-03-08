#!/usr/bin/env python3
"""
Fantasy Basketball Tools - A suite of interactive tools for fantasy basketball managers
"""

from nba_api.stats.static import players
import time
import sys

# Import from our refactored package
from src.ui import create_interface

def print_with_animation(text, delay=0.05):
    """Print text with a typing animation effect"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

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
        
    print_with_animation("Setting up interface...", 0.03)
    demo = create_interface()
    print_with_animation("🚀 Launching interface... Opening in your browser!", 0.03)
    demo.launch(share=False, inbrowser=True) 