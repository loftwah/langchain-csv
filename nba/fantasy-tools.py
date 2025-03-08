#!/usr/bin/env python3
"""
Fantasy Basketball Tools - A suite of interactive tools for fantasy basketball managers
"""

from nba_api.stats.static import players

# Import from our refactored package
from src.ui import create_interface

if __name__ == "__main__":
    print("Starting Fantasy Basketball Tools...")
    print("Loading NBA data - this may take a moment...")
    
    # Test API connectivity
    try:
        test_players = players.get_players()
        print(f"Successfully connected to NBA API. Found {len(test_players)} players in static data.")
    except Exception as e:
        print(f"Warning: Could not load static player data: {e}")
        print("Will attempt to use API endpoints directly.")
        
    demo = create_interface()
    print("Launching interface...")
    demo.launch() 