#!/usr/bin/env python
"""
Test script to validate player name matching with special characters.
"""

import sys
from src.api import get_player_id, get_players

def test_player_matching():
    """Test player name matching with special characters."""
    test_cases = [
        "Nikola Jokic",      # Without special character
        "Nikola Jokić",      # With special character
        "Jokic",             # Just last name without special character 
        "Jokić",             # Just last name with special character
        "Luka Doncic",       # Another player with special character in last name
        "Giannis Antetokounmpo",  # Long name
    ]
    
    print("Testing player name matching...")
    print("-" * 50)
    
    # Try to get all player names for reference
    all_players = get_players()
    name_column = 'DISPLAY_FIRST_LAST' if 'DISPLAY_FIRST_LAST' in all_players.columns else 'full_name'
    if not all_players.empty and name_column in all_players.columns:
        print(f"Found {len(all_players)} players in database.")
    else:
        print("Warning: Could not retrieve player names for reference.")
    
    # Test each case
    for name in test_cases:
        player_id = get_player_id(name)
        
        if player_id:
            # Look up the player's official name
            if not all_players.empty:
                id_column = 'PERSON_ID' if 'PERSON_ID' in all_players.columns else 'id'
                match = all_players[all_players[id_column].astype(str) == player_id]
                if not match.empty:
                    official_name = match.iloc[0][name_column]
                    print(f"✅ '{name}' matched to '{official_name}' (ID: {player_id})")
                else:
                    print(f"⚠️ '{name}' matched to ID {player_id}, but couldn't find name in database")
            else:
                print(f"⚠️ '{name}' matched to ID {player_id}, but couldn't verify name")
        else:
            print(f"❌ No match found for '{name}'")
    
    print("\nDone.")

if __name__ == "__main__":
    test_player_matching() 