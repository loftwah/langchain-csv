import pandas as pd
import os
import json
import time
from functools import lru_cache
from pathlib import Path

# NBA API imports - using only confirmed endpoints
from nba_api.stats.static import players
from nba_api.stats.endpoints import (
    commonallplayers,
    playergamelog,
    leaguegamefinder,
    leaguedashplayerstats,
    leagueleaders,
    playercareerstats
)

from .config import DEFAULT_SEASON, CACHE_DIR

# Set up offline mode - will use cached data if True
OFFLINE_MODE = os.environ.get("NBA_OFFLINE_MODE", "0").lower() in ("1", "true", "yes")

def save_to_cache(data, cache_file):
    """Save data to cache file"""
    cache_path = Path(CACHE_DIR) / cache_file
    
    if isinstance(data, pd.DataFrame):
        data.to_csv(cache_path, index=False)
    else:
        with open(cache_path, 'w') as f:
            json.dump(data, f)
    
    print(f"Data saved to cache: {cache_file}")

def load_from_cache(cache_file, as_dataframe=True):
    """Load data from cache file"""
    cache_path = Path(CACHE_DIR) / cache_file
    
    if not cache_path.exists():
        return None
    
    try:
        if as_dataframe:
            return pd.read_csv(cache_path)
        else:
            with open(cache_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading from cache {cache_file}: {e}")
        return None

@lru_cache(maxsize=32)
def get_players():
    """Get all NBA players from static data or API"""
    # Try to load from cache first if in offline mode
    if OFFLINE_MODE:
        cached_data = load_from_cache("all_players.csv")
        if cached_data is not None:
            print("Using cached player data (offline mode)")
            return cached_data
    
    try:
        # Try static data first (faster)
        all_players = players.get_players()
        if all_players:
            df = pd.DataFrame(all_players)
            # Ensure we have standardized column names
            if 'full_name' in df.columns and 'DISPLAY_FIRST_LAST' not in df.columns:
                df['DISPLAY_FIRST_LAST'] = df['full_name']
            if 'id' in df.columns and 'PERSON_ID' not in df.columns:
                df['PERSON_ID'] = df['id']
            
            # Save to cache for offline use
            save_to_cache(df, "all_players.csv")
            return df
    except Exception as e:
        print(f"Error retrieving players from static data: {e}")
    
    # Fall back to API
    try:
        all_players_df = commonallplayers.CommonAllPlayers().get_data_frames()[0]
        # Save to cache for offline use
        save_to_cache(all_players_df, "all_players.csv")
        return all_players_df
    except Exception as e:
        print(f"Error retrieving players from API: {e}")
        
        # Last resort - try to load from cache even if not in offline mode
        cached_data = load_from_cache("all_players.csv")
        if cached_data is not None:
            print("API call failed. Using cached player data as fallback.")
            return cached_data
            
        return pd.DataFrame(columns=['PERSON_ID', 'DISPLAY_FIRST_LAST'])

def get_player_id(player_name):
    """Get player ID from name"""
    if not player_name or pd.isna(player_name) or player_name.strip() == "":
        return None
    
    # Normalize the input name    
    player_name = player_name.strip()
    
    # Handle common name variations manually
    name_variations = {
        'nikola jokic': ['Nikola Jokic', 'Nikola Jokić'],
        'giannis antetokounmpo': ['Giannis Antetokounmpo', 'Giannis Antetokounpo'],
        'luka doncic': ['Luka Doncic', 'Luka Dončić'],
        'nikola jokić': ['Nikola Jokic', 'Nikola Jokić'],
        'luka dončić': ['Luka Doncic', 'Luka Dončić'],
    }
    
    # Convert player name to lowercase for matching
    player_name_lower = player_name.lower()
    
    # Check if this player has known variations
    if player_name_lower in name_variations:
        # This is a player with variations, so we'll try all of them
        variation_list = name_variations[player_name_lower]
    else:
        # No known variations, just use the original name
        variation_list = [player_name]
    
    all_players = get_players()
    
    # Determine which column to use for player names
    if 'DISPLAY_FIRST_LAST' in all_players.columns:
        name_column = 'DISPLAY_FIRST_LAST'
    elif 'full_name' in all_players.columns:
        name_column = 'full_name'
    elif 'PLAYER_NAME' in all_players.columns:
        name_column = 'PLAYER_NAME'
    else:
        print(f"Warning: No recognizable name column in player data. Columns: {all_players.columns.tolist()}")
        return None
    
    # Determine which column to use for player IDs
    if 'PERSON_ID' in all_players.columns:
        id_column = 'PERSON_ID'
    elif 'id' in all_players.columns:
        id_column = 'id'
    elif 'PLAYER_ID' in all_players.columns:
        id_column = 'PLAYER_ID'
    else:
        print(f"Warning: No recognizable ID column in player data. Columns: {all_players.columns.tolist()}")
        return None
    
    # Try exact match with any of the known variations
    for variation in variation_list:
        exact_match = all_players[all_players[name_column] == variation]
        if not exact_match.empty:
            return exact_match.iloc[0][id_column]
    
    # If no exact match, try contains match
    try:
        # First try with the original name
        contains_match = all_players[all_players[name_column].str.contains(player_name, case=False, na=False)]
        if not contains_match.empty:
            return contains_match.iloc[0][id_column]
        
        # If that didn't work, try with variations
        for variation in variation_list:
            if variation != player_name:  # Skip the one we just tried
                contains_match = all_players[all_players[name_column].str.contains(variation, case=False, na=False)]
                if not contains_match.empty:
                    return contains_match.iloc[0][id_column]
        
        # Last resort: try matching on just the last name for players with distinctive last names
        if ' ' in player_name:
            last_name = player_name.split(' ')[-1]
            if len(last_name) > 3:  # Only try with substantial last names
                last_name_match = all_players[all_players[name_column].str.contains(last_name, case=False, na=False)]
                if not last_name_match.empty:
                    print(f"Found player by last name: {last_name_match.iloc[0][name_column]}")
                    return last_name_match.iloc[0][id_column]
    
    except Exception as e:
        print(f"Error in string matching for player '{player_name}': {e}")
    
    print(f"Could not find player with name '{player_name}'")
    return None

def get_league_leaders(season=DEFAULT_SEASON, stat_category="PTS", per_mode="PerGame", limit=50):
    """Get league leaders for a specific stat category"""
    try:
        leaders = leagueleaders.LeagueLeaders(
            season=season,
            stat_category_abbreviation=stat_category,
            per_mode48=per_mode,
            season_type_all_star="Regular Season"
        )
        leaders_df = leaders.get_data_frames()[0]
        return leaders_df.head(limit)
    except Exception as e:
        print(f"Error fetching league leaders: {e}")
        return pd.DataFrame()

def get_player_stats(season=DEFAULT_SEASON, min_games=20):
    """Get comprehensive stats for all players"""
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed='PerGame',
            measure_type_detailed_defense='Base',
            season_type_all_star='Regular Season',
            pace_adjust='N',
            plus_minus='N',
            rank='N',
            last_n_games=0
        )
        stats_df = stats.get_data_frames()[0]
        
        # Filter by minimum games
        if min_games > 0:
            stats_df = stats_df[stats_df['GP'] >= min_games]
            
        return stats_df
    except Exception as e:
        print(f"Error fetching player stats: {e}")
        return pd.DataFrame()

def get_player_games(player_id, season=DEFAULT_SEASON, last_n_games=10):
    """Get last N games for a player"""
    try:
        # PlayerGameLog doesn't accept last_n_games as a direct parameter
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star='Regular Season'
        )
        
        # Get all games and then filter to the last N games
        games_df = gamelog.get_data_frames()[0]
        
        # Return only the last N games if we have enough games
        if not games_df.empty and len(games_df) > 0:
            # Sort by date (newest first) if needed
            if 'GAME_DATE' in games_df.columns:
                games_df = games_df.sort_values('GAME_DATE', ascending=False)
            
            # Return only the requested number of games
            return games_df.head(last_n_games)
        
        return games_df
    except Exception as e:
        print(f"Error fetching game logs for player {player_id}: {e}")
        return pd.DataFrame()

def get_player_career(player_id):
    """Get player career stats"""
    try:
        career = playercareerstats.PlayerCareerStats(player_id=player_id)
        return career.get_data_frames()[0]
    except Exception as e:
        print(f"Error fetching career stats for player {player_id}: {e}")
        return pd.DataFrame() 