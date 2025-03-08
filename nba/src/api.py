import pandas as pd
import os
import json
import time
import datetime
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
from .api_cache import cache_api_response

# Set up offline mode - will use cached data if True
OFFLINE_MODE = os.environ.get("NBA_OFFLINE_MODE", "0").lower() in ("1", "true", "yes")

# Legacy caching functions - will be deprecated
def save_to_cache(data, cache_file):
    """Save data to cache file (DEPRECATED: Use the cache_api_response decorator instead)"""
    cache_path = Path(CACHE_DIR) / cache_file
    
    if isinstance(data, pd.DataFrame):
        data.to_csv(cache_path, index=False)
    else:
        with open(cache_path, 'w') as f:
            json.dump(data, f)
    
    print(f"Data saved to cache: {cache_file}")

def load_from_cache(cache_file, as_dataframe=True):
    """Load data from cache file (DEPRECATED: Use the cache_api_response decorator instead)"""
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

@cache_api_response(cache_timeout=7*24*60*60, as_dataframe=True)
def get_players():
    """Get all NBA players from static data or API"""
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
            
            return df
    except Exception as e:
        print(f"Error retrieving players from static data: {e}")
    
    # Fall back to API
    try:
        all_players_df = commonallplayers.CommonAllPlayers().get_data_frames()[0]
        return all_players_df
    except Exception as e:
        print(f"Error retrieving players from API: {e}")
        return pd.DataFrame(columns=['PERSON_ID', 'DISPLAY_FIRST_LAST'])

@lru_cache(maxsize=128)
def get_player_id(player_name):
    """Get player ID from name using fuzzy matching"""
    if not player_name:
        return None
        
    # Get all players and standardize names
    all_players = get_players()
    if all_players.empty:
        return None
    
    # Determine which column to use for names
    name_column = 'DISPLAY_FIRST_LAST' if 'DISPLAY_FIRST_LAST' in all_players.columns else 'full_name'
    id_column = 'PERSON_ID' if 'PERSON_ID' in all_players.columns else 'id'
    
    if name_column not in all_players.columns or id_column not in all_players.columns:
        print("ERROR: Required columns not found in player data")
        return None
    
    # First try exact match
    exact_match = all_players[all_players[name_column].str.lower() == player_name.lower()]
    if not exact_match.empty:
        return str(exact_match.iloc[0][id_column])
    
    # Try contains match (case insensitive)
    contains_matches = all_players[all_players[name_column].str.lower().str.contains(player_name.lower())]
    if not contains_matches.empty:
        return str(contains_matches.iloc[0][id_column])
        
    # No match found
    return None

@cache_api_response(cache_timeout=7*24*60*60, as_dataframe=True, cache_subdir="player_games")
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

@cache_api_response(cache_timeout=30*24*60*60, as_dataframe=True, cache_subdir="player_career")
def get_player_career(player_id):
    """Get player career stats"""
    try:
        career = playercareerstats.PlayerCareerStats(player_id=player_id)
        return career.get_data_frames()[0]
    except Exception as e:
        print(f"Error fetching career stats for player {player_id}: {e}")
        return pd.DataFrame()

@cache_api_response(cache_timeout=365*24*60*60, as_dataframe=True, cache_subdir="player_historical")
def get_player_historical_data(player_id):
    """Get historical data for a player (especially useful for retired players)"""
    try:
        # First try to get career stats
        career_stats = get_player_career(player_id)
        
        if not career_stats.empty:
            # Calculate per game averages if needed
            if 'GP' in career_stats.columns and career_stats['GP'].sum() > 0:
                return career_stats
        
        # Fallback - try to fetch data in a different way
        # This would depend on the specific implementation needed for historical players
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Error fetching historical data for player {player_id}: {e}")
        return pd.DataFrame()

@cache_api_response(cache_timeout=24*60*60, as_dataframe=True, cache_subdir="league_leaders")
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

@cache_api_response(cache_timeout=24*60*60, as_dataframe=True, cache_subdir="player_stats")
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

@cache_api_response(cache_timeout=24*60*60, as_dataframe=True, cache_subdir="games")
def get_games(season=DEFAULT_SEASON, season_type="Regular Season", team_id=None, date_from=None, date_to=None):
    """Get game data for a specific season, team, or date range"""
    try:
        # Create parameters dictionary
        params = {
            'season': season,
            'season_type_nullable': season_type
        }
        
        # Add optional parameters if provided
        if team_id:
            params['team_id_nullable'] = team_id
        if date_from:
            params['date_from_nullable'] = date_from
        if date_to:
            params['date_to_nullable'] = date_to
        
        games = leaguegamefinder.LeagueGameFinder(**params)
        games_df = games.get_data_frames()[0]
        
        return games_df
        
    except Exception as e:
        print(f"Error fetching games: {e}")
        return pd.DataFrame() 