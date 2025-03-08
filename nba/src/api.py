import pandas as pd
from functools import lru_cache

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

from .config import DEFAULT_SEASON

@lru_cache(maxsize=32)
def get_players():
    """Get all NBA players from static data or API"""
    try:
        # Try static data first (faster)
        all_players = players.get_players()
        if all_players:
            return pd.DataFrame(all_players)
    except Exception as e:
        print(f"Error retrieving players from static data: {e}")
    
    # Fall back to API
    try:
        all_players_df = commonallplayers.CommonAllPlayers().get_data_frames()[0]
        return all_players_df
    except Exception as e:
        print(f"Error retrieving players from API: {e}")
        return pd.DataFrame(columns=['PERSON_ID', 'DISPLAY_FIRST_LAST'])

def get_player_id(player_name):
    """Get player ID from name"""
    all_players = get_players()
    
    # Try exact match
    exact_match = all_players[all_players['DISPLAY_FIRST_LAST'] == player_name]
    if not exact_match.empty:
        return exact_match.iloc[0]['PERSON_ID']
    
    # Try contains match
    contains_match = all_players[all_players['DISPLAY_FIRST_LAST'].str.contains(player_name, case=False, na=False)]
    if not contains_match.empty:
        return contains_match.iloc[0]['PERSON_ID']
    
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
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star='Regular Season',
            last_n_games=last_n_games
        )
        return gamelog.get_data_frames()[0]
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