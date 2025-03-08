"""
Fantasy Basketball Tools - A suite of interactive tools for fantasy basketball managers
Built using the NBA API with tools for draft help, matchup analysis, and player consistency tracking
"""

# Import key components to make them available at the package level
from .api import get_players, get_player_id, get_player_stats, get_league_leaders
from .fantasy import calculate_fantasy_points, calculate_consistency
from .tools import draft_helper, matchup_analyzer, consistency_tracker
from .ui import create_interface 