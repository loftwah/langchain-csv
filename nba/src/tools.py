import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nba_api.stats.static import players
import time
import re
from functools import lru_cache
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .config import NBA_COLORS, DEFAULT_SEASON
from .api import (
    get_league_leaders, 
    get_player_stats, 
    get_player_id, 
    get_player_games,
    get_players,
    get_player_career,
    get_player_historical_data
)
from .fantasy import calculate_fantasy_points, calculate_consistency

# Set matplotlib style for dark mode
plt.style.use('dark_background')

# Import the new game simulator
from .game import game_simulator as refactored_game_simulator

# Import the new consistency tracker
from .tracker import consistency_tracker as refactored_consistency_tracker

def draft_helper(scoring_system='standard', min_games=20, stat_category="PTS"):
    """Find value players for fantasy drafts"""
    # Get league leaders for the specified stat category
    leaders_df = get_league_leaders(stat_category=stat_category)
    
    # Get comprehensive player stats
    stats_df = get_player_stats(min_games=min_games)
    
    # Use whichever dataset has more data
    if not leaders_df.empty and not stats_df.empty:
        # If both are available, prefer the more comprehensive dataset
        if len(stats_df) > len(leaders_df):
            data_df = stats_df
        else:
            # Rename columns for consistency if needed
            if 'PLAYER' in leaders_df.columns and 'PLAYER_NAME' not in leaders_df.columns:
                leaders_df = leaders_df.rename(columns={'PLAYER': 'PLAYER_NAME'})
            data_df = leaders_df
    else:
        # Use whichever one is not empty
        data_df = stats_df if not stats_df.empty else leaders_df
    
    if data_df.empty:
        return "Could not retrieve player data. API might be unavailable.", None
    
    # Calculate fantasy points
    fantasy_df = calculate_fantasy_points(data_df, scoring_system)
    
    # Calculate value metrics
    if 'MIN' in fantasy_df.columns:
        fantasy_df['VALUE'] = fantasy_df['FANTASY_POINTS'] / fantasy_df['MIN']
    else:
        fantasy_df['VALUE'] = fantasy_df['FANTASY_POINTS']
    
    # Sort by value
    sorted_df = fantasy_df.sort_values('VALUE', ascending=False)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=NBA_COLORS['background'])
    
    # Get top 20 players
    top_df = sorted_df.head(20)
    
    # Determine player name column
    player_col = 'PLAYER_NAME' if 'PLAYER_NAME' in top_df.columns else 'PLAYER'
    
    # Create color gradient based on values
    max_value = top_df['FANTASY_POINTS'].max()
    norm = plt.Normalize(0, max_value)
    colors = plt.cm.coolwarm(norm(top_df['FANTASY_POINTS']))
    
    # Create bar chart
    bars = ax.barh(top_df[player_col], top_df['FANTASY_POINTS'], color=colors)
    
    # Add labels
    ax.set_xlabel('Fantasy Points per Game', color=NBA_COLORS['accent'], fontsize=12)
    ax.set_title(f'Top 20 Players by Fantasy Value ({scoring_system.title()} Scoring)', 
                 fontsize=16, color=NBA_COLORS['accent'], pad=20)
    ax.invert_yaxis()  # Highest value at the top
    
    # Style the chart for dark mode
    ax.set_facecolor(NBA_COLORS['background'])
    ax.spines['bottom'].set_color(NBA_COLORS['accent'])
    ax.spines['top'].set_color(NBA_COLORS['accent'])
    ax.spines['left'].set_color(NBA_COLORS['accent'])
    ax.spines['right'].set_color(NBA_COLORS['accent'])
    ax.tick_params(axis='both', colors=NBA_COLORS['accent'])
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f"{top_df['VALUE'].iloc[i]:.2f}", 
                ha='left', va='center', color=NBA_COLORS['accent'])
    
    # Add Loftwah branding to the plot
    fig.text(0.95, 0.02, "Created by Loftwah", fontsize=10, 
             ha='right', va='bottom', color=NBA_COLORS['highlight'],
             url="https://linkarooie.com/loftwah")
    
    # Add subtle grid
    ax.grid(True, linestyle='--', alpha=0.2, color=NBA_COLORS['accent'])
    
    plt.tight_layout()
    
    # Format dataframe for display
    display_cols = [col for col in [player_col, 'TEAM_ABBREVIATION', 'GP', 'MIN', 
                   'PTS', 'REB', 'AST', 'STL', 'BLK', 'FG_PCT', 'FT_PCT', 'FG3M', 'TOV',
                   'FANTASY_POINTS', 'VALUE'] if col in sorted_df.columns]
    
    display_df = sorted_df[display_cols].head(50)
    
    return display_df, fig

def matchup_analyzer(team1_players, team2_players, scoring_system='standard'):
    """Compare two fantasy teams using recent player performance"""
    # Process team1 players
    team1_stats = []
    for player in team1_players.split(','):
        player = player.strip()
        if not player:
            continue
            
        player_id = get_player_id(player)
        if player_id:
            # Get recent games
            recent_games = get_player_games(player_id)
            
            if not recent_games.empty:
                # Calculate average stats
                avg_stats = recent_games.mean(numeric_only=True)
                avg_stats['PLAYER_NAME'] = player
                team1_stats.append(avg_stats)
    
    # Process team2 players
    team2_stats = []
    for player in team2_players.split(','):
        player = player.strip()
        if not player:
            continue
            
        player_id = get_player_id(player)
        if player_id:
            # Get recent games
            recent_games = get_player_games(player_id)
            
            if not recent_games.empty:
                # Calculate average stats
                avg_stats = recent_games.mean(numeric_only=True)
                avg_stats['PLAYER_NAME'] = player
                team2_stats.append(avg_stats)
    
    # Convert to DataFrames
    team1_df = pd.DataFrame(team1_stats) if team1_stats else pd.DataFrame()
    team2_df = pd.DataFrame(team2_stats) if team2_stats else pd.DataFrame()
    
    if team1_df.empty or team2_df.empty:
        return "Could not find enough player data. Check player names and try again.", None
    
    # Calculate team totals
    categories = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'TOV']
    categories = [c for c in categories if c in team1_df.columns and c in team2_df.columns]
    
    percentages = ['FG_PCT', 'FT_PCT']
    percentages = [p for p in percentages if p in team1_df.columns and p in team2_df.columns]
    
    team1_totals = {cat: team1_df[cat].sum() for cat in categories}
    team2_totals = {cat: team2_df[cat].sum() for cat in categories}
    
    # For percentages, use simple averages
    for pct in percentages:
        team1_totals[pct] = team1_df[pct].mean()
        team2_totals[pct] = team2_df[pct].mean()
    
    # Create visualization
    fig, axs = plt.subplots(3, 3, figsize=(15, 12), facecolor=NBA_COLORS['background'])
    axs = axs.flatten()
    
    all_cats = categories + percentages
    colors = [NBA_COLORS['primary'], NBA_COLORS['secondary']]  # Primary for team 1, secondary for team 2
    
    for i, cat in enumerate(all_cats):
        if i >= len(axs):  # Ensure we don't exceed available subplots
            break
            
        # Set subplot style
        axs[i].set_facecolor(NBA_COLORS['background'])
        axs[i].spines['bottom'].set_color(NBA_COLORS['accent'])
        axs[i].spines['top'].set_color(NBA_COLORS['accent'])
        axs[i].spines['left'].set_color(NBA_COLORS['accent'])
        axs[i].spines['right'].set_color(NBA_COLORS['accent'])
        axs[i].tick_params(axis='both', colors=NBA_COLORS['accent'])
        
        # For turnovers, lower is better
        if cat == 'TOV':
            heights = [team2_totals[cat], team1_totals[cat]]
            labels = ['Team 2', 'Team 1']
            color_order = [colors[1], colors[0]]
        else:
            heights = [team1_totals[cat], team2_totals[cat]]
            labels = ['Team 1', 'Team 2']
            color_order = colors
        
        bars = axs[i].bar(labels, heights, color=color_order, alpha=0.8)
        axs[i].set_title(cat, color=NBA_COLORS['accent'], fontsize=12)
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            axs[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', 
                        color=NBA_COLORS['accent'], fontweight='bold')
        
        # Add subtle grid
        axs[i].grid(True, linestyle='--', alpha=0.2, color=NBA_COLORS['accent'])
    
    # Calculate expected category wins
    team1_wins = 0
    team2_wins = 0
    
    for cat in all_cats:
        # For TOV, lower is better
        if cat == 'TOV':
            if team1_totals[cat] < team2_totals[cat]:
                team1_wins += 1
            elif team2_totals[cat] < team1_totals[cat]:
                team2_wins += 1
        else:
            if team1_totals[cat] > team2_totals[cat]:
                team1_wins += 1
            elif team2_totals[cat] > team1_totals[cat]:
                team2_wins += 1
    
    # Display winner projection
    if len(axs) > len(all_cats):
        last_ax = axs[len(all_cats)]
        last_ax.axis('off')
        last_ax.set_facecolor(NBA_COLORS['background'])
        
        # Create a stylish winner announcement
        winner_color = NBA_COLORS['primary'] if team1_wins > team2_wins else NBA_COLORS['secondary'] if team2_wins > team1_wins else NBA_COLORS['highlight']
        
        if team1_wins > team2_wins:
            winner_text = f"Team 1 wins ({team1_wins}-{team2_wins})"
        elif team2_wins > team1_wins:
            winner_text = f"Team 2 wins ({team2_wins}-{team1_wins})"
        else:
            winner_text = f"Tie ({team1_wins}-{team2_wins})"
        
        last_ax.text(0.5, 0.5, "PROJECTED MATCHUP RESULT", fontsize=14, 
                    ha='center', va='center', color=NBA_COLORS['accent'],
                    transform=last_ax.transAxes)
        last_ax.text(0.5, 0.4, winner_text, fontsize=18, fontweight='bold',
                    ha='center', va='center', color=winner_color,
                    transform=last_ax.transAxes)
    
    # Hide any unused subplots
    for i in range(len(all_cats) + 1, len(axs)):
        axs[i].axis('off')
        axs[i].set_facecolor(NBA_COLORS['background'])
    
    # Add Loftwah branding to the plot
    fig.text(0.95, 0.02, "Created by Loftwah", fontsize=10, 
             ha='right', va='bottom', color=NBA_COLORS['highlight'],
             url="https://linkarooie.com/loftwah")
    
    # Add a title to the entire figure
    fig.suptitle("Fantasy Basketball Matchup Analysis", fontsize=20, 
                color=NBA_COLORS['accent'], y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Create summary text with rich styling for markdown
    summary = f"""
    ## 🏀 Team Comparison: Team 1 vs Team 2
    
    ### 📊 Projected Category Wins:
    - **Team 1**: {team1_wins}
    - **Team 2**: {team2_wins}
    - **Result**: {"Team 1 Wins! 🔥" if team1_wins > team2_wins else "Team 2 Wins! 🔥" if team2_wins > team1_wins else "Tie Game! ⚖️"}
    
    ### 👥 Team 1 Players:
    {", ".join(team1_df['PLAYER_NAME'].tolist())}
    
    ### 👥 Team 2 Players:
    {", ".join(team2_df['PLAYER_NAME'].tolist())}
    
    ### 📈 Category Breakdown:
    """
    
    for cat in all_cats:
        val1 = team1_totals[cat]
        val2 = team2_totals[cat]
        
        # For TOV, lower is better
        if cat == 'TOV':
            winner = "Team 1 ✅" if val1 < val2 else "Team 2 ✅" if val2 < val1 else "Tie ⚖️"
        else:
            winner = "Team 1 ✅" if val1 > val2 else "Team 2 ✅" if val2 > val1 else "Tie ⚖️"
            
        summary += f"- **{cat}**: Team 1 ({val1:.1f}) vs Team 2 ({val2:.1f}) - Winner: {winner}\n"
    
    # Add Loftwah credit to the summary
    summary += "\n\n*Analysis provided by [Loftwah's Fantasy Basketball Assistant](https://linkarooie.com/loftwah)*"
    
    return summary, fig

def consistency_tracker(player_name, num_games=10, scoring_system='standard'):
    """Analyze a player's consistency in fantasy performance"""
    # Use the refactored consistency tracker
    return refactored_consistency_tracker(
        player_name=player_name, 
        num_games=num_games, 
        scoring_system=scoring_system
    )

def game_simulator(team1_players, team2_players, team1_name="Team 1", team2_name="Team 2", quarters=4, quarter_length=12):
    """
    Simulate a full basketball game between two teams of NBA players.
    
    Args:
        team1_players (str): Comma-separated list of players on team 1
        team2_players (str): Comma-separated list of players on team 2
        team1_name (str): Name for team 1
        team2_name (str): Name for team 2
        quarters (int): Number of quarters to play
        quarter_length (int): Length of each quarter in minutes
        
    Returns:
        tuple: (play_by_play_text, stats_visualization)
    """
    # Use the refactored game simulator
    return refactored_game_simulator(
        team1_players=team1_players,
        team2_players=team2_players,
        team1_name=team1_name,
        team2_name=team2_name,
        quarters=quarters,
        quarter_length=quarter_length
    )

def normalize_stats(df, column):
    """Helper function to normalize a column to use as probability weights"""
    if column not in df.columns or df[column].sum() == 0:
        return np.ones(len(df)) / len(df)  # Equal probabilities if no valid data
    
    values = df[column].values
    # Replace NaNs with minimum value
    values[np.isnan(values)] = np.nanmin(values) if np.nanmin(values) > 0 else 0.1
    # Ensure no negative values
    values = np.maximum(values, 0.1)
    # Normalize to sum to 1
    return values / np.sum(values) 