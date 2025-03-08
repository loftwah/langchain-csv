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
    # Get player ID
    player_id = get_player_id(player_name)
    
    if not player_id:
        print(f"DEBUG: Could not find player ID for '{player_name}'. Checking available player names...")
        # Get first 5 players from dataset as an example
        all_players = get_players()
        name_column = 'DISPLAY_FIRST_LAST' if 'DISPLAY_FIRST_LAST' in all_players.columns else 'full_name'
        if name_column in all_players.columns:
            sample_players = all_players[name_column].head(5).tolist()
            print(f"DEBUG: Sample of available players: {sample_players}")
        return f"No player found matching '{player_name}'. Please check the spelling and try again.", None
    
    # Get recent games
    recent_games = get_player_games(player_id, last_n_games=num_games)
    
    if recent_games.empty:
        return f"No recent game data found for {player_name}", None
    
    # Calculate consistency metrics
    mean_fp, cv, min_fp, max_fp, fp_trend = calculate_consistency(recent_games, scoring_system)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   facecolor=NBA_COLORS['background'])
    
    # Style for dark mode
    for ax in [ax1, ax2]:
        ax.set_facecolor(NBA_COLORS['background'])
        for spine in ax.spines.values():
            spine.set_color(NBA_COLORS['accent'])
        ax.tick_params(axis='both', colors=NBA_COLORS['accent'])
    
    # Game-by-game fantasy points trend with gradient color
    game_indices = list(range(len(fp_trend)))
    
    # Create color gradient based on performance
    norm = plt.Normalize(min(fp_trend), max(fp_trend))
    colors = plt.cm.coolwarm(norm(fp_trend))
    
    # Plot points and connect with gradient line
    for i in range(len(game_indices) - 1):
        ax1.plot(game_indices[i:i+2], fp_trend[i:i+2], 
                color=colors[i], linewidth=2, alpha=0.8)
        
    # Add points on top
    scatter = ax1.scatter(game_indices, fp_trend, c=fp_trend, cmap='coolwarm', 
                         s=100, zorder=5, edgecolor='white', linewidth=1)
    
    # Add labels
    ax1.set_xlabel('Game Number (Most Recent First)', color=NBA_COLORS['accent'], fontsize=12)
    ax1.set_ylabel('Fantasy Points', color=NBA_COLORS['accent'], fontsize=12)
    ax1.set_title(f'{player_name} - Fantasy Performance Analysis', 
                 fontsize=16, color=NBA_COLORS['accent'], pad=20)
    
    # Add mean line
    mean_line = ax1.axhline(y=mean_fp, color=NBA_COLORS['highlight'], 
                           linestyle='--', linewidth=2,
                           label=f'Average: {mean_fp:.1f}')
    
    # Add min and max lines
    min_line = ax1.axhline(y=min_fp, color='#00FF00', linestyle=':', linewidth=1.5,
                         label=f'Min: {min_fp:.1f}')
    max_line = ax1.axhline(y=max_fp, color='#FF00FF', linestyle=':', linewidth=1.5,
                         label=f'Max: {max_fp:.1f}')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Fantasy Points', color=NBA_COLORS['accent'])
    cbar.ax.yaxis.set_tick_params(color=NBA_COLORS['accent'])
    cbar.outline.set_edgecolor(NBA_COLORS['accent'])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=NBA_COLORS['accent'])
    
    ax1.grid(True, linestyle='--', alpha=0.2, color=NBA_COLORS['accent'])
    ax1.legend(framealpha=0.7, facecolor=NBA_COLORS['background'], 
              edgecolor=NBA_COLORS['accent'], loc='upper right')
    
    # Add consistency visualization
    consistency_score = 100 * (1 - min(cv, 1))  # Convert CV to a 0-100 scale (higher is more consistent)
    
    # Custom gauge chart for consistency
    consistency_categories = [
        (0, 20, 'Very Inconsistent', '#FF3030'),
        (20, 40, 'Inconsistent', '#FF8C00'),
        (40, 60, 'Moderate', '#FFFF00'),
        (60, 80, 'Consistent', '#7CFC00'),
        (80, 100, 'Very Consistent', '#00FF7F')
    ]
    
    # Find player's consistency category
    for low, high, label, color in consistency_categories:
        if low <= consistency_score < high:
            consistency_label = label
            consistency_color = color
            break
    else:
        consistency_label = "Unknown"
        consistency_color = "gray"
    
    # Create gauge chart effect
    gauge_colors = [cat[3] for cat in consistency_categories]
    gauge_positions = np.linspace(0, 100, len(gauge_colors) + 1)[:-1]
    gauge_widths = [20] * len(gauge_colors)
    
    ax2.barh(y=[1] * len(gauge_positions), width=gauge_widths, left=gauge_positions, 
             color=gauge_colors, alpha=0.8, height=0.5)
    
    # Add pointer for this player's consistency
    ax2.scatter(consistency_score, 1, color='white', edgecolor='black',
               zorder=10, s=400, marker='^')
    
    # Add consistency score and label with glowing effect
    for offset in [-1, 1]:  # Create shadow effect
        ax2.text(50 + offset*0.5, 1.5 + offset*0.1, 
                f"Consistency Score: {consistency_score:.1f}/100", 
                ha='center', va='center', fontsize=14, 
                color='black', alpha=0.3)
        
    ax2.text(50, 1.5, f"Consistency Score: {consistency_score:.1f}/100", 
             ha='center', va='center', fontsize=14, fontweight='bold',
             color=NBA_COLORS['accent'])
             
    ax2.text(50, 1.0, f"Rating: {consistency_label}", 
             ha='center', va='center', fontsize=16, fontweight='bold',
             color=consistency_color)
    
    # Set up ax2 styling
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 2.5)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    # Add Loftwah branding to the plot
    fig.text(0.95, 0.02, "Created by Loftwah", fontsize=10, 
             ha='right', va='bottom', color=NBA_COLORS['highlight'],
             url="https://linkarooie.com/loftwah")
    
    plt.tight_layout()
    
    # Get player's stats for each recent game
    if not recent_games.empty and 'GAME_DATE' in recent_games.columns:
        # Format game log for display
        game_log = recent_games.copy()
        
        # Calculate fantasy points
        fantasy_games = calculate_fantasy_points(game_log, scoring_system)
        
        # Format date
        if 'GAME_DATE' in fantasy_games.columns:
            fantasy_games['GAME_DATE'] = pd.to_datetime(fantasy_games['GAME_DATE']).dt.strftime('%Y-%m-%d')
        
        # Create summary text with emojis and rich formatting
        summary = f"""
        ## 📊 {player_name} Consistency Analysis
        
        ### 🔍 Overview:
        - **Average Fantasy Points**: {mean_fp:.1f} pts
        - **Consistency Score**: {consistency_score:.1f}/100 ({consistency_label})
        - **Range**: {min_fp:.1f} to {max_fp:.1f} fantasy points
        - **Games Analyzed**: {len(fp_trend)}
        
        ### ⚖️ Consistency Interpretation:
        This player is **{consistency_label.lower()}** in their fantasy production. 
        {"💯 Their high consistency makes them a reliable starter each week." if consistency_score >= 60 else
         "👍 Their moderate consistency means they're generally reliable but can have off games." if consistency_score >= 40 else
         "🎲 Their inconsistency makes them a boom-or-bust player who can win or lose your matchup."}
        
        ### 📈 Recent Performance Trend:
        {"🔥 Their production has been trending upward recently." if sum(fp_trend[:3]) > sum(fp_trend[-3:]) else
         "➡️ Their production has been fairly stable recently." if abs(sum(fp_trend[:3]) - sum(fp_trend[-3:])) < mean_fp * 0.1 else
         "📉 Their production has been trending downward recently."}
        
        *Analysis provided by [Loftwah's Fantasy Basketball Assistant](https://linkarooie.com/loftwah)*
        """
        
        return summary, fig
    
    else:
        return f"Insufficient game data for {player_name}", fig 

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
    # Process team players
    team1_stats = []
    team2_stats = []
    
    # Process team1 players
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
            else:
                # Try to get historical data for retired players
                historical_data = get_player_historical_data(player_id)
                if not historical_data.empty:
                    avg_stats = historical_data.mean(numeric_only=True)
                    avg_stats['PLAYER_NAME'] = player
                    print(f"Using historical data for {player}")
                    team1_stats.append(avg_stats)
    
    # Process team2 players
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
            else:
                # Try to get historical data for retired players
                historical_data = get_player_historical_data(player_id)
                if not historical_data.empty:
                    avg_stats = historical_data.mean(numeric_only=True)
                    avg_stats['PLAYER_NAME'] = player
                    print(f"Using historical data for {player}")
                    team2_stats.append(avg_stats)
                else:
                    print(f"⚠️ Could not find sufficient data for {player}")
    
    # Convert to DataFrames
    team1_df = pd.DataFrame(team1_stats) if team1_stats else pd.DataFrame()
    team2_df = pd.DataFrame(team2_stats) if team2_stats else pd.DataFrame()
    
    if team1_df.empty or team2_df.empty:
        return "Could not find enough player data. Check player names and try again.", "<div style='text-align:center; padding:20px; background-color:#1e1e1e; color:white; border-radius:10px;'><h2>⚠️ Simulation Error</h2><p>Insufficient player data available. Please check player names and try again.</p></div>"
    
    # List of recognized NBA legends for enhanced gameplay description
    legends = [
        "Michael Jordan", "Kobe Bryant", "LeBron James", "Magic Johnson", 
        "Larry Bird", "Wilt Chamberlain", "Kareem Abdul-Jabbar", "Hakeem Olajuwon", 
        "Shaquille O'Neal", "Bill Russell", "Tim Duncan", "Julius Erving", 
        "Stephen Curry", "Kevin Durant", "Oscar Robertson", "Jerry West",
        "Allen Iverson", "Charles Barkley", "Dirk Nowitzki", "John Stockton"
    ]
    
    # Initialize game data
    team1_score = 0
    team2_score = 0
    
    # Initialize team game stats
    team1_game_stats = {player: {'PTS': 0, 'AST': 0, 'REB': 0, 'FG': 0, 'FGA': 0, 'FG3': 0, 'FG3A': 0, 'FT': 0, 'FTA': 0, 'TO': 0, 'STL': 0, 'BLK': 0} for player in team1_df['PLAYER_NAME']}
    team2_game_stats = {player: {'PTS': 0, 'AST': 0, 'REB': 0, 'FG': 0, 'FGA': 0, 'FG3': 0, 'FG3A': 0, 'FT': 0, 'FTA': 0, 'TO': 0, 'STL': 0, 'BLK': 0} for player in team2_df['PLAYER_NAME']}
    
    # Each play will be stored as text in this list
    play_by_play = []
    
    # Record quarter-by-quarter scores for plotting
    team1_quarter_scores = []
    team2_quarter_scores = []
    
    # Helper function for normalized probabilities
    def normalize_stats(df, column):
        if column not in df.columns or df.empty:
            # Return equal probability for each player
            return [1.0 / len(df)] * len(df)
        
        values = df[column].values
        # Replace NaNs with minimum value
        values[np.isnan(values)] = np.nanmin(values) if np.nanmin(values) > 0 else 0.1
        # Ensure no negative values
        values = np.maximum(values, 0.1)
        # Normalize to sum to 1
        return values / np.sum(values)

    # Calculate team offensive and defensive ratings
    team1_off_rtg = team1_df['PTS'].sum() / len(team1_df) if 'PTS' in team1_df.columns else 25
    team2_off_rtg = team2_df['PTS'].sum() / len(team2_df) if 'PTS' in team2_df.columns else 25
    
    team1_def_rtg = team1_df['STL'].sum() + team1_df['BLK'].sum() if 'STL' in team1_df.columns and 'BLK' in team1_df.columns else 5
    team2_def_rtg = team2_df['STL'].sum() + team2_df['BLK'].sum() if 'STL' in team2_df.columns and 'BLK' in team2_df.columns else 5
    
    # Possession variables
    team1_possessions = []
    team2_possessions = []
    
    # Add game intro
    play_by_play.append(f"# 🏀 {team1_name} vs {team2_name} - GAME SIMULATION")
    play_by_play.append(f"\n## Starting Lineups")
    play_by_play.append(f"\n### {team1_name}:")
    play_by_play.append(", ".join(team1_df['PLAYER_NAME'].tolist()))
    play_by_play.append(f"\n### {team2_name}:")
    play_by_play.append(", ".join(team2_df['PLAYER_NAME'].tolist()))
    play_by_play.append("\n---\n")
    
    # Simulate each quarter
    for quarter in range(1, quarters + 1):
        play_by_play.append(f"\n## Quarter {quarter}")
        
        # Initialize quarter scores
        quarter_team1_score = 0
        quarter_team2_score = 0
        
        # Calculate possessions for the quarter (random but based on team pace)
        possessions = int(quarter_length * 2.2)  # ~2.2 possessions per minute
        
        for poss in range(possessions):
            # Alternate possessions, with slight randomness
            if np.random.random() < 0.5:
                # Team 1 has possession
                offensive_team = team1_name
                defensive_team = team2_name
                off_players = team1_df['PLAYER_NAME'].tolist()
                def_players = team2_df['PLAYER_NAME'].tolist()
                off_ratings = team1_df
                def_ratings = team2_df
                game_stats = team1_game_stats
            else:
                # Team 2 has possession
                offensive_team = team2_name
                defensive_team = team1_name
                off_players = team2_df['PLAYER_NAME'].tolist()
                def_players = team1_df['PLAYER_NAME'].tolist()
                off_ratings = team2_df
                def_ratings = team1_df
                game_stats = team2_game_stats
            
            # Select players involved in this play
            primary_player = np.random.choice(off_players, p=normalize_stats(off_ratings, 'PTS'))
            
            # Get player's actual efficiency stats
            player_idx = off_ratings[off_ratings['PLAYER_NAME'] == primary_player].index[0]
            fg_pct = off_ratings.loc[player_idx, 'FG_PCT'] if 'FG_PCT' in off_ratings.columns else 0.45
            fg3_pct = off_ratings.loc[player_idx, 'FG3_PCT'] if 'FG3_PCT' in off_ratings.columns else 0.35
            
            # Determine play type
            play_types = ['three_point', 'mid_range', 'layup', 'dunk', 'pass', 'turnover']
            play_probs = [0.25, 0.25, 0.25, 0.1, 0.1, 0.05]  # Default probabilities
            
            # Adjust probabilities based on player stats
            if 'FG3M' in off_ratings.columns:
                three_tendency = off_ratings.loc[player_idx, 'FG3M'] / max(1, off_ratings.loc[player_idx, 'PTS']) * 3
                play_probs[0] = max(0.1, min(0.4, three_tendency * 0.8))
            
            play_probs = [p/sum(play_probs) for p in play_probs]  # Normalize probabilities
            play_type = np.random.choice(play_types, p=play_probs)
            
            # Initialize play result
            points_scored = 0
            play_text = ""
            
            # Handle different play types
            if play_type == 'turnover':
                # Turnover
                defender = np.random.choice(def_players)
                
                # Enhanced descriptions for legends
                if primary_player in legends:
                    if defender in legends:
                        # Legendary matchup
                        turnover_types = [
                            f"In a clash of titans, {defender} forces {primary_player} into a rare turnover!",
                            f"{defender} shows why he's one of the greatest defenders, stealing the ball from {primary_player}!",
                            f"The legendary {primary_player} loses the handle, and {defender} is there to capitalize!",
                            f"An epic defensive play by {defender}, stripping the ball from the great {primary_player}!"
                        ]
                    else:
                        turnover_types = [
                            f"The normally flawless {primary_player} commits an uncharacteristic turnover!",
                            f"{primary_player} loses the ball in a rare mistake!",
                            f"Even the great {primary_player} can make a mistake - turnover!",
                            f"{defender} will remember this forever - stealing the ball from the legendary {primary_player}!"
                        ]
                else:
                    # Standard turnover descriptions
                    turnover_types = [
                        f"{primary_player} loses the ball out of bounds!",
                        f"{defender} strips {primary_player} for the steal!",
                        f"{primary_player} throws a pass that's intercepted by {defender}!",
                        f"{primary_player} called for an offensive foul, turnover!"
                    ]
                play_text = np.random.choice(turnover_types)
                
                # Update stats
                game_stats[primary_player]['TO'] += 1
                
            elif play_type == 'pass':
                # Assist to another player
                receivers = [p for p in off_players if p != primary_player]
                if receivers:
                    receiver = np.random.choice(receivers)
                    
                    # Determine shot type
                    if np.random.random() < 0.3:  # 30% chance of three pointer
                        success_prob = fg3_pct - 0.05  # Slightly harder than player's avg
                        
                        if np.random.random() < success_prob:
                            points_scored = 3
                            play_text = f"{primary_player} finds {receiver} behind the arc... BANG! A beautiful three-pointer!"
                            
                            # Update stats
                            game_stats[receiver]['PTS'] += 3
                            game_stats[receiver]['FG'] += 1
                            game_stats[receiver]['FGA'] += 1
                            game_stats[receiver]['FG3'] += 1
                            game_stats[receiver]['FG3A'] += 1
                            game_stats[primary_player]['AST'] += 1
                        else:
                            play_text = f"{primary_player} kicks it out to {receiver} for three... but it rims out!"
                            
                            # Update stats
                            game_stats[receiver]['FGA'] += 1
                            game_stats[receiver]['FG3A'] += 1
                            
                            # 25% chance of offensive rebound
                            if np.random.random() < 0.25:
                                rebounder = np.random.choice(off_players)
                                play_text += f" {rebounder} gets the offensive rebound!"
                                game_stats[rebounder]['REB'] += 1
                            else:
                                rebounder = np.random.choice(def_players)
                                play_text += f" {rebounder} secures the defensive rebound."
                    else:
                        # Two pointer after pass
                        success_prob = fg_pct
                        
                        if np.random.random() < success_prob:
                            points_scored = 2
                            play_text = f"{primary_player} with a perfect pass to {receiver} who scores!"
                            
                            # Update stats
                            game_stats[receiver]['PTS'] += 2
                            game_stats[receiver]['FG'] += 1
                            game_stats[receiver]['FGA'] += 1
                            game_stats[primary_player]['AST'] += 1
                        else:
                            play_text = f"{primary_player} passes to {receiver} who misses the shot!"
                            
                            # Update stats
                            game_stats[receiver]['FGA'] += 1
                            
                            # Handle rebound
                            if np.random.random() < 0.25:
                                rebounder = np.random.choice(off_players)
                                play_text += f" {rebounder} grabs the offensive board!"
                                game_stats[rebounder]['REB'] += 1
                            else:
                                rebounder = np.random.choice(def_players)
                                play_text += f" Rebound {defensive_team}, {rebounder}."
                                # No need to update defensive team stats
            
            else:  # Direct shot attempts
                defender = np.random.choice(def_players)
                
                # Base success probability for different shots
                success_probs = {
                    'three_point': fg3_pct,
                    'mid_range': fg_pct + 0.05,  # Easier than average FG
                    'layup': fg_pct + 0.15,      # Quite high success rate
                    'dunk': 0.85                 # Very high success rate
                }
                
                # Defender adjustment
                def_idx = def_ratings[def_ratings['PLAYER_NAME'] == defender].index[0]
                def_impact = def_ratings.loc[def_idx, 'BLK'] * 0.02 if 'BLK' in def_ratings.columns else 0
                
                # Shot is blocked?
                is_blocked = np.random.random() < def_impact
                
                if is_blocked:
                    play_text = f"{primary_player} drives to the basket... BLOCKED by {defender}!"
                    # Update stats
                    game_stats[primary_player]['FGA'] += 1
                    # No FG3A update needed here as blocks typically happen inside
                    
                    # Sometimes a foul occurs on the block (And-1 opportunity)
                    if np.random.random() < 0.15:  # 15% chance of a foul on the block
                        play_text += f" But a foul is called on {defender}! {primary_player} goes to the line for free throws."
                        
                        # Free throw attempts (typically 2)
                        ft_attempts = 2
                        made_fts = 0
                        
                        # Calculate free throw percentage (use 75% if not available)
                        ft_pct = off_ratings.loc[player_idx, 'FT_PCT'] if 'FT_PCT' in off_ratings.columns else 0.75
                        
                        # Simulate free throws
                        for _ in range(ft_attempts):
                            if np.random.random() < ft_pct:
                                made_fts += 1
                        
                        # Update stats
                        game_stats[primary_player]['FT'] += made_fts
                        game_stats[primary_player]['FTA'] += ft_attempts
                        game_stats[primary_player]['PTS'] += made_fts
                        
                        play_text += f" {primary_player} makes {made_fts} of {ft_attempts} from the line."
                    
                    # Handle rebound after block
                    if np.random.random() < 0.6:  # Defense more likely to get rebound after block
                        rebounder = np.random.choice(def_players)
                        play_text += f" {rebounder} recovers the ball."
                    else:
                        rebounder = np.random.choice(off_players)
                        play_text += f" But {rebounder} gets the loose ball!"
                        game_stats[rebounder]['REB'] += 1
                
                else:  # Not blocked
                    # Calculate final success probability
                    success_prob = success_probs[play_type] - (def_impact / 2)
                    
                    # Shot descriptions
                    shot_descriptions = {
                        'three_point': [
                            f"{primary_player} pulls up from deep...",
                            f"{primary_player} gets space behind the arc...",
                            f"{primary_player} with the step-back three...",
                            f"The ball swings to {primary_player} at the corner..."
                        ],
                        'mid_range': [
                            f"{primary_player} with the elbow jumper...",
                            f"{primary_player} pulls up from 15 feet...",
                            f"{primary_player} with the fadeaway..."
                        ],
                        'layup': [
                            f"{primary_player} drives past {defender}...",
                            f"{primary_player} finds a lane to the basket...",
                            f"{primary_player} with a spin move into the paint..."
                        ],
                        'dunk': [
                            f"{primary_player} blows by {defender}...",
                            f"{primary_player} cuts to the basket...",
                            f"{primary_player} gets the step on {defender}..."
                        ]
                    }
                    
                    # Enhanced descriptions for legends
                    if primary_player in legends:
                        legend_descriptions = {
                            'three_point': [
                                f"The legendary {primary_player} sizes up the defense from beyond the arc...",
                                f"{primary_player} with that iconic shooting form from downtown...",
                                f"From way downtown, it's {primary_player} with the signature move...",
                                f"{primary_player} creates just enough space for the long-range bomb..."
                            ],
                            'mid_range': [
                                f"{primary_player} with the iconic fadeaway jumper that made him famous...",
                                f"The patented {primary_player} mid-range move that defenders have nightmares about...",
                                f"{primary_player} rises up with that textbook form from the elbow..."
                            ],
                            'layup': [
                                f"{primary_player} showcases that legendary footwork on the drive...",
                                f"With that signature grace, {primary_player} glides to the hoop...",
                                f"{primary_player} with the move that made him a Hall of Famer..."
                            ],
                            'dunk': [
                                f"{primary_player} elevates like in his prime...",
                                f"Turning back the clock, {primary_player} rises up...",
                                f"The crowd is on its feet as {primary_player} soars toward the rim..."
                            ]
                        }
                        
                        # Use legend descriptions instead of standard ones
                        shot_text = np.random.choice(legend_descriptions[play_type])
                    else:
                        shot_text = np.random.choice(shot_descriptions[play_type])
                    
                    if np.random.random() < success_prob:
                        # Shot successful
                        if play_type == 'three_point':
                            points_scored = 3
                            play_text = f"{shot_text} BANG! {points_scored} points!"
                            game_stats[primary_player]['FG3'] += 1
                            game_stats[primary_player]['FG3A'] += 1
                        elif play_type == 'dunk':
                            points_scored = 2
                            play_text = f"{shot_text} AND THROWS IT DOWN! Monster jam by {primary_player}!"
                        else:
                            points_scored = 2
                            play_text = f"{shot_text} and it's GOOD! {points_scored} points!"
                        
                        # Update stats for all successful shots
                        game_stats[primary_player]['PTS'] += points_scored
                        game_stats[primary_player]['FG'] += 1
                        game_stats[primary_player]['FGA'] += 1
                        
                        # Check for and-one opportunity on drives to the basket
                        if play_type in ['layup', 'dunk'] and np.random.random() < 0.15:  # 15% chance of and-one
                            play_text += f" AND ONE! Foul on {defender}!"
                            
                            # Shoot one free throw
                            ft_pct = off_ratings.loc[player_idx, 'FT_PCT'] if 'FT_PCT' in off_ratings.columns else 0.75
                            made_ft = 1 if np.random.random() < ft_pct else 0
                            
                            # Update stats
                            game_stats[primary_player]['FTA'] += 1
                            game_stats[primary_player]['FT'] += made_ft
                            game_stats[primary_player]['PTS'] += made_ft
                            
                            if made_ft:
                                play_text += f" {primary_player} completes the three-point play!"
                                points_scored += 1  # Update points_scored to include the free throw
                            else:
                                play_text += f" {primary_player} misses the free throw."
                        
                    else:
                        # Shot missed
                        miss_descriptions = [
                            "but it's no good!",
                            "but it rims out!",
                            "but it's a bit short!",
                            "but it's off the mark!"
                        ]
                        miss_text = np.random.choice(miss_descriptions)
                        play_text = f"{shot_text} {miss_text}"
                        
                        # Check for shooting foul on drives (higher chance on drives)
                        foul_chance = 0.25 if play_type in ['layup', 'dunk'] else 0.08
                        if np.random.random() < foul_chance:
                            play_text += f" But a foul is called on {defender}! {primary_player} goes to the line."
                            
                            # Determine number of free throws
                            ft_attempts = 3 if play_type == 'three_point' else 2
                            ft_pct = off_ratings.loc[player_idx, 'FT_PCT'] if 'FT_PCT' in off_ratings.columns else 0.75
                            made_fts = 0
                            
                            # Simulate free throws
                            for _ in range(ft_attempts):
                                if np.random.random() < ft_pct:
                                    made_fts += 1
                            
                            # Update stats
                            game_stats[primary_player]['FT'] += made_fts
                            game_stats[primary_player]['FTA'] += ft_attempts
                            game_stats[primary_player]['PTS'] += made_fts
                            points_scored = made_fts  # Update points_scored for team score calculation
                            
                            play_text += f" {primary_player} makes {made_fts} of {ft_attempts} from the line."
                        else:
                            # Update stats for missed shots
                            game_stats[primary_player]['FGA'] += 1
                            if play_type == 'three_point':
                                game_stats[primary_player]['FG3A'] += 1
                            
                            # Handle rebound
                            if np.random.random() < 0.25:  # 25% chance of offensive rebound
                                rebounder = np.random.choice(off_players)
                                play_text += f" {rebounder} gets the offensive rebound!"
                                game_stats[rebounder]['REB'] += 1
                            else:
                                rebounder = np.random.choice(def_players)
                                play_text += f" {rebounder} grabs the defensive board."
            
            # Update team scores
            if offensive_team == team1_name and points_scored > 0:
                team1_score += points_scored
                quarter_team1_score += points_scored  # Update quarter score
            elif offensive_team == team2_name and points_scored > 0:
                team2_score += points_scored
                quarter_team2_score += points_scored  # Update quarter score
            
            # Format the possession with score
            possession_text = f"{offensive_team} {team1_score}-{team2_score} {defensive_team}: {play_text}"
            
            # Add dramatic commentary for certain situations
            if abs(team1_score - team2_score) >= 15:
                if team1_score > team2_score and offensive_team == team1_name and points_scored > 0:
                    possession_text += " They're pulling away!"
                elif team2_score > team1_score and offensive_team == team2_name and points_scored > 0:
                    possession_text += " The lead continues to grow!"
            elif abs(team1_score - team2_score) <= 5 and (quarter == quarters) and poss > possessions * 0.7:
                possession_text += " This is a close one down the stretch!"
            
            # Add to play-by-play
            play_by_play.append(f"* {possession_text}")
            
            # Add quarter breaks and special commentary
            if poss == int(possessions * 0.3):
                play_by_play.append(f"\n**Current Score:** {team1_name} {team1_score} - {team2_score} {team2_name}")
            elif poss == int(possessions * 0.7):
                play_by_play.append(f"\n**Current Score:** {team1_name} {team1_score} - {team2_score} {team2_name}")
                
        # Record quarter scores for visualization
        team1_quarter_scores.append(quarter_team1_score)
        team2_quarter_scores.append(quarter_team2_score)
        
        # End of quarter summary
        play_by_play.append(f"\n### End of Quarter {quarter}")
        play_by_play.append(f"**Score:** {team1_name} {team1_score} - {team2_score} {team2_name}")
        
        # Add storylines between quarters
        if quarter < quarters:
            if abs(team1_score - team2_score) <= 3:
                play_by_play.append("This game is coming down to the wire! Both teams are battling hard.")
            elif team1_score > team2_score + 10:
                play_by_play.append(f"{team2_name} needs to find an answer quickly if they want to get back in this game.")
            elif team2_score > team1_score + 10:
                play_by_play.append(f"{team1_name} will need to regroup and find a way to cut into this deficit.")
            
            play_by_play.append("\n---\n")
    
    # Final game summary
    play_by_play.append("\n## 🏁 Final Score")
    play_by_play.append(f"### {team1_name} {team1_score} - {team2_score} {team2_name}")
    
    # Determine winner
    if team1_score > team2_score:
        play_by_play.append(f"\n**{team1_name} WINS!**")
        winner = team1_name
    elif team2_score > team1_score:
        play_by_play.append(f"\n**{team2_name} WINS!**")
        winner = team2_name
    else:
        play_by_play.append("\n**IT'S A TIE!**")
        winner = "Tie"
    
    # Compile player stats
    play_by_play.append("\n## 📊 Box Score")
    
    # Team 1 Stats
    play_by_play.append(f"\n### {team1_name}")
    play_by_play.append("| Player | PTS | REB | AST | STL | BLK | FG | 3PT | FT | TO |")
    play_by_play.append("|--------|-----|-----|-----|-----|-----|----|----|----|----|")
    
    for player, stats in team1_game_stats.items():
        fg_text = f"{stats['FG']}/{stats['FGA']}"
        fg3_text = f"{stats['FG3']}/{stats['FG3A']}"
        ft_text = f"{stats['FT']}/{stats['FTA']}"
        
        play_by_play.append(f"| {player} | {stats['PTS']} | {stats['REB']} | {stats['AST']} | " + 
                           f"{stats['STL']} | {stats['BLK']} | {fg_text} | {fg3_text} | {ft_text} | {stats['TO']} |")
    
    # Team 2 Stats
    play_by_play.append(f"\n### {team2_name}")
    play_by_play.append("| Player | PTS | REB | AST | STL | BLK | FG | 3PT | FT | TO |")
    play_by_play.append("|--------|-----|-----|-----|-----|-----|----|----|----|----|")
    
    for player, stats in team2_game_stats.items():
        fg_text = f"{stats['FG']}/{stats['FGA']}"
        fg3_text = f"{stats['FG3']}/{stats['FG3A']}"
        ft_text = f"{stats['FT']}/{stats['FTA']}"
        
        play_by_play.append(f"| {player} | {stats['PTS']} | {stats['REB']} | {stats['AST']} | " + 
                           f"{stats['STL']} | {stats['BLK']} | {fg_text} | {fg3_text} | {ft_text} | {stats['TO']} |")
    
    # Determine game MVP
    all_stats = {}
    for player, stats in {**team1_game_stats, **team2_game_stats}.items():
        # Simplified PER calculation
        per = stats['PTS'] + stats['REB'] + stats['AST'] * 1.5 + stats['STL'] * 2 + stats['BLK'] * 2 - stats['TO']
        all_stats[player] = per
    
    mvp = max(all_stats.items(), key=lambda x: x[1])[0]
    mvp_stats = team1_game_stats.get(mvp) or team2_game_stats.get(mvp)
    
    play_by_play.append("\n## 🌟 Game MVP")
    play_by_play.append(f"**{mvp}** with {mvp_stats['PTS']} points, {mvp_stats['REB']} rebounds, " + 
                      f"and {mvp_stats['AST']} assists!")
    
    # Add footer
    play_by_play.append("\n---")
    play_by_play.append("*Simulation by Loftwah's Fantasy Basketball Tools*")
    
    # Join play-by-play into a single string
    play_by_play_text = "\n".join(play_by_play)
    
    # Create an enhanced HTML visualization
    # Extract top scorers from each team
    team1_top = sorted([(p, s['PTS']) for p, s in team1_game_stats.items()], key=lambda x: x[1], reverse=True)
    team2_top = sorted([(p, s['PTS']) for p, s in team2_game_stats.items()], key=lambda x: x[1], reverse=True)
    
    # Calculate team totals
    team1_totals = {
        'PTS': sum(s['PTS'] for s in team1_game_stats.values()),
        'REB': sum(s['REB'] for s in team1_game_stats.values()),
        'AST': sum(s['AST'] for s in team1_game_stats.values()),
        'STL': sum(s['STL'] for s in team1_game_stats.values()),
        'BLK': sum(s['BLK'] for s in team1_game_stats.values()),
        'TO': sum(s['TO'] for s in team1_game_stats.values())
    }
    
    team2_totals = {
        'PTS': sum(s['PTS'] for s in team2_game_stats.values()),
        'REB': sum(s['REB'] for s in team2_game_stats.values()),
        'AST': sum(s['AST'] for s in team2_game_stats.values()),
        'STL': sum(s['STL'] for s in team2_game_stats.values()),
        'BLK': sum(s['BLK'] for s in team2_game_stats.values()),
        'TO': sum(s['TO'] for s in team2_game_stats.values())
    }
    
    # Determine the winner and set colors
    if team1_score > team2_score:
        winner = team1_name
        winner_color = "#17408B"  # Team 1 blue color
    elif team2_score > team1_score:
        winner = team2_name
        winner_color = "#C9082A"  # Team 2 red color
    else:
        winner = "Tie"
        winner_color = "#F7B801"  # Gold color for tie
    
    # Create a fancy HTML visualization
    html_content = f"""
    <div style="font-family: Arial, sans-serif; padding: 20px; color: white; background-color: #121212;">
        <!-- Title Banner -->
        <div style="background: linear-gradient(90deg, #17408B, #C9082A); padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0; font-size: 28px;">{team1_name} vs {team2_name}</h1>
            <h2 style="color: #F7B801; margin: 10px 0 0 0; font-size: 24px;">Final Score: {team1_score} - {team2_score}</h2>
        </div>
        
        <!-- Quarter by Quarter Breakdown -->
        <div style="background-color: rgba(40, 40, 50, 0.8); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: white; margin-top: 0; text-align: center;">Quarter by Quarter</h2>
            <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                <thead>
                    <tr style="border-bottom: 2px solid rgba(255,255,255,0.3);">
                        <th style="padding: 8px; text-align: left; color: #F7B801;">Team</th>
                        <th style="padding: 8px; text-align: center; color: #F7B801;">Q1</th>
                        <th style="padding: 8px; text-align: center; color: #F7B801;">Q2</th>
                        <th style="padding: 8px; text-align: center; color: #F7B801;">Q3</th>
                        <th style="padding: 8px; text-align: center; color: #F7B801;">Q4</th>
                        <th style="padding: 8px; text-align: center; color: #F7B801;">Final</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                        <td style="padding: 8px; text-align: left; font-weight: bold; color: #17408B;">{team1_name}</td>
                        <td style="padding: 8px; text-align: center;">{team1_quarter_scores[0] if len(team1_quarter_scores) > 0 else 0}</td>
                        <td style="padding: 8px; text-align: center;">{team1_quarter_scores[1] if len(team1_quarter_scores) > 1 else 0}</td>
                        <td style="padding: 8px; text-align: center;">{team1_quarter_scores[2] if len(team1_quarter_scores) > 2 else 0}</td>
                        <td style="padding: 8px; text-align: center;">{team1_quarter_scores[3] if len(team1_quarter_scores) > 3 else 0}</td>
                        <td style="padding: 8px; text-align: center; font-weight: bold;">{team1_score}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; text-align: left; font-weight: bold; color: #C9082A;">{team2_name}</td>
                        <td style="padding: 8px; text-align: center;">{team2_quarter_scores[0] if len(team2_quarter_scores) > 0 else 0}</td>
                        <td style="padding: 8px; text-align: center;">{team2_quarter_scores[1] if len(team2_quarter_scores) > 1 else 0}</td>
                        <td style="padding: 8px; text-align: center;">{team2_quarter_scores[2] if len(team2_quarter_scores) > 2 else 0}</td>
                        <td style="padding: 8px; text-align: center;">{team2_quarter_scores[3] if len(team2_quarter_scores) > 3 else 0}</td>
                        <td style="padding: 8px; text-align: center; font-weight: bold;">{team2_score}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <!-- MVP Section -->
        <div style="background-color: rgba(40, 40, 50, 0.8); padding: 15px; border-radius: 10px; margin-bottom: 20px; border: 2px solid #F7B801;">
            <h2 style="color: #F7B801; margin-top: 0; text-align: center;">🌟 GAME MVP: {mvp}</h2>
            <p style="text-align: center; font-size: 18px;">{mvp_stats['PTS']} PTS | {mvp_stats['REB']} REB | {mvp_stats['AST']} AST | {mvp_stats['STL']} STL | {mvp_stats['BLK']} BLK</p>
            <p style="text-align: center; font-size: 16px;">FG: {mvp_stats['FG']}/{mvp_stats['FGA']} | 3PT: {mvp_stats['FG3']}/{mvp_stats['FG3A']} | TO: {mvp_stats['TO']}</p>
        </div>
        
        <!-- Team Stats Side by Side -->
        <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px;">
            <div style="flex: 1; min-width: 300px; background-color: rgba(23, 64, 139, 0.3); padding: 15px; border-radius: 10px; border-left: 4px solid #17408B;">
                <h3 style="color: white; margin-top: 0;">{team1_name} Team Stats</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">
                        <td style="padding: 8px 0; color: rgba(255,255,255,0.8);">Points</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: bold;">{team1_totals['PTS']}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">
                        <td style="padding: 8px 0; color: rgba(255,255,255,0.8);">Rebounds</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: bold;">{team1_totals['REB']}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">
                        <td style="padding: 8px 0; color: rgba(255,255,255,0.8);">Assists</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: bold;">{team1_totals['AST']}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">
                        <td style="padding: 8px 0; color: rgba(255,255,255,0.8);">Steals</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: bold;">{team1_totals['STL']}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">
                        <td style="padding: 8px 0; color: rgba(255,255,255,0.8);">Blocks</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: bold;">{team1_totals['BLK']}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: rgba(255,255,255,0.8);">Turnovers</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: bold;">{team1_totals['TO']}</td>
                    </tr>
                </table>
            </div>
            
            <div style="flex: 1; min-width: 300px; background-color: rgba(201, 8, 42, 0.3); padding: 15px; border-radius: 10px; border-left: 4px solid #C9082A;">
                <h3 style="color: white; margin-top: 0;">{team2_name} Team Stats</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">
                        <td style="padding: 8px 0; color: rgba(255,255,255,0.8);">Points</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: bold;">{team2_totals['PTS']}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">
                        <td style="padding: 8px 0; color: rgba(255,255,255,0.8);">Rebounds</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: bold;">{team2_totals['REB']}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">
                        <td style="padding: 8px 0; color: rgba(255,255,255,0.8);">Assists</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: bold;">{team2_totals['AST']}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">
                        <td style="padding: 8px 0; color: rgba(255,255,255,0.8);">Steals</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: bold;">{team2_totals['STL']}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">
                        <td style="padding: 8px 0; color: rgba(255,255,255,0.8);">Blocks</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: bold;">{team2_totals['BLK']}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: rgba(255,255,255,0.8);">Turnovers</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: bold;">{team2_totals['TO']}</td>
                    </tr>
                </table>
            </div>
        </div>
        
        <!-- Team 1 Player Stats Table -->
        <div style="margin-bottom: 20px; background-color: rgba(30,30,40,0.9); padding: 15px; border-radius: 10px;">
            <h3 style="color: #17408B; margin-top: 0; border-bottom: 2px solid #17408B; padding-bottom: 8px;">
                {team1_name} Player Stats
            </h3>
            <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="border-bottom: 2px solid rgba(255,255,255,0.2);">
                            <th style="text-align: left; padding: 10px 15px; color: white;">Player</th>
                            <th style="text-align: center; padding: 10px 15px; color: white;">PTS</th>
                            <th style="text-align: center; padding: 10px 15px; color: white;">REB</th>
                            <th style="text-align: center; padding: 10px 15px; color: white;">AST</th>
                            <th style="text-align: center; padding: 10px 15px; color: white;">STL</th>
                            <th style="text-align: center; padding: 10px 15px; color: white;">BLK</th>
                            <th style="text-align: center; padding: 10px 15px; color: white;">FG</th>
                            <th style="text-align: center; padding: 10px 15px; color: white;">3PT</th>
                            <th style="text-align: center; padding: 10px 15px; color: white;">TO</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    # Add Team 1 player rows
    for player, stats in sorted(team1_game_stats.items(), key=lambda x: x[1]['PTS'], reverse=True):
        fg_pct = f"{(stats['FG']/stats['FGA']*100):.1f}%" if stats['FGA'] > 0 else "0.0%"
        fg3_pct = f"{(stats['FG3']/stats['FG3A']*100):.1f}%" if stats['FG3A'] > 0 else "0.0%"
        
        html_content += f"""
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                            <td style="text-align: left; padding: 8px 15px; font-weight: bold;">{player}</td>
                            <td style="text-align: center; padding: 8px 15px;">{stats['PTS']}</td>
                            <td style="text-align: center; padding: 8px 15px;">{stats['REB']}</td>
                            <td style="text-align: center; padding: 8px 15px;">{stats['AST']}</td>
                            <td style="text-align: center; padding: 8px 15px;">{stats['STL']}</td>
                            <td style="text-align: center; padding: 8px 15px;">{stats['BLK']}</td>
                            <td style="text-align: center; padding: 8px 15px;">{stats['FG']}/{stats['FGA']} ({fg_pct})</td>
                            <td style="text-align: center; padding: 8px 15px;">{stats['FG3']}/{stats['FG3A']} ({fg3_pct})</td>
                            <td style="text-align: center; padding: 8px 15px;">{stats['TO']}</td>
                        </tr>
        """
    
    html_content += f"""
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Team 2 Player Stats Table -->
        <div style="margin-bottom: 20px; background-color: rgba(30,30,40,0.9); padding: 15px; border-radius: 10px;">
            <h3 style="color: #C9082A; margin-top: 0; border-bottom: 2px solid #C9082A; padding-bottom: 8px;">
                {team2_name} Player Stats
            </h3>
            <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="border-bottom: 2px solid rgba(255,255,255,0.2);">
                            <th style="text-align: left; padding: 10px 15px; color: white;">Player</th>
                            <th style="text-align: center; padding: 10px 15px; color: white;">PTS</th>
                            <th style="text-align: center; padding: 10px 15px; color: white;">REB</th>
                            <th style="text-align: center; padding: 10px 15px; color: white;">AST</th>
                            <th style="text-align: center; padding: 10px 15px; color: white;">STL</th>
                            <th style="text-align: center; padding: 10px 15px; color: white;">BLK</th>
                            <th style="text-align: center; padding: 10px 15px; color: white;">FG</th>
                            <th style="text-align: center; padding: 10px 15px; color: white;">3PT</th>
                            <th style="text-align: center; padding: 10px 15px; color: white;">TO</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    # Add Team 2 player rows
    for player, stats in sorted(team2_game_stats.items(), key=lambda x: x[1]['PTS'], reverse=True):
        fg_pct = f"{(stats['FG']/stats['FGA']*100):.1f}%" if stats['FGA'] > 0 else "0.0%"
        fg3_pct = f"{(stats['FG3']/stats['FG3A']*100):.1f}%" if stats['FG3A'] > 0 else "0.0%"
        
        html_content += f"""
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                            <td style="text-align: left; padding: 8px 15px; font-weight: bold;">{player}</td>
                            <td style="text-align: center; padding: 8px 15px;">{stats['PTS']}</td>
                            <td style="text-align: center; padding: 8px 15px;">{stats['REB']}</td>
                            <td style="text-align: center; padding: 8px 15px;">{stats['AST']}</td>
                            <td style="text-align: center; padding: 8px 15px;">{stats['STL']}</td>
                            <td style="text-align: center; padding: 8px 15px;">{stats['BLK']}</td>
                            <td style="text-align: center; padding: 8px 15px;">{stats['FG']}/{stats['FGA']} ({fg_pct})</td>
                            <td style="text-align: center; padding: 8px 15px;">{stats['FG3']}/{stats['FG3A']} ({fg3_pct})</td>
                            <td style="text-align: center; padding: 8px 15px;">{stats['TO']}</td>
                        </tr>
        """
    
    html_content += f"""
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Top Scorers -->
        <div style="margin-bottom: 20px; background-color: rgba(30,30,40,0.9); padding: 15px; border-radius: 10px;">
            <h3 style="color: #F7B801; margin-top: 0; border-bottom: 2px solid #F7B801; padding-bottom: 8px;">
                Top Scorers
            </h3>
            <div style="display: flex; flex-wrap: wrap; justify-content: space-around; gap: 15px; margin-top: 15px;">
    """
    
    # Add top 3 scorers from each team
    for i, (player, pts) in enumerate(team1_top[:3]):
        player_stats = team1_game_stats[player]
        html_content += f"""
                <div style="background-color: rgba(23, 64, 139, 0.3); padding: 12px; border-radius: 8px; min-width: 200px; text-align: center;">
                    <h4 style="margin-top: 0; color: white;">{player}</h4>
                    <p style="font-size: 24px; font-weight: bold; margin: 5px 0;">{pts} pts</p>
                    <p style="margin: 5px 0; font-size: 14px;">
                        {player_stats['REB']} reb | {player_stats['AST']} ast | {player_stats['FG']}/{player_stats['FGA']} FG
                    </p>
                </div>
        """
    
    for i, (player, pts) in enumerate(team2_top[:3]):
        player_stats = team2_game_stats[player]
        html_content += f"""
                <div style="background-color: rgba(201, 8, 42, 0.3); padding: 12px; border-radius: 8px; min-width: 200px; text-align: center;">
                    <h4 style="margin-top: 0; color: white;">{player}</h4>
                    <p style="font-size: 24px; font-weight: bold; margin: 5px 0;">{pts} pts</p>
                    <p style="margin: 5px 0; font-size: 14px;">
                        {player_stats['REB']} reb | {player_stats['AST']} ast | {player_stats['FG']}/{player_stats['FGA']} FG
                    </p>
                </div>
        """
    
    html_content += """
            </div>
        </div>
        
        <!-- Game Winner Banner -->
        <div style="text-align: center; background-color: """ + winner_color + """; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h2 style="color: white; margin: 0; font-size: 28px;">
    """
    
    if winner == "Tie":
        html_content += "It's a Tie!"
    else:
        html_content += f"{winner} Wins!"
    
    html_content += """
            </h2>
        </div>
    </div>
    """
    
    return play_by_play_text, html_content

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