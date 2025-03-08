import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .config import NBA_COLORS, DEFAULT_SEASON
from .api import (
    get_league_leaders, 
    get_player_stats, 
    get_player_id, 
    get_player_games
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
        return f"No player found matching '{player_name}'", None
    
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