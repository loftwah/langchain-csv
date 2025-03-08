import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
from datetime import datetime, timedelta
import os
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

# Create a cache directory
CACHE_DIR = "nba_api_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# -------------------- API Wrapper Functions --------------------

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

def get_league_leaders(season="2023-24", stat_category="PTS", per_mode="PerGame", limit=50):
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

def get_player_stats(season="2023-24", min_games=20):
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

def get_player_games(player_id, season="2023-24", last_n_games=10):
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

# -------------------- Fantasy Basketball Functions --------------------

def calculate_fantasy_points(stats_df, scoring_system='standard'):
    """Calculate fantasy points based on different scoring systems"""
    # Create a copy to avoid modifying the original
    df = stats_df.copy()
    
    # Ensure required columns exist
    for col in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M']:
        if col not in df.columns:
            df[col] = 0
    
    if scoring_system == 'standard':
        # Standard scoring: PTS=1, REB=1.2, AST=1.5, STL=3, BLK=3, TOV=-1, 3PM=0.5
        df['FANTASY_POINTS'] = (
            df['PTS'] * 1.0 +
            df['REB'] * 1.2 +
            df['AST'] * 1.5 +
            df['STL'] * 3.0 +
            df['BLK'] * 3.0 +
            df['TOV'] * -1.0 +
            df['FG3M'] * 0.5
        )
    elif scoring_system == 'points':
        # Points league: PTS=1, REB=1, AST=1, STL=1, BLK=1, TOV=-1
        df['FANTASY_POINTS'] = (
            df['PTS'] * 1.0 +
            df['REB'] * 1.0 +
            df['AST'] * 1.0 +
            df['STL'] * 1.0 +
            df['BLK'] * 1.0 +
            df['TOV'] * -1.0
        )
    elif scoring_system == 'categories':
        # For category leagues, we calculate z-scores for each category
        categories = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG_PCT', 'FT_PCT', 'FG3M']
        negative_cats = ['TOV']
        
        # Calculate z-score for each category
        for cat in categories:
            if cat in df.columns:
                mean = df[cat].mean()
                std = df[cat].std()
                if std > 0:  # Avoid division by zero
                    df[f'{cat}_ZSCORE'] = (df[cat] - mean) / std
                else:
                    df[f'{cat}_ZSCORE'] = 0
        
        # For negative categories, multiply by -1 so that lower is better
        for cat in negative_cats:
            if cat in df.columns:
                mean = df[cat].mean()
                std = df[cat].std()
                if std > 0:
                    df[f'{cat}_ZSCORE'] = -1 * (df[cat] - mean) / std
                else:
                    df[f'{cat}_ZSCORE'] = 0
        
        # Sum all z-scores to get total value
        z_cols = [f'{cat}_ZSCORE' for cat in categories + negative_cats if f'{cat}_ZSCORE' in df.columns]
        df['FANTASY_POINTS'] = df[z_cols].sum(axis=1)
    
    return df

def calculate_consistency(games_df, scoring_system='standard'):
    """Calculate player consistency based on fantasy points variation"""
    if games_df.empty:
        return 0, 0, 0, 0, []
    
    # Calculate fantasy points for each game
    fantasy_games = calculate_fantasy_points(games_df, scoring_system)
    
    # Calculate stats
    mean_fp = fantasy_games['FANTASY_POINTS'].mean()
    std_fp = fantasy_games['FANTASY_POINTS'].std()
    min_fp = fantasy_games['FANTASY_POINTS'].min()
    max_fp = fantasy_games['FANTASY_POINTS'].max()
    
    # Calculate coefficient of variation (lower means more consistent)
    cv = std_fp / mean_fp if mean_fp > 0 else float('inf')
    
    # Game-by-game fantasy points
    fp_trend = fantasy_games['FANTASY_POINTS'].tolist()
    
    return mean_fp, cv, min_fp, max_fp, fp_trend

# -------------------- Fantasy Basketball Tool Implementations --------------------

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
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get top 20 players
    top_df = sorted_df.head(20)
    
    # Determine player name column
    player_col = 'PLAYER_NAME' if 'PLAYER_NAME' in top_df.columns else 'PLAYER'
    
    # Create bar chart
    bars = ax.barh(top_df[player_col], top_df['FANTASY_POINTS'], color='skyblue')
    
    # Add labels
    ax.set_xlabel('Fantasy Points per Game')
    ax.set_title(f'Top 20 Players by Fantasy Value ({scoring_system} scoring)', fontsize=16)
    ax.invert_yaxis()  # Highest value at the top
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f"{top_df['VALUE'].iloc[i]:.2f}", 
                ha='left', va='center')
    
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
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    axs = axs.flatten()
    
    all_cats = categories + percentages
    colors = ['#1f77b4', '#ff7f0e']  # Blue for team 1, orange for team 2
    
    for i, cat in enumerate(all_cats):
        if i >= len(axs):  # Ensure we don't exceed available subplots
            break
            
        # For turnovers, lower is better
        if cat == 'TOV':
            heights = [team2_totals[cat], team1_totals[cat]]
            labels = ['Team 2', 'Team 1']
            color_order = [colors[1], colors[0]]
        else:
            heights = [team1_totals[cat], team2_totals[cat]]
            labels = ['Team 1', 'Team 2']
            color_order = colors
        
        bars = axs[i].bar(labels, heights, color=color_order)
        axs[i].set_title(cat)
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            axs[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom')
    
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
        
        winner_text = f"Projected Result: "
        if team1_wins > team2_wins:
            winner_text += f"Team 1 wins ({team1_wins}-{team2_wins})"
        elif team2_wins > team1_wins:
            winner_text += f"Team 2 wins ({team2_wins}-{team1_wins})"
        else:
            winner_text += f"Tie ({team1_wins}-{team2_wins})"
            
        last_ax.text(0.5, 0.5, winner_text, fontsize=14, ha='center', va='center')
    
    # Hide any unused subplots
    for i in range(len(all_cats) + 1, len(axs)):
        axs[i].axis('off')
    
    plt.tight_layout()
    
    # Create summary text
    summary = f"""
    ## Team Comparison: Team 1 vs Team 2
    
    ### Projected Category Wins:
    - Team 1: {team1_wins}
    - Team 2: {team2_wins}
    
    ### Team 1 Players:
    {", ".join(team1_df['PLAYER_NAME'].tolist())}
    
    ### Team 2 Players:
    {", ".join(team2_df['PLAYER_NAME'].tolist())}
    
    ### Category Breakdown:
    """
    
    for cat in all_cats:
        val1 = team1_totals[cat]
        val2 = team2_totals[cat]
        
        # For TOV, lower is better
        if cat == 'TOV':
            winner = "Team 1" if val1 < val2 else "Team 2" if val2 < val1 else "Tie"
        else:
            winner = "Team 1" if val1 > val2 else "Team 2" if val2 > val1 else "Tie"
            
        summary += f"- **{cat}**: Team 1 ({val1:.1f}) vs Team 2 ({val2:.1f}) - Winner: {winner}\n"
    
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Game-by-game fantasy points trend
    game_indices = list(range(len(fp_trend)))
    ax1.plot(game_indices, fp_trend, marker='o', linestyle='-', color='blue', linewidth=2)
    ax1.set_xlabel('Game Number (Most Recent First)')
    ax1.set_ylabel('Fantasy Points')
    ax1.set_title(f'{player_name} - Last {len(fp_trend)} Games Fantasy Performance', fontsize=16)
    
    # Add mean line
    ax1.axhline(y=mean_fp, color='r', linestyle='--', label=f'Average: {mean_fp:.1f}')
    
    # Add min and max lines
    ax1.axhline(y=min_fp, color='g', linestyle=':', label=f'Min: {min_fp:.1f}')
    ax1.axhline(y=max_fp, color='purple', linestyle=':', label=f'Max: {max_fp:.1f}')
    
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add consistency visualization
    consistency_score = 100 * (1 - min(cv, 1))  # Convert CV to a 0-100 scale (higher is more consistent)
    
    # Custom gauge chart for consistency
    consistency_categories = [
        (0, 20, 'Very Inconsistent', 'red'),
        (20, 40, 'Inconsistent', 'orange'),
        (40, 60, 'Moderate', 'yellow'),
        (60, 80, 'Consistent', 'lightgreen'),
        (80, 100, 'Very Consistent', 'green')
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
             color=gauge_colors, alpha=0.6, height=0.5)
    
    # Add pointer for this player's consistency
    ax2.scatter(consistency_score, 1, color='black', zorder=10, s=300, marker='^')
    
    # Add consistency score and label
    ax2.text(50, 1.5, f"Consistency Score: {consistency_score:.1f}/100 - {consistency_label}", 
             ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Set up ax2 styling
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
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
        
        # Create summary text
        summary = f"""
        ## {player_name} Consistency Analysis
        
        ### Overview:
        - **Average Fantasy Points**: {mean_fp:.1f}
        - **Consistency Score**: {consistency_score:.1f}/100 ({consistency_label})
        - **Range**: {min_fp:.1f} to {max_fp:.1f} fantasy points
        - **Games Analyzed**: {len(fp_trend)}
        
        ### Consistency Interpretation:
        This player is **{consistency_label.lower()}** in their fantasy production. 
        {"Their high consistency makes them a reliable starter each week." if consistency_score >= 60 else
         "Their moderate consistency means they're generally reliable but can have off games." if consistency_score >= 40 else
         "Their inconsistency makes them a boom-or-bust player who can win or lose your matchup."}
        
        ### Recent Performance Trend:
        {"Their production has been trending upward recently." if sum(fp_trend[:3]) > sum(fp_trend[-3:]) else
         "Their production has been fairly stable recently." if abs(sum(fp_trend[:3]) - sum(fp_trend[-3:])) < mean_fp * 0.1 else
         "Their production has been trending downward recently."}
        """
        
        return summary, fig
    
    else:
        return f"Insufficient game data for {player_name}", fig

# -------------------- Gradio Interface --------------------

def create_interface():
    with gr.Blocks(title="Fantasy Basketball Tools") as demo:
        gr.Markdown("# Fantasy Basketball Assistant")
        gr.Markdown("A suite of tools to help you dominate your fantasy basketball league.")
        
        with gr.Tabs():
            # Draft Helper Tab
            with gr.Tab("Draft Helper"):
                gr.Markdown("## Fantasy Draft Helper")
                gr.Markdown("Find undervalued players for your fantasy draft based on statistical analysis.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        draft_scoring = gr.Radio(
                            choices=["standard", "points", "categories"],
                            label="Scoring System",
                            value="standard"
                        )
                        min_games = gr.Slider(
                            minimum=5, maximum=50, value=20, step=5,
                            label="Minimum Games Played"
                        )
                        stat_category = gr.Dropdown(
                            choices=["PTS", "AST", "REB", "STL", "BLK", "FG_PCT", "FT_PCT", "FG3M"],
                            label="Stat Category to Emphasize",
                            value="PTS"
                        )
                        draft_btn = gr.Button("Find Value Players")
                    
                    with gr.Column(scale=2):
                        draft_plot = gr.Plot(label="Top Players by Value")
                
                draft_table = gr.Dataframe(label="Player Rankings")
                
                draft_btn.click(
                    fn=draft_helper,
                    inputs=[draft_scoring, min_games, stat_category],
                    outputs=[draft_table, draft_plot]
                )
            
            # Matchup Analyzer Tab
            with gr.Tab("Matchup Analyzer"):
                gr.Markdown("## Weekly Matchup Analyzer")
                gr.Markdown("Compare two fantasy teams to predict matchup outcomes.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        team1 = gr.Textbox(
                            label="Team 1 Players (comma-separated)",
                            placeholder="LeBron James, Anthony Davis, Kevin Durant"
                        )
                        team2 = gr.Textbox(
                            label="Team 2 Players (comma-separated)",
                            placeholder="Nikola Jokic, Stephen Curry, Luka Doncic"
                        )
                        matchup_scoring = gr.Radio(
                            choices=["standard", "points", "categories"],
                            label="Scoring System",
                            value="standard"
                        )
                        matchup_btn = gr.Button("Compare Teams")
                
                matchup_result = gr.Markdown(label="Matchup Analysis")
                matchup_plot = gr.Plot(label="Category Comparison")
                
                matchup_btn.click(
                    fn=matchup_analyzer,
                    inputs=[team1, team2, matchup_scoring],
                    outputs=[matchup_result, matchup_plot]
                )
            
            # Consistency Tracker Tab
            with gr.Tab("Consistency Tracker"):
                gr.Markdown("## Player Consistency Tracker")
                gr.Markdown("Analyze a player's consistency to identify reliable starters vs. boom/bust players.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        player_name = gr.Textbox(
                            label="Player Name",
                            placeholder="LeBron James"
                        )
                        num_games = gr.Slider(
                            minimum=5, maximum=20, value=10, step=1,
                            label="Number of Recent Games to Analyze"
                        )
                        consistency_scoring = gr.Radio(
                            choices=["standard", "points", "categories"],
                            label="Scoring System",
                            value="standard"
                        )
                        consistency_btn = gr.Button("Analyze Player")
                
                consistency_result = gr.Markdown(label="Consistency Analysis")
                consistency_plot = gr.Plot(label="Performance Consistency")
                
                consistency_btn.click(
                    fn=consistency_tracker,
                    inputs=[player_name, num_games, consistency_scoring],
                    outputs=[consistency_result, consistency_plot]
                )
        
        gr.Markdown("""
        ## About this Tool
        
        This Fantasy Basketball Assistant uses data from the official NBA API to provide analytics and insights for fantasy basketball managers.
        
        **Features:**
        - **Draft Helper**: Identifies value players based on fantasy production relative to their expected draft position
        - **Matchup Analyzer**: Projects head-to-head category winners based on recent performance
        - **Consistency Tracker**: Evaluates player reliability to help you balance consistent producers with high-upside options
        
        Data is updated daily during the NBA season.
        """)
    
    return demo

# Launch the application
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