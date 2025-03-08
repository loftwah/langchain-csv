"""
Consistency Tracker Module - Provides classes for analyzing player consistency.

This module implements an object-oriented approach to the consistency tracker,
breaking down the original function into manageable classes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional

from .api import get_player_id, get_player_games, get_players
from .fantasy import calculate_fantasy_points, calculate_consistency
from .config import NBA_COLORS, CONSISTENCY_CONFIG


class ConsistencyTracker:
    """
    Analyzes a player's consistency in fantasy performance.
    """
    def __init__(self, player_name: str, num_games: int = 10, scoring_system: str = 'standard'):
        """
        Initialize the consistency tracker.
        
        Args:
            player_name: The name of the player to analyze
            num_games: Number of recent games to analyze
            scoring_system: Fantasy scoring system to use
        """
        self.player_name = player_name
        self.num_games = num_games
        self.scoring_system = scoring_system
        self.player_id = None
        self.recent_games = None
        self.consistency_metrics = None
        
    def run_analysis(self) -> Tuple[str, Optional[plt.Figure]]:
        """
        Run the full consistency analysis for the player.
        
        Returns:
            Tuple containing: (summary text, visualization figure)
        """
        print(f"Starting analysis for player: '{self.player_name}' using {self.num_games} games")
        
        # Get player ID
        self.player_id = get_player_id(self.player_name)
        
        if not self.player_id:
            return self._handle_missing_player_id(), None
            
        print(f"Found player ID: {self.player_id} for '{self.player_name}'")
            
        # Get recent games
        try:
            self.recent_games = get_player_games(self.player_id, last_n_games=self.num_games)
        except Exception as e:
            error_msg = f"Error fetching game data: {str(e)}"
            print(error_msg)
            return error_msg, None
        
        if self.recent_games is None or self.recent_games.empty:
            return f"No recent game data found for {self.player_name}", None
        
        print(f"Retrieved {len(self.recent_games)} games for analysis")
            
        # Calculate consistency metrics
        try:
            self.consistency_metrics = calculate_consistency(self.recent_games, self.scoring_system)
        except Exception as e:
            error_msg = f"Error calculating consistency metrics: {str(e)}"
            print(error_msg)
            return error_msg, None
        
        # Create visualization
        try:
            fig = self._create_visualization()
        except Exception as e:
            error_msg = f"Error creating visualization: {str(e)}"
            print(error_msg)
            return f"Analysis completed, but visualization failed: {error_msg}", None
        
        # Generate summary
        summary = self._generate_summary()
        
        return summary, fig
        
    def _handle_missing_player_id(self) -> str:
        """
        Handle the case where a player ID is not found.
        
        Returns:
            Error message with debugging info
        """
        print(f"DEBUG: Could not find player ID for '{self.player_name}'. Checking available player names...")
        # Get first 5 players from dataset as examples
        all_players = get_players()
        name_column = 'DISPLAY_FIRST_LAST' if 'DISPLAY_FIRST_LAST' in all_players.columns else 'full_name'
        if name_column in all_players.columns:
            # Try to find similar names to help with debugging
            sample_players = all_players[name_column].head(5).tolist()
            print(f"DEBUG: Sample of available players: {sample_players}")
            
            # Try to find close matches if it's a known player with special characters
            known_players = {
                "jokic": "Nikola Jokić",
                "doncic": "Luka Dončić",
                "jokić": "Nikola Jokić",
                "dončić": "Luka Dončić"
            }
            
            normalized_input = self.player_name.lower()
            for key, suggestion in known_players.items():
                if key in normalized_input:
                    return f"No player found matching '{self.player_name}'. Did you mean '{suggestion}'? Please try with the suggested name."
        
        return f"No player found matching '{self.player_name}'. Please check the spelling and try again."
        
    def _create_visualization(self) -> plt.Figure:
        """
        Create visualization of player consistency.
        
        Returns:
            matplotlib Figure with the visualization
        """
        # Unpack consistency metrics
        mean_fp, cv, min_fp, max_fp, fp_trend = self.consistency_metrics
        
        # Create the figure
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
        ax1.set_title(f'{self.player_name} - Fantasy Performance Analysis', 
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
        consistency_score = 100 * (1 - min(cv, 1))
        
        # Use config for consistency categories
        consistency_categories = CONSISTENCY_CONFIG['consistency_thresholds']
        
        # Find player's consistency category
        consistency_label = 'Unknown'
        consistency_color = 'gray'
        for low, high, label, color in consistency_categories:
            if low <= consistency_score < high:
                consistency_label = label
                consistency_color = color
                break
        
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
        
        # Add branding to the plot
        fig.text(0.95, 0.02, "Created by Loftwah", fontsize=10, 
                 ha='right', va='bottom', color=NBA_COLORS['highlight'],
                 url="https://linkarooie.com/loftwah")
        
        plt.tight_layout()
        
        return fig
        
    def _generate_summary(self) -> str:
        """
        Generate a text summary of the consistency analysis.
        
        Returns:
            Formatted text summary
        """
        # Unpack consistency metrics
        mean_fp, cv, min_fp, max_fp, fp_trend = self.consistency_metrics
        consistency_score = 100 * (1 - min(cv, 1))
        
        # Use config for determining consistency label
        consistency_label = 'Unknown'
        for low, high, label, _ in CONSISTENCY_CONFIG['consistency_thresholds']:
            if low <= consistency_score < high:
                consistency_label = label
                break
                
        # Format game log for display if we have data
        if not self.recent_games.empty and 'GAME_DATE' in self.recent_games.columns:
            # Calculate fantasy points
            fantasy_games = calculate_fantasy_points(self.recent_games, self.scoring_system)
            
            # Format date
            if 'GAME_DATE' in fantasy_games.columns:
                fantasy_games['GAME_DATE'] = pd.to_datetime(fantasy_games['GAME_DATE']).dt.strftime('%Y-%m-%d')
            
        # Check if we have enough data points for trend analysis
        min_data_points = CONSISTENCY_CONFIG['min_data_points']
        if len(fp_trend) >= min_data_points:
            recent_trend = sum(fp_trend[:3]) > sum(fp_trend[-3:])
            stable_trend = abs(sum(fp_trend[:3]) - sum(fp_trend[-3:])) < mean_fp * 0.1
            
            if recent_trend:
                trend_text = "🔥 Their production has been trending upward recently."
            elif stable_trend:
                trend_text = "➡️ Their production has been fairly stable recently."
            else:
                trend_text = "📉 Their production has been trending downward recently."
        else:
            trend_text = "➡️ Not enough games to determine a clear trend."
            
        # Consistency interpretation
        if consistency_score >= 60:
            consistency_text = "💯 Their high consistency makes them a reliable starter each week."
        elif consistency_score >= 40:
            consistency_text = "👍 Their moderate consistency means they're generally reliable but can have off games."
        else:
            consistency_text = "🎲 Their inconsistency makes them a boom-or-bust player who can win or lose your matchup."
            
        # Create summary text with emojis and rich formatting
        summary = f"""
        ## 📊 {self.player_name} Consistency Analysis
        
        ### 🔍 Overview:
        - **Average Fantasy Points**: {mean_fp:.1f} pts
        - **Consistency Score**: {consistency_score:.1f}/100 ({consistency_label})
        - **Range**: {min_fp:.1f} to {max_fp:.1f} fantasy points
        - **Games Analyzed**: {len(fp_trend)}
        
        ### ⚖️ Consistency Interpretation:
        This player is **{consistency_label.lower()}** in their fantasy production. 
        {consistency_text}
        
        ### 📈 Recent Performance Trend:
        {trend_text}
        
        *Analysis provided by [Loftwah's Fantasy Basketball Assistant](https://linkarooie.com/loftwah)*
        """
        
        return summary


# Wrapper function for backwards compatibility with the original API
def consistency_tracker(player_name, num_games=10, scoring_system='standard'):
    """
    Wrapper function that maintains compatibility with the original API.
    
    Args:
        player_name (str): Name of the player to analyze
        num_games (int): Number of recent games to analyze
        scoring_system (str): Fantasy scoring system to use
        
    Returns:
        tuple: (summary_text, visualization_figure)
    """
    tracker = ConsistencyTracker(
        player_name=player_name,
        num_games=num_games,
        scoring_system=scoring_system
    )
    
    return tracker.run_analysis() 