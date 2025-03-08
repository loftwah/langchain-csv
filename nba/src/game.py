"""
Game Simulator Module - Provides classes for simulating NBA basketball games.

This module implements an object-oriented approach to game simulation,
breaking down the large game_simulator function into manageable classes
with clear responsibilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

from .api import (
    get_player_id,
    get_player_games,
    get_player_historical_data
)
from .config import GAME_CONFIG, NBA_COLORS, NOTABLE_PLAYERS


class Player:
    """
    Represents a basketball player with their stats and game performance.
    """
    def __init__(self, name: str, stats: pd.Series = None):
        self.name = name
        self.stats = pd.Series() if stats is None else stats
        self.game_stats = self._init_game_stats()
        
    def _init_game_stats(self) -> Dict[str, int]:
        """Initialize empty game statistics dictionary."""
        return {
            'PTS': 0, 'FG': 0, 'FGA': 0, 'FG3': 0, 'FG3A': 0,
            'FT': 0, 'FTA': 0, 'REB': 0, 'AST': 0, 'STL': 0,
            'BLK': 0, 'TO': 0, 'PF': 0
        }
        
    @property
    def fg_pct(self) -> float:
        """Get field goal percentage from stats or use default."""
        return self.stats.get('FG_PCT', 0.45)
        
    @property
    def fg3_pct(self) -> float:
        """Get 3-point percentage from stats or use default."""
        return self.stats.get('FG3_PCT', 0.35)
        
    @property
    def ft_pct(self) -> float:
        """Get free throw percentage from stats or use default."""
        return self.stats.get('FT_PCT', 0.75)
        
    def update_stats(self, stat_type: str, value: int = 1) -> None:
        """Update a specific game statistic."""
        if stat_type in self.game_stats:
            self.game_stats[stat_type] += value
            

class Team:
    """
    Represents a basketball team with players and team statistics.
    """
    def __init__(self, name: str, players: List[Player] = None):
        self.name = name
        self.players = players or []
        self.score = 0
        self.quarter_scores = []
        
    def add_player(self, player: Player) -> None:
        """Add a player to the team."""
        self.players.append(player)
        
    def add_points(self, points: int) -> None:
        """Add points to the team's score."""
        self.score += points
        
    def end_quarter(self, quarter_score: int) -> None:
        """Record the score for a quarter."""
        self.quarter_scores.append(quarter_score)
        
    @property
    def total_stats(self) -> Dict[str, int]:
        """Calculate total team stats from all players."""
        totals = {
            'PTS': 0, 'REB': 0, 'AST': 0, 'STL': 0, 'BLK': 0, 'TO': 0
        }
        
        if not self.players:
            return totals
            
        for player in self.players:
            for stat in totals:
                totals[stat] += player.game_stats.get(stat, 0)
        return totals


class GameState:
    """
    Tracks the state of an ongoing basketball game.
    """
    def __init__(self, team1: Team, team2: Team):
        self.team1 = team1
        self.team2 = team2
        self.current_quarter = 0
        self.play_by_play: List[str] = []
        self.time_remaining = 0
        
    def add_play(self, play_text: str) -> None:
        """Add a play to the play-by-play list."""
        self.play_by_play.append(play_text)
        
    def format_score(self) -> str:
        """Format the current game score."""
        return f"{self.team1.name} {self.team1.score}-{self.team2.score} {self.team2.name}"
        
    @property
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.current_quarter >= 4  # Adjust for overtime later
    
    
class PlayGenerator:
    """
    Generates basketball plays during the simulation.
    """
    def __init__(self, game_state: GameState):
        self.game_state = game_state
        
    def generate_play(self, offensive_team: Team, defensive_team: Team) -> Tuple[str, int]:
        """
        Generate a play for the offensive team against the defensive team.
        
        Returns:
            Tuple[str, int]: A tuple containing the play text and points scored.
        """
        # Select primary offensive player (weighted by points scoring)
        primary_player = self._select_primary_player(offensive_team)
        
        # Determine play type
        play_type = self._determine_play_type()
        
        # Generate the actual play
        if play_type == 'turnover':
            return self._generate_turnover(primary_player, defensive_team)
        elif play_type == 'pass':
            return self._generate_pass_play(primary_player, offensive_team, defensive_team)
        else:  # Shot attempts
            return self._generate_shot_attempt(play_type, primary_player, offensive_team, defensive_team)
    
    def _select_primary_player(self, team: Team) -> Player:
        """Select a primary player based on scoring ability."""
        # Simple random selection for now - can be enhanced with weighted selection
        return np.random.choice(team.players)
    
    def _determine_play_type(self) -> str:
        """Determine the type of play to run."""
        # Use config values
        shot_types = list(GAME_CONFIG['shot_types'].keys())
        shot_probs = [GAME_CONFIG['shot_types'][t]['probability'] for t in shot_types]
        
        # Add non-shot play types
        play_types = shot_types + ['pass', 'turnover']
        play_probs = shot_probs + [
            GAME_CONFIG['play_types']['pass'],
            GAME_CONFIG['play_types']['turnover']
        ]
        
        # Normalize probabilities
        play_probs = [p/sum(play_probs) for p in play_probs]
        
        return np.random.choice(play_types, p=play_probs)
    
    def _generate_turnover(self, player: Player, defensive_team: Team) -> Tuple[str, int]:
        """Generate a turnover play."""
        defender = np.random.choice(defensive_team.players)
        
        # Add enhanced descriptions for notable players
        if player.name in NOTABLE_PLAYERS and defender.name in NOTABLE_PLAYERS:
            # Both players are notable
            turnover_types = [
                f"In a clash of titans, {defender.name} forces {player.name} into a rare turnover!",
                f"{defender.name} shows why he's one of the greatest defenders, stealing the ball from {player.name}!",
                f"The legendary {player.name} loses the handle, and {defender.name} is there to capitalize!",
                f"An epic defensive play by {defender.name}, stripping the ball from the great {player.name}!"
            ]
        elif player.name in NOTABLE_PLAYERS:
            # Only offensive player is notable
            turnover_types = [
                f"The normally flawless {player.name} commits an uncharacteristic turnover!",
                f"{player.name} loses the ball in a rare mistake!",
                f"Even the great {player.name} can make a mistake - turnover!",
                f"{defender.name} will remember this forever - stealing the ball from the legendary {player.name}!"
            ]
        elif defender.name in NOTABLE_PLAYERS:
            # Only defender is notable
            turnover_types = [
                f"The defensive wizard {defender.name} creates another turnover!",
                f"{defender.name} showcases his elite defensive skills, stripping {player.name}!",
                f"Another one for the highlight reel as {defender.name} forces the turnover!",
                f"{defender.name} reads the play perfectly and steals the ball from {player.name}!"
            ]
        else:
            # Standard turnover descriptions
            turnover_types = [
                f"{player.name} loses the ball out of bounds!",
                f"{player.name} steps on the sideline, turnover!",
                f"{player.name} throws an errant pass that goes out of bounds!",
                f"{defender.name} strips {player.name} for the steal!",
                f"{player.name} dribbles the ball off their foot!"
            ]
        
        play_text = np.random.choice(turnover_types)
        
        # Update stats
        player.update_stats('TO')
        if "strips" in play_text or "steal" in play_text:
            defender.update_stats('STL')
            
        return play_text, 0
    
    def _generate_pass_play(self, passer: Player, offensive_team: Team, defensive_team: Team) -> Tuple[str, int]:
        """Generate a play where the primary player passes to a teammate."""
        # Find potential receivers (excluding the passer)
        receivers = [p for p in offensive_team.players if p.name != passer.name]
        if not receivers:  # Fallback if somehow no receivers
            return self._generate_shot_attempt('mid_range', passer, offensive_team, defensive_team)
            
        receiver = np.random.choice(receivers)
        
        # Determine shot type after pass
        if np.random.random() < 0.3:  # 30% chance of three pointer
            shot_type = 'three_point'
            success_prob = receiver.fg3_pct - 0.05  # Slightly harder than player's avg
            points = 3
            success_text = f"{passer.name} finds {receiver.name} behind the arc... BANG! A beautiful three-pointer!"
            miss_text = f"{passer.name} kicks it out to {receiver.name} for three... No good!"
        else:
            shot_type = 'two_point'
            success_prob = receiver.fg_pct
            points = 2
            success_text = f"{passer.name} with a perfect pass to {receiver.name} who scores!"
            miss_text = f"{passer.name} passes to {receiver.name} who misses the shot!"
            
        # Determine success
        if np.random.random() < success_prob:
            play_text = success_text
            
            # Update stats
            receiver.update_stats('PTS', points)
            receiver.update_stats('FG')
            receiver.update_stats('FGA')
            if shot_type == 'three_point':
                receiver.update_stats('FG3')
                receiver.update_stats('FG3A')
            passer.update_stats('AST')
            
            return play_text, points
        else:
            play_text = miss_text
            
            # Update stats
            receiver.update_stats('FGA')
            if shot_type == 'three_point':
                receiver.update_stats('FG3A')
                
            # Handle rebound
            if np.random.random() < 0.25:  # 25% chance of offensive rebound
                rebounder = np.random.choice(offensive_team.players)
                play_text += f" {rebounder.name} grabs the offensive board!"
                rebounder.update_stats('REB')
            else:
                rebounder = np.random.choice(defensive_team.players)
                play_text += f" Rebound {defensive_team.name}, {rebounder.name}."
                rebounder.update_stats('REB')
                
            return play_text, 0
    
    def _generate_shot_attempt(self, play_type: str, shooter: Player, offensive_team: Team, defensive_team: Team) -> Tuple[str, int]:
        """Generate a shot attempt play."""
        defender = np.random.choice(defensive_team.players)
        
        # Base success probability for different shots
        success_probs = {
            'three_point': shooter.fg3_pct,
            'mid_range': shooter.fg_pct + 0.05,  # Easier than average FG
            'layup': shooter.fg_pct + 0.15,      # Quite high success rate
            'dunk': 0.85                         # Very high success rate
        }
        
        # Defender adjustment (simplified)
        def_impact = 0.05  # Can be enhanced with actual defensive stats
        
        # Shot is blocked?
        is_blocked = np.random.random() < def_impact
        if is_blocked:
            play_text = f"{defender.name} with a huge block on {shooter.name}!"
            
            # Update stats
            shooter.update_stats('FGA')
            if play_type == 'three_point':
                shooter.update_stats('FG3A')
            defender.update_stats('BLK')
            
            # Handle rebound after block
            if np.random.random() < 0.3:  # 30% chance offensive team gets it
                rebounder = np.random.choice(offensive_team.players)
                play_text += f" But {rebounder.name} recovers the ball for {offensive_team.name}!"
                rebounder.update_stats('REB')
            else:
                rebounder = np.random.choice(defensive_team.players)
                play_text += f" {defensive_team.name} gets possession with {rebounder.name} securing the ball."
                rebounder.update_stats('REB')
                
            return play_text, 0
            
        # Shot descriptions
        shot_descriptions = {
            'three_point': [
                f"{shooter.name} pulls up from downtown",
                f"{shooter.name} launches from beyond the arc",
                f"{shooter.name} steps back for a deep three"
            ],
            'mid_range': [
                f"{shooter.name} pulls up from mid-range",
                f"{shooter.name} with the fadeaway jumper",
                f"{shooter.name} shoots over {defender.name}"
            ],
            'layup': [
                f"{shooter.name} drives to the hoop",
                f"{shooter.name} with a crafty move to the basket",
                f"{shooter.name} goes in for the layup"
            ],
            'dunk': [
                f"{shooter.name} drives down the lane",
                f"{shooter.name} gets past {defender.name}",
                f"{shooter.name} elevates"
            ]
        }
        
        shot_text = np.random.choice(shot_descriptions[play_type])
        success_prob = success_probs[play_type]
        
        if np.random.random() < success_prob:
            # Shot successful
            if play_type == 'three_point':
                points = 3
                play_text = f"{shot_text} BANG! {points} points!"
                shooter.update_stats('FG3')
                shooter.update_stats('FG3A')
            elif play_type == 'dunk':
                points = 2
                play_text = f"{shot_text} AND THROWS IT DOWN! Monster jam by {shooter.name}!"
            else:
                points = 2
                play_text = f"{shot_text} and it's GOOD! {points} points!"
            
            # Update stats for all successful shots
            shooter.update_stats('PTS', points)
            shooter.update_stats('FG')
            shooter.update_stats('FGA')
            
            return play_text, points
        else:
            # Shot missed
            if play_type == 'three_point':
                play_text = f"{shot_text} off the mark!"
                shooter.update_stats('FG3A')
            elif play_type == 'dunk':
                play_text = f"{shot_text} but can't finish the slam!"
            else:
                play_text = f"{shot_text} but it rims out!"
                
            # Update stats for missed shots
            shooter.update_stats('FGA')
            
            # Handle rebound
            if np.random.random() < 0.25:  # 25% chance of offensive rebound
                rebounder = np.random.choice(offensive_team.players)
                play_text += f" {rebounder.name} gets the offensive rebound!"
                rebounder.update_stats('REB')
            else:
                rebounder = np.random.choice(defensive_team.players)
                play_text += f" {rebounder.name} grabs the defensive board."
                rebounder.update_stats('REB')
                
            return play_text, 0


class GameRenderer:
    """
    Renders game results in different formats (text, HTML).
    """
    def __init__(self, game_state: GameState):
        self.game_state = game_state
        
    def generate_play_by_play_text(self) -> str:
        """Generate a formatted text version of the play-by-play."""
        return "\n".join(self.game_state.play_by_play)
        
    def generate_html(self) -> str:
        """Generate an HTML visualization of the game."""
        team1 = self.game_state.team1
        team2 = self.game_state.team2
        
        # Determine winner and set colors
        if team1.score > team2.score:
            winner = team1.name
            winner_color = "#17408B"  # Blue color
        elif team2.score > team1.score:
            winner = team2.name
            winner_color = "#C9082A"  # Red color
        else:
            winner = "Tie"
            winner_color = "#F7B801"  # Gold color for tie
        
        # Build HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{team1.name} vs {team2.name} - Game Summary</title>
            <style>
                body {{
                    font-family: 'Roboto', Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .game-title {{
                    font-size: 28px;
                    font-weight: bold;
                    margin-bottom: 10px;
                }}
                .final-score {{
                    font-size: 42px;
                    font-weight: bold;
                    margin: 20px 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }}
                .team-name {{
                    margin: 0 15px;
                    text-align: center;
                }}
                .team1 {{ color: #17408B; }}
                .team2 {{ color: #C9082A; }}
                .score-separator {{
                    font-size: 24px;
                    color: #666;
                }}
                .winner-tag {{
                    display: inline-block;
                    padding: 5px 10px;
                    font-size: 14px;
                    font-weight: bold;
                    color: white;
                    background-color: {winner_color};
                    border-radius: 15px;
                    margin-top: 5px;
                }}
                .quarter-scores {{
                    margin: 20px 0;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    overflow: hidden;
                    width: 100%;
                }}
                .quarter-scores table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                .quarter-scores th, .quarter-scores td {{
                    padding: 10px;
                    text-align: center;
                    border: 1px solid #ddd;
                }}
                .quarter-scores th {{
                    background-color: #f3f3f3;
                    font-weight: bold;
                }}
                .team-stats {{
                    display: flex;
                    justify-content: space-around;
                    width: 100%;
                    margin: 30px 0;
                }}
                .stats-container {{
                    flex: 1;
                    max-width: 100%;
                    overflow-x: auto;
                }}
                .stats-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 10px;
                    border: 1px solid #ddd;
                    box-shadow: 0 2px 3px rgba(0, 0, 0, 0.1);
                }}
                .stats-table th, .stats-table td {{
                    padding: 10px;
                    text-align: center;
                    border: 1px solid #ddd;
                }}
                .stats-table th {{
                    background-color: #f3f3f3;
                    position: sticky;
                    top: 0;
                    font-weight: bold;
                }}
                .section-title {{
                    font-size: 20px;
                    font-weight: bold;
                    margin: 20px 0 10px;
                    padding-bottom: 5px;
                    border-bottom: 2px solid #ddd;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <div class="game-title">Game Simulation Summary</div>
                <div class="final-score">
                    <div class="team-name team1">
                        {team1.name}
                        {f'<div class="winner-tag">WINNER</div>' if team1.score > team2.score else ''}
                    </div>
                    <div class="score-separator">
                        {team1.score} - {team2.score}
                    </div>
                    <div class="team-name team2">
                        {team2.name}
                        {f'<div class="winner-tag">WINNER</div>' if team2.score > team1.score else ''}
                    </div>
                </div>
            </div>
            
            <div class="section-title">Quarter Scores</div>
            <div class="quarter-scores">
                <table>
                    <thead>
                        <tr>
                            <th>Team</th>
                            {' '.join([f'<th>Q{i+1}</th>' for i in range(len(team1.quarter_scores))])}
                            <th>Final</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>{team1.name}</td>
                            {' '.join([f'<td>{score}</td>' for score in team1.quarter_scores])}
                            <td>{team1.score}</td>
                        </tr>
                        <tr>
                            <td>{team2.name}</td>
                            {' '.join([f'<td>{score}</td>' for score in team2.quarter_scores])}
                            <td>{team2.score}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <div class="section-title">Team Stats Comparison</div>
            <div class="team-stats">
                <div class="stats-container" style="width: 100%;">
                    <table class="stats-table">
                        <thead>
                            <tr>
                                <th>Stat</th>
                                <th>{team1.name}</th>
                                <th>{team2.name}</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Points</td>
                                <td>{team1.total_stats['PTS']}</td>
                                <td>{team2.total_stats['PTS']}</td>
                            </tr>
                            <tr>
                                <td>Rebounds</td>
                                <td>{team1.total_stats['REB']}</td>
                                <td>{team2.total_stats['REB']}</td>
                            </tr>
                            <tr>
                                <td>Assists</td>
                                <td>{team1.total_stats['AST']}</td>
                                <td>{team2.total_stats['AST']}</td>
                            </tr>
                            <tr>
                                <td>Steals</td>
                                <td>{team1.total_stats['STL']}</td>
                                <td>{team2.total_stats['STL']}</td>
                            </tr>
                            <tr>
                                <td>Blocks</td>
                                <td>{team1.total_stats['BLK']}</td>
                                <td>{team2.total_stats['BLK']}</td>
                            </tr>
                            <tr>
                                <td>Turnovers</td>
                                <td>{team1.total_stats['TO']}</td>
                                <td>{team2.total_stats['TO']}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="section-title">{team1.name} Player Stats</div>
            <div class="team-stats">
                <div class="stats-container" style="width: 100%;">
                    <table class="stats-table">
                        <thead>
                            <tr>
                                <th>Player</th>
                                <th>PTS</th>
                                <th>REB</th>
                                <th>AST</th>
                                <th>STL</th>
                                <th>BLK</th>
                                <th>TO</th>
                            </tr>
                        </thead>
                        <tbody>
                            {"".join([f'''
                            <tr>
                                <td>{player.name}</td>
                                <td>{player.game_stats['PTS']}</td>
                                <td>{player.game_stats['REB']}</td>
                                <td>{player.game_stats['AST']}</td>
                                <td>{player.game_stats['STL']}</td>
                                <td>{player.game_stats['BLK']}</td>
                                <td>{player.game_stats['TO']}</td>
                            </tr>
                            ''' for player in team1.players])}
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="section-title">{team2.name} Player Stats</div>
            <div class="team-stats">
                <div class="stats-container" style="width: 100%;">
                    <table class="stats-table">
                        <thead>
                            <tr>
                                <th>Player</th>
                                <th>PTS</th>
                                <th>REB</th>
                                <th>AST</th>
                                <th>STL</th>
                                <th>BLK</th>
                                <th>TO</th>
                            </tr>
                        </thead>
                        <tbody>
                            {"".join([f'''
                            <tr>
                                <td>{player.name}</td>
                                <td>{player.game_stats['PTS']}</td>
                                <td>{player.game_stats['REB']}</td>
                                <td>{player.game_stats['AST']}</td>
                                <td>{player.game_stats['STL']}</td>
                                <td>{player.game_stats['BLK']}</td>
                                <td>{player.game_stats['TO']}</td>
                            </tr>
                            ''' for player in team2.players])}
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="section-title">Game Synopsis</div>
            <p>
                {f"The {winner} won with a final score of {team1.score}-{team2.score}." if winner != "Tie" else f"The game ended in a tie with a score of {team1.score}-{team2.score}."}
                {team1.name} {f"led the way with {max([p.game_stats['PTS'] for p in team1.players])} points from {max(team1.players, key=lambda p: p.game_stats['PTS']).name}" if team1.players else "had a balanced attack"}.
                {team2.name} {f"was paced by {max([p.game_stats['PTS'] for p in team2.players])} points from {max(team2.players, key=lambda p: p.game_stats['PTS']).name}" if team2.players else "had a balanced attack"}.
            </p>
            
            <footer style="text-align: center; margin-top: 30px; font-size: 12px; color: #777;">
                This is a simulated game generated based on player statistics.
            </footer>
        </body>
        </html>
        """
        
        return html_content


class GameSimulator:
    """
    Main simulator class that coordinates the game simulation.
    """
    def __init__(self, 
                 team1_players: str, 
                 team2_players: str, 
                 team1_name: str = "Team 1", 
                 team2_name: str = "Team 2", 
                 quarters: int = GAME_CONFIG['default_quarters'], 
                 quarter_length: int = GAME_CONFIG['default_quarter_length']):
        """
        Initialize the game simulator.
        
        Args:
            team1_players: Comma-separated list of players on team 1
            team2_players: Comma-separated list of players on team 2
            team1_name: Name for team 1
            team2_name: Name for team 2
            quarters: Number of quarters to play
            quarter_length: Length of each quarter in minutes
        """
        self.team1_name = team1_name
        self.team2_name = team2_name
        self.quarters = quarters
        self.quarter_length = quarter_length
        
        # Create teams and load player stats
        self.team1 = self._load_team(team1_players, team1_name)
        self.team2 = self._load_team(team2_players, team2_name)
        
        # Initialize game state
        self.game_state = GameState(self.team1, self.team2)
        
        # Initialize play generator
        self.play_generator = PlayGenerator(self.game_state)
        
        # Initialize renderer
        self.renderer = GameRenderer(self.game_state)
        
    def _load_team(self, players_str: str, team_name: str) -> Team:
        """
        Load a team with player stats from player names.
        
        Args:
            players_str: Comma-separated list of player names
            team_name: Name of the team
            
        Returns:
            Team: A team object with initialized players
        """
        team = Team(team_name)
        
        # Process players
        for player_name in players_str.split(','):
            player_name = player_name.strip()
            if not player_name:
                continue
                
            # Get player ID
            player_id = get_player_id(player_name)
            if not player_id:
                continue
                
            # Get player stats
            player_stats = self._get_player_stats(player_id, player_name)
            if player_stats is not None:
                player = Player(player_name, player_stats)
                team.add_player(player)
                
        return team
        
    def _get_player_stats(self, player_id: str, player_name: str) -> Optional[pd.Series]:
        """
        Get a player's stats from recent games or historical data.
        
        Args:
            player_id: The player's ID
            player_name: The player's name
            
        Returns:
            Optional[pd.Series]: Player's stats or None if not available
        """
        # Try to get recent games
        recent_games = get_player_games(player_id)
        
        if not recent_games.empty:
            # Calculate average stats
            avg_stats = recent_games.mean(numeric_only=True)
            avg_stats['PLAYER_NAME'] = player_name
            return avg_stats
        else:
            # Try to get historical data for retired players
            historical_data = get_player_historical_data(player_id)
            if not historical_data.empty:
                avg_stats = historical_data.mean(numeric_only=True)
                avg_stats['PLAYER_NAME'] = player_name
                return avg_stats
                
        return None
        
    def simulate_game(self) -> Tuple[str, str]:
        """
        Simulate the full basketball game.
        
        Returns:
            Tuple[str, str]: A tuple containing the play-by-play text and HTML visualization
        """
        # Add game intro
        self.game_state.add_play(f"# 🏀 {self.team1_name} vs {self.team2_name} - GAME SIMULATION")
        self.game_state.add_play(f"\n## Starting Lineups")
        self.game_state.add_play(f"\n### {self.team1_name}:")
        self.game_state.add_play(", ".join([p.name for p in self.team1.players]))
        self.game_state.add_play(f"\n### {self.team2_name}:")
        self.game_state.add_play(", ".join([p.name for p in self.team2.players]))
        self.game_state.add_play("\n---\n")
        
        # Simulate each quarter
        for quarter in range(1, self.quarters + 1):
            self.simulate_quarter(quarter)
            
        # Generate final results
        play_by_play_text = self.renderer.generate_play_by_play_text()
        html_content = self.renderer.generate_html()
        
        return play_by_play_text, html_content
        
    def simulate_quarter(self, quarter: int) -> None:
        """
        Simulate a single quarter of the game.
        
        Args:
            quarter: The quarter number (1-4)
        """
        self.game_state.current_quarter = quarter
        self.game_state.add_play(f"\n## Quarter {quarter}")
        
        # Initialize quarter scores
        quarter_team1_score = 0
        quarter_team2_score = 0
        
        # Calculate possessions for the quarter using config
        possessions = int(self.quarter_length * GAME_CONFIG['possessions_per_minute'])
        
        for _ in range(possessions):
            # Alternate possessions with slight randomness
            if np.random.random() < 0.5:
                # Team 1 has possession
                offensive_team = self.team1
                defensive_team = self.team2
            else:
                # Team 2 has possession
                offensive_team = self.team2
                defensive_team = self.team1
                
            # Generate play
            play_text, points_scored = self.play_generator.generate_play(offensive_team, defensive_team)
            
            # Update team scores
            if points_scored > 0:
                offensive_team.add_points(points_scored)
                if offensive_team == self.team1:
                    quarter_team1_score += points_scored
                else:
                    quarter_team2_score += points_scored
            
            # Format the possession with score
            possession_text = f"{offensive_team.name} {self.team1.score}-{self.team2.score} {defensive_team.name}: {play_text}"
            self.game_state.add_play(possession_text)
        
        # End of quarter summary
        self.game_state.add_play(f"\n**End of Quarter {quarter}**: {self.team1_name} {self.team1.score}-{self.team2.score} {self.team2_name}")
        
        # Update quarter scores
        self.team1.end_quarter(quarter_team1_score)
        self.team2.end_quarter(quarter_team2_score)


# Wrapper function for backwards compatibility with the original API
def game_simulator(team1_players, team2_players, team1_name="Team 1", team2_name="Team 2", quarters=4, quarter_length=12):
    """
    Wrapper function that maintains compatibility with the original API.
    
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
    simulator = GameSimulator(
        team1_players=team1_players,
        team2_players=team2_players,
        team1_name=team1_name,
        team2_name=team2_name,
        quarters=quarters,
        quarter_length=quarter_length
    )
    
    return simulator.simulate_game() 