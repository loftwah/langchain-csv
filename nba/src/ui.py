import gradio as gr
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from .config import NBA_COLORS, CUSTOM_CSS
from .tools import draft_helper, matchup_analyzer, consistency_tracker, game_simulator
from .api import get_player_id, get_players, get_player_games

# Import the new AI UI components
from .ui_ai import create_ai_features_interface

def create_draft_helper_ui():
    """Create the UI for the Draft Helper tool"""
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Draft Settings")
            num_teams = gr.Slider(minimum=6, maximum=20, value=12, step=1, label="Number of Teams")
            scoring_type = gr.Dropdown(
                choices=["Standard Points", "Standard Categories", "Custom"], 
                value="Standard Points", 
                label="Scoring Type"
            )
            draft_pos = gr.Slider(minimum=1, maximum=20, value=6, step=1, label="Your Draft Position")
            
            analyze_btn = gr.Button("Generate Draft Analysis", variant="primary")
            
        with gr.Column(scale=2):
            output_area = gr.Dataframe(label="Draft Rankings and Analysis")
            
    analyze_btn.click(
        fn=lambda teams, scoring, pos: draft_helper(int(teams), scoring, int(pos)),
        inputs=[num_teams, scoring_type, draft_pos],
        outputs=output_area
    )
    
    return output_area

def create_matchup_analyzer_ui():
    """Create the UI for the Matchup Analyzer tool"""
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Team 1")
            team1_players = gr.Textbox(
                placeholder="Enter players, separated by commas",
                label="Team 1 Players"
            )
        with gr.Column():
            gr.Markdown("### Team 2")
            team2_players = gr.Textbox(
                placeholder="Enter players, separated by commas",
                label="Team 2 Players"
            )
    
    analyze_matchup_btn = gr.Button("Analyze Matchup", variant="primary")
    matchup_output = gr.Plot(label="Matchup Analysis")
    matchup_text = gr.Markdown(label="Matchup Results")
    
    analyze_matchup_btn.click(
        fn=matchup_analyzer,
        inputs=[team1_players, team2_players],
        outputs=[matchup_text, matchup_output]
    )
    
    return matchup_output

def create_consistency_tracker_ui():
    """Create the UI for the Consistency Tracker tool"""
    with gr.Row():
        with gr.Column():
            player_name = gr.Textbox(
                placeholder="Enter player name",
                label="Player Name"
            )
            time_period = gr.Dropdown(
                choices=["Last 10 Games", "Last 30 Days", "Season", "Custom Range"],
                value="Last 10 Games",
                label="Time Period"
            )
            analyze_consistency_btn = gr.Button("Analyze Consistency", variant="primary")
        
        with gr.Column():
            consistency_output = gr.Plot(label="Consistency Analysis")
            consistency_text = gr.Markdown(label="Consistency Insights")
    
    analyze_consistency_btn.click(
        fn=consistency_tracker,
        inputs=[player_name, time_period],
        outputs=[consistency_text, consistency_output]
    )
    
    return consistency_output

def load_preset_team(preset_name):
    """Load a preset team configuration for demo purposes"""
    presets = {
        "all_stars": "LeBron James, Kevin Durant, Stephen Curry, Giannis Antetokounmpo, Nikola Jokic",
        "young_guns": "Luka Doncic, Trae Young, Ja Morant, Zion Williamson, Anthony Edwards",
        "big_men": "Joel Embiid, Nikola Jokic, Anthony Davis, Karl-Anthony Towns, Bam Adebayo",
        "guards": "Stephen Curry, Damian Lillard, Kyrie Irving, Devin Booker, Donovan Mitchell",
        "legends": "Michael Jordan, Kobe Bryant, LeBron James, Magic Johnson, Larry Bird",
        "offense": "James Harden, Stephen Curry, Kevin Durant, Giannis Antetokounmpo, Joel Embiid",
        "defense": "Jrue Holiday, Marcus Smart, Kawhi Leonard, Draymond Green, Rudy Gobert"
    }
    return presets.get(preset_name, "")

def create_game_simulator_ui():
    """Create the UI for the Game Simulator tool"""
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Team Setup")
            team1_name = gr.Textbox(label="Team 1 Name", value="All Stars")
            team1_players = gr.Textbox(
                placeholder="Enter players for Team 1, separated by commas",
                label="Team 1 Players",
                lines=3,
                value="LeBron James, Kevin Durant, Stephen Curry"
            )
            
            team2_name = gr.Textbox(label="Team 2 Name", value="Young Guns")
            team2_players = gr.Textbox(
                placeholder="Enter players for Team 2, separated by commas",
                label="Team 2 Players",
                lines=3,
                value="Luka Doncic, Ja Morant, Zion Williamson"
            )
            
            gr.Markdown("### Game Settings")
            quarters = gr.Slider(minimum=1, maximum=4, value=4, step=1, label="Number of Quarters")
            quarter_length = gr.Slider(minimum=6, maximum=12, value=12, step=1, label="Quarter Length (minutes)")
            
            gr.Markdown("### Preset Teams")
            with gr.Row():
                preset_team1_btn = gr.Button("Load Preset Team 1")
                preset_team1 = gr.Dropdown(
                    choices=["all_stars", "young_guns", "big_men", "guards", "legends", "offense", "defense"],
                    label="Team 1 Preset",
                    value="all_stars"
                )
            
            with gr.Row():
                preset_team2_btn = gr.Button("Load Preset Team 2")
                preset_team2 = gr.Dropdown(
                    choices=["all_stars", "young_guns", "big_men", "guards", "legends", "offense", "defense"],
                    label="Team 2 Preset",
                    value="young_guns"
                )
            
            simulate_btn = gr.Button("Run Game Simulation", variant="primary")
        
        with gr.Column(scale=2):
            sim_html = gr.HTML(label="Game Visualization")
            sim_output = gr.Markdown(label="Play-by-Play")
    
    # Load preset team handlers
    preset_team1_btn.click(
        fn=load_preset_team,
        inputs=[preset_team1],
        outputs=[team1_players]
    )
    
    preset_team2_btn.click(
        fn=load_preset_team,
        inputs=[preset_team2],
        outputs=[team2_players]
    )
    
    # Run simulation handler
    simulate_btn.click(
        fn=game_simulator,
        inputs=[team1_players, team2_players, team1_name, team2_name, quarters, quarter_length],
        outputs=[sim_output, sim_html]
    )
    
    return sim_output, sim_html

def create_interface():
    """Create and configure the Gradio interface for the application"""
    
    # Set up the title and theme
    title = "🏀 Fantasy Basketball Tools"
    description = """
    A suite of interactive tools for fantasy basketball managers. 
    Analyze draft value, compare matchups, track player consistency, and simulate games.
    """
    
    # Start the interface creation
    demo = gr.Blocks(
        title=title,
        css=CUSTOM_CSS
    )
    
    # Build the UI
    with demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)
        
        # Create tabs for different tools
        with gr.Tabs():
            with gr.Tab("Draft Helper"):
                create_draft_helper_ui()
            
            with gr.Tab("Matchup Analyzer"):
                create_matchup_analyzer_ui()
            
            with gr.Tab("Consistency Tracker"):
                create_consistency_tracker_ui()
            
            with gr.Tab("Game Simulator"):
                create_game_simulator_ui()
            
            # Add AI-powered tabs if available
            ai_available = create_ai_features_interface()
        
        # Add a footer
        gr.Markdown("""
        <div class="footer">
        <p>Fantasy Basketball Tools | Using NBA API data | Created with Gradio</p>
        <p>Data updates daily during the NBA season</p>
        </div>
        """)
    
    return demo