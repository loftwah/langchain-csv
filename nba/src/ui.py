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
        fn=lambda teams, scoring, pos: draft_helper.analyze_draft_value(int(teams), scoring, int(pos)),
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
    
    analyze_matchup_btn.click(
        fn=lambda t1, t2: matchup_analyzer.compare_teams(t1.split(","), t2.split(",")),
        inputs=[team1_players, team2_players],
        outputs=matchup_output
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
    
    analyze_consistency_btn.click(
        fn=lambda player, period: consistency_tracker.analyze_consistency(player, period),
        inputs=[player_name, time_period],
        outputs=consistency_output
    )
    
    return consistency_output

def create_game_simulator_ui():
    """Create the UI for the Game Simulator tool"""
    with gr.Row():
        with gr.Column():
            team1 = gr.Textbox(
                placeholder="Enter team name (e.g., Lakers)",
                label="Team 1"
            )
            team2 = gr.Textbox(
                placeholder="Enter team name (e.g., Celtics)",
                label="Team 2"
            )
            num_simulations = gr.Slider(minimum=10, maximum=1000, value=100, step=10, label="Number of Simulations")
            simulate_btn = gr.Button("Run Simulation", variant="primary")
        
        with gr.Column():
            simulation_output = gr.Plot(label="Simulation Results")
    
    simulate_btn.click(
        fn=lambda t1, t2, sims: game_simulator.simulate_game(t1, t2, int(sims)),
        inputs=[team1, team2, num_simulations],
        outputs=simulation_output
    )
    
    return simulation_output

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