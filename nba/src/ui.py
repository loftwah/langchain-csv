import gradio as gr
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from .config import NBA_COLORS, CUSTOM_CSS, PRESETS
from .tools import draft_helper, matchup_analyzer, consistency_tracker, game_simulator
from .api import get_player_id, get_players, get_player_games
from .presets import create_preset_section, load_team_preset, create_preset_buttons, get_preset_draft_settings

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
            
            # Add preset draft configurations section
            gr.Markdown("### Quick Presets")
            with gr.Column(elem_classes=["preset-section"]):
                with gr.Row():
                    std_12_btn = gr.Button("Standard 12-Team", elem_classes=["preset-button"])
                    std_10_btn = gr.Button("Standard 10-Team", elem_classes=["preset-button"])
                
                with gr.Row():
                    cat_12_btn = gr.Button("Categories 12-Team", elem_classes=["preset-button"])
                    cat_10_btn = gr.Button("Categories 10-Team", elem_classes=["preset-button"])
            
            analyze_btn = gr.Button("Generate Draft Analysis", variant="primary")
    
    with gr.Row():
        with gr.Column(scale=2):
            draft_plot = gr.Plot(label="Top Players Visualization")
            
        with gr.Column(scale=3):
            output_area = gr.Dataframe(label="Draft Rankings and Analysis")
    
    # Set up preset button handlers
    draft_presets = get_preset_draft_settings()
    
    def apply_draft_preset(preset_key):
        preset = draft_presets.get(preset_key, {})
        return preset.get("teams", 12), preset.get("scoring", "Standard Points"), preset.get("position", 6)
    
    std_12_btn.click(
        fn=lambda: apply_draft_preset("standard_12_team"),
        inputs=[],
        outputs=[num_teams, scoring_type, draft_pos]
    )
    
    std_10_btn.click(
        fn=lambda: apply_draft_preset("standard_10_team"),
        inputs=[],
        outputs=[num_teams, scoring_type, draft_pos]
    )
    
    cat_12_btn.click(
        fn=lambda: apply_draft_preset("categories_12_team"),
        inputs=[],
        outputs=[num_teams, scoring_type, draft_pos]
    )
    
    cat_10_btn.click(
        fn=lambda: apply_draft_preset("categories_10_team"),
        inputs=[],
        outputs=[num_teams, scoring_type, draft_pos]
    )
    
    def process_draft_helper(teams, scoring, pos):
        df, fig = draft_helper(int(teams), scoring, int(pos))
        return fig, df
    
    analyze_btn.click(
        fn=process_draft_helper,
        inputs=[num_teams, scoring_type, draft_pos],
        outputs=[draft_plot, output_area]
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
            
            # Add preset team buttons for Team 1
            gr.Markdown("#### Quick Team Select")
            with gr.Row():
                team1_stars_btn = gr.Button("All Stars", elem_classes=["preset-button"])
                team1_guards_btn = gr.Button("Guards", elem_classes=["preset-button"])
            with gr.Row():
                team1_bigmen_btn = gr.Button("Big Men", elem_classes=["preset-button"])
                team1_defense_btn = gr.Button("Defense", elem_classes=["preset-button"])
                
        with gr.Column():
            gr.Markdown("### Team 2")
            team2_players = gr.Textbox(
                placeholder="Enter players, separated by commas",
                label="Team 2 Players"
            )
            
            # Add preset team buttons for Team 2
            gr.Markdown("#### Quick Team Select")
            with gr.Row():
                team2_young_btn = gr.Button("Young Guns", elem_classes=["preset-button"])
                team2_offense_btn = gr.Button("Offense", elem_classes=["preset-button"])
            with gr.Row():
                team2_legends_btn = gr.Button("Legends", elem_classes=["preset-button"])
                team2_forwards_btn = gr.Button("Forwards", elem_classes=["preset-button"])
    
    analyze_matchup_btn = gr.Button("Analyze Matchup", variant="primary")
    matchup_output = gr.Plot(label="Matchup Analysis")
    matchup_text = gr.Markdown(label="Matchup Results")
    
    # Team preset handlers
    team1_stars_btn.click(lambda: load_team_preset("all_stars"), inputs=[], outputs=[team1_players])
    team1_guards_btn.click(lambda: load_team_preset("guards"), inputs=[], outputs=[team1_players])
    team1_bigmen_btn.click(lambda: load_team_preset("big_men"), inputs=[], outputs=[team1_players])
    team1_defense_btn.click(lambda: load_team_preset("defense"), inputs=[], outputs=[team1_players])
    
    team2_young_btn.click(lambda: load_team_preset("young_guns"), inputs=[], outputs=[team2_players])
    team2_offense_btn.click(lambda: load_team_preset("offense"), inputs=[], outputs=[team2_players])
    team2_legends_btn.click(lambda: load_team_preset("legends"), inputs=[], outputs=[team2_players])
    # Adding a preset for forwards (need to update in config)
    forwards_preset = "LeBron James, Kevin Durant, Kawhi Leonard, Jayson Tatum, Jimmy Butler"
    team2_forwards_btn.click(lambda: forwards_preset, inputs=[], outputs=[team2_players])
    
    analyze_matchup_btn.click(
        fn=matchup_analyzer,
        inputs=[team1_players, team2_players],
        outputs=[matchup_text, matchup_output]
    )
    
    return matchup_output

def create_consistency_tracker_ui():
    """Create the UI for the Consistency Tracker tool"""
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Player Selection")
            player_name = gr.Textbox(
                placeholder="Enter player name (e.g., Nikola Jokić)",
                label="Player Name",
                info="You can use names with or without special characters (ć, č, etc.)"
            )
            
            # Add popular player presets for quick selection with proper special characters
            gr.Markdown("#### Popular Players")
            with gr.Row():
                with gr.Column(scale=1):
                    player_btn1 = gr.Button("LeBron James", elem_classes=["preset-button"])
                    player_btn2 = gr.Button("Stephen Curry", elem_classes=["preset-button"])
                with gr.Column(scale=1):
                    player_btn3 = gr.Button("Nikola Jokić", elem_classes=["preset-button"])
                    player_btn4 = gr.Button("Giannis Antetokounmpo", elem_classes=["preset-button"])
            
            with gr.Row():
                with gr.Column(scale=1):
                    player_btn5 = gr.Button("Luka Dončić", elem_classes=["preset-button"])
                    player_btn6 = gr.Button("Joel Embiid", elem_classes=["preset-button"])
                with gr.Column(scale=1):
                    player_btn7 = gr.Button("Kevin Durant", elem_classes=["preset-button"])
                    player_btn8 = gr.Button("Jayson Tatum", elem_classes=["preset-button"])
            
            gr.Markdown("### Analysis Options")
            time_period = gr.Dropdown(
                choices=["Last 10 Games", "Last 20 Games", "Last 30 Days", "Season"],
                value="Last 10 Games",
                label="Time Period",
                info="Select the time period for analysis"
            )
            
            scoring_system = gr.Dropdown(
                choices=["standard", "categories", "custom"],
                value="standard",
                label="Scoring System",
                info="Select fantasy scoring system to use"
            )
            
            analyze_consistency_btn = gr.Button("Analyze Consistency", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.Markdown("### Consistency Analysis Results")
            consistency_output = gr.Plot(label="Visual Analysis")
            consistency_text = gr.Markdown(label="Insights")
    
    # Set up player preset buttons with exact names including special characters
    player_btn1.click(lambda: "LeBron James", inputs=[], outputs=[player_name])
    player_btn2.click(lambda: "Stephen Curry", inputs=[], outputs=[player_name])
    player_btn3.click(lambda: "Nikola Jokić", inputs=[], outputs=[player_name])
    player_btn4.click(lambda: "Giannis Antetokounmpo", inputs=[], outputs=[player_name])
    player_btn5.click(lambda: "Luka Dončić", inputs=[], outputs=[player_name])
    player_btn6.click(lambda: "Joel Embiid", inputs=[], outputs=[player_name])
    player_btn7.click(lambda: "Kevin Durant", inputs=[], outputs=[player_name])
    player_btn8.click(lambda: "Jayson Tatum", inputs=[], outputs=[player_name])
    
    analyze_consistency_btn.click(
        fn=consistency_tracker,
        inputs=[player_name, time_period, scoring_system],
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
            
            gr.Markdown("### Team 1 Presets")
            with gr.Row():
                team1_preset1 = gr.Button("All Stars", elem_classes=["preset-button"])
                team1_preset2 = gr.Button("Guards", elem_classes=["preset-button"])
                team1_preset3 = gr.Button("Big Men", elem_classes=["preset-button"])
            
            with gr.Row():
                team1_preset4 = gr.Button("Legends", elem_classes=["preset-button"])
                team1_preset5 = gr.Button("Defense", elem_classes=["preset-button"])
                team1_preset6 = gr.Button("Offense", elem_classes=["preset-button"])
            
            gr.Markdown("### Team 2 Presets")
            with gr.Row():
                team2_preset1 = gr.Button("Young Guns", elem_classes=["preset-button"])
                team2_preset2 = gr.Button("Guards", elem_classes=["preset-button"])
                team2_preset3 = gr.Button("Big Men", elem_classes=["preset-button"])
            
            with gr.Row():
                team2_preset4 = gr.Button("Legends", elem_classes=["preset-button"])
                team2_preset5 = gr.Button("Defense", elem_classes=["preset-button"])
                team2_preset6 = gr.Button("Offense", elem_classes=["preset-button"])
            
            simulate_btn = gr.Button("Run Game Simulation", variant="primary")
        
        with gr.Column(scale=2):
            sim_html = gr.HTML(label="Game Visualization")
            sim_output = gr.Markdown(label="Play-by-Play")
    
    # Team 1 preset handlers
    team1_preset1.click(lambda: load_team_preset("all_stars"), inputs=[], outputs=[team1_players])
    team1_preset2.click(lambda: load_team_preset("guards"), inputs=[], outputs=[team1_players])
    team1_preset3.click(lambda: load_team_preset("big_men"), inputs=[], outputs=[team1_players])
    team1_preset4.click(lambda: load_team_preset("legends"), inputs=[], outputs=[team1_players])
    team1_preset5.click(lambda: load_team_preset("defense"), inputs=[], outputs=[team1_players])
    team1_preset6.click(lambda: load_team_preset("offense"), inputs=[], outputs=[team1_players])
    
    # Team 2 preset handlers
    team2_preset1.click(lambda: load_team_preset("young_guns"), inputs=[], outputs=[team2_players])
    team2_preset2.click(lambda: load_team_preset("guards"), inputs=[], outputs=[team2_players])
    team2_preset3.click(lambda: load_team_preset("big_men"), inputs=[], outputs=[team2_players])
    team2_preset4.click(lambda: load_team_preset("legends"), inputs=[], outputs=[team2_players])
    team2_preset5.click(lambda: load_team_preset("defense"), inputs=[], outputs=[team2_players])
    team2_preset6.click(lambda: load_team_preset("offense"), inputs=[], outputs=[team2_players])
    
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
    Made with ❤️ by <a href="https://linkarooie.com/loftwah" target="_blank">Loftwah</a>
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
        <p>Made with ❤️ by <a href="https://linkarooie.com/loftwah" target="_blank">Loftwah</a></p>
        </div>
        """)
        
        # Add floating attribution
        gr.Markdown("""
        <div class="app-footer">
        <a href="https://linkarooie.com/loftwah" target="_blank"></a>
        </div>
        """)
    
    return demo