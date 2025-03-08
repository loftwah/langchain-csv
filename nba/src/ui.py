import gradio as gr

from .config import CUSTOM_CSS
from .tools import draft_helper, matchup_analyzer, consistency_tracker

def create_interface():
    """Create the Gradio web interface for the Fantasy Basketball Tools"""
    
    with gr.Blocks(title="Loftwah's Fantasy Basketball Tools", css=CUSTOM_CSS) as demo:
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="color: #F7B801; font-size: 36px; margin-bottom: 5px;">🏀 Loftwah's Fantasy Basketball Assistant</h1>
            <p style="font-size: 18px;">Professional analytics to dominate your fantasy basketball league</p>
            <a href="https://linkarooie.com/loftwah" target="_blank" style="color: #F7B801; font-size: 14px;">
                Visit Loftwah's Website
            </a>
        </div>
        """)
        
        with gr.Tabs():
            # Draft Helper Tab
            with gr.Tab("🏆 Draft Helper"):
                with gr.Group(elem_classes=["content-block"]):
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
                            draft_btn = gr.Button("Find Value Players", variant="primary")
                        
                        with gr.Column(scale=2):
                            draft_plot = gr.Plot(label="Top Players by Value", elem_classes=["plot-container"])
                    
                    draft_table = gr.Dataframe(label="Player Rankings")
                    
                    draft_btn.click(
                        fn=draft_helper,
                        inputs=[draft_scoring, min_games, stat_category],
                        outputs=[draft_table, draft_plot]
                    )
            
            # Matchup Analyzer Tab
            with gr.Tab("⚔️ Matchup Analyzer"):
                with gr.Group(elem_classes=["content-block"]):
                    gr.Markdown("## Weekly Matchup Analyzer")
                    gr.Markdown("Compare two fantasy teams to predict matchup outcomes.")
                    
                    # Move preset team compositions to the top for better visibility
                    gr.Markdown("### Quick Team Matchup Presets", elem_classes=["section-header"])
                    gr.Markdown("Click any preset to instantly load two teams for comparison:")
                    
                    with gr.Row():
                        preset1_btn = gr.Button("⭐ Stars vs All-Around", size="lg", elem_classes=["preset-button"])
                        preset2_btn = gr.Button("🏀 Scoring vs Defense", size="lg", elem_classes=["preset-button"])
                        preset3_btn = gr.Button("⚡ Young Guns vs Veterans", size="lg", elem_classes=["preset-button"])
                    
                    gr.Markdown("### Or enter your own teams:")
                    
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
                            matchup_btn = gr.Button("Compare Teams", variant="primary")
                    
                    # Define preset team compositions
                    def load_preset_stars_allround():
                        return {
                            team1: "LeBron James, Kevin Durant, Stephen Curry, Kyrie Irving, Devin Booker",
                            team2: "Nikola Jokic, Giannis Antetokounmpo, Luka Doncic, Jayson Tatum, Anthony Edwards"
                        }
                    
                    def load_preset_offense_defense():
                        return {
                            team1: "Damian Lillard, Trae Young, Anthony Edwards, Zach LaVine, Karl-Anthony Towns",
                            team2: "Rudy Gobert, Bam Adebayo, Draymond Green, Jrue Holiday, Mikal Bridges"
                        }
                    
                    def load_preset_young_vets():
                        return {
                            team1: "Anthony Edwards, LaMelo Ball, Cade Cunningham, Scottie Barnes, Paolo Banchero",
                            team2: "Chris Paul, LeBron James, Kevin Durant, Jimmy Butler, Al Horford"
                        }
                    
                    # Connect preset buttons
                    preset1_btn.click(fn=load_preset_stars_allround, outputs=[team1, team2])
                    preset2_btn.click(fn=load_preset_offense_defense, outputs=[team1, team2])
                    preset3_btn.click(fn=load_preset_young_vets, outputs=[team1, team2])
                    
                    matchup_result = gr.Markdown(label="Matchup Analysis")
                    matchup_plot = gr.Plot(label="Category Comparison", elem_classes=["plot-container"])
                    
                    matchup_btn.click(
                        fn=matchup_analyzer,
                        inputs=[team1, team2, matchup_scoring],
                        outputs=[matchup_result, matchup_plot]
                    )
            
            # Consistency Tracker Tab
            with gr.Tab("📊 Consistency Tracker"):
                with gr.Group(elem_classes=["content-block"]):
                    gr.Markdown("## Player Consistency Tracker")
                    gr.Markdown("Analyze a player's consistency to identify reliable starters vs. boom/bust players.")
                    
                    # Move player preset buttons to the top
                    gr.Markdown("### Select a Popular Player", elem_classes=["section-header"])
                    gr.Markdown("Click any player button to analyze their fantasy consistency:")
                    
                    with gr.Row():
                        player1_btn = gr.Button("🔥 LeBron James", size="lg", elem_classes=["preset-button"])
                        player2_btn = gr.Button("🧙‍♂️ Nikola Jokić", size="lg", elem_classes=["preset-button"])
                        player3_btn = gr.Button("👨‍🍳 Stephen Curry", size="lg", elem_classes=["preset-button"])
                    
                    with gr.Row():
                        player4_btn = gr.Button("🦌 Giannis Antetokounmpo", size="lg", elem_classes=["preset-button"])
                        player5_btn = gr.Button("🪄 Luka Dončić", size="lg", elem_classes=["preset-button"])
                        player6_btn = gr.Button("🦅 Joel Embiid", size="lg", elem_classes=["preset-button"])
                    
                    gr.Markdown("### Or analyze any other player:")
                    
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
                            consistency_btn = gr.Button("Analyze Player", variant="primary")
                    
                    # Define preset player functions
                    def set_player_lebron():
                        return "LeBron James"
                    
                    def set_player_jokic():
                        return "Nikola Jokic"  # Using standard ASCII version for maximum compatibility
                    
                    def set_player_curry():
                        return "Stephen Curry"
                    
                    def set_player_giannis():
                        return "Giannis Antetokounmpo"
                    
                    def set_player_luka():
                        return "Luka Doncic"  # Using standard ASCII version for maximum compatibility
                    
                    def set_player_embiid():
                        return "Joel Embiid"
                    
                    # Define output components first
                    consistency_result = gr.Markdown(label="Consistency Analysis")
                    consistency_plot = gr.Plot(label="Performance Consistency", elem_classes=["plot-container"])
                    
                    # Auto-analyze after selecting a preset player
                    def set_player_and_analyze(player_fn):
                        name = player_fn()
                        # Using default values for the other parameters
                        result, plot = consistency_tracker(name, num_games=10, scoring_system='standard')
                        return name, result, plot
                    
                    # Connect player preset buttons with auto-analysis
                    player1_btn.click(
                        fn=lambda: set_player_and_analyze(set_player_lebron), 
                        outputs=[player_name, consistency_result, consistency_plot]
                    )
                    player2_btn.click(
                        fn=lambda: set_player_and_analyze(set_player_jokic), 
                        outputs=[player_name, consistency_result, consistency_plot]
                    )
                    player3_btn.click(
                        fn=lambda: set_player_and_analyze(set_player_curry), 
                        outputs=[player_name, consistency_result, consistency_plot]
                    )
                    player4_btn.click(
                        fn=lambda: set_player_and_analyze(set_player_giannis), 
                        outputs=[player_name, consistency_result, consistency_plot]
                    )
                    player5_btn.click(
                        fn=lambda: set_player_and_analyze(set_player_luka), 
                        outputs=[player_name, consistency_result, consistency_plot]
                    )
                    player6_btn.click(
                        fn=lambda: set_player_and_analyze(set_player_embiid), 
                        outputs=[player_name, consistency_result, consistency_plot]
                    )
                    
                    # Regular analysis button for custom player input
                    consistency_btn.click(
                        fn=consistency_tracker,
                        inputs=[player_name, num_games, consistency_scoring],
                        outputs=[consistency_result, consistency_plot]
                    )
        
        gr.HTML("""
        <div class="footer">
            <p>Powered by NBA data | Created by <a href="https://linkarooie.com/loftwah" target="_blank">Loftwah</a> | © 2025</p>
            <p style="font-size: 14px;">
                <a href="https://linkarooie.com/loftwah" target="_blank">Website</a> | 
                <a href="https://twitter.com/loftwah" target="_blank">Twitter</a> | 
                <a href="https://github.com/loftwah" target="_blank">GitHub</a>
            </p>
        </div>
        """)
    
    return demo 