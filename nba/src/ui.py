import gradio as gr

from .config import CUSTOM_CSS
from .tools import draft_helper, matchup_analyzer, consistency_tracker, game_simulator

def create_interface():
    """Create the Gradio web interface for the Fantasy Basketball Tools"""
    
    with gr.Blocks(title="Loftwah's Fantasy Basketball Tools", css=CUSTOM_CSS) as demo:
        # Enhanced header with banner and description
        gr.HTML("""
        <div style="text-align: center; max-width: 100%; overflow: hidden; margin-bottom: 30px; background: linear-gradient(90deg, #17408B, #C9082A); padding: 30px 0; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
            <h1 style="color: #FFFFFF; font-size: 42px; margin-bottom: 15px; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">🏀 Fantasy Basketball Assistant</h1>
            <p style="font-size: 18px; color: #FFFFFF; margin-bottom: 20px; max-width: 800px; margin-left: auto; margin-right: auto;">
                Professional analytics to dominate your fantasy basketball league using official NBA data
            </p>
            <div style="display: flex; justify-content: center; gap: 20px; margin-top: 15px;">
                <div style="background: rgba(255,255,255,0.1); padding: 10px 20px; border-radius: 8px; backdrop-filter: blur(5px);">
                    <h3 style="color: #F7B801; margin: 0;">Data-Driven Drafting</h3>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 10px 20px; border-radius: 8px; backdrop-filter: blur(5px);">
                    <h3 style="color: #F7B801; margin: 0;">Matchup Analysis</h3>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 10px 20px; border-radius: 8px; backdrop-filter: blur(5px);">
                    <h3 style="color: #F7B801; margin: 0;">Player Consistency</h3>
                </div>
            </div>
            <p style="margin-top: 20px; font-size: 14px; color: #FFFFFF;">
                <a href="https://linkarooie.com/loftwah" target="_blank" style="color: #F7B801; text-decoration: underline;">
                    Visit Author's Website
                </a> | 
                <a href="https://github.com/loftwah/langchain-csv" target="_blank" style="color: #F7B801; text-decoration: underline;">
                    GitHub Repository
                </a>
            </p>
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
            
            # Game Simulator Tab
            with gr.Tab("🎮 Game Simulator"):
                with gr.Group(elem_classes=["content-block"]):
                    gr.Markdown("## NBA Game Simulator")
                    gr.Markdown("Simulate a full NBA basketball game with your favorite players and watch the play-by-play action!")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Team Setup")
                            
                            team1_name = gr.Textbox(
                                label="Team 1 Name",
                                value="Dream Team",
                                placeholder="Enter a name for Team 1"
                            )
                            
                            team1_players = gr.Textbox(
                                label="Team 1 Players (comma-separated)",
                                placeholder="e.g., LeBron James, Stephen Curry, Nikola Jokic",
                                lines=3
                            )
                            
                            team2_name = gr.Textbox(
                                label="Team 2 Name",
                                value="All-Stars",
                                placeholder="Enter a name for Team 2"
                            )
                            
                            team2_players = gr.Textbox(
                                label="Team 2 Players (comma-separated)",
                                placeholder="e.g., Giannis Antetokounmpo, Kevin Durant, Joel Embiid",
                                lines=3
                            )
                            
                            gr.Markdown("### Game Settings")
                            
                            with gr.Row():
                                quarters = gr.Slider(
                                    minimum=1, maximum=4, value=4, step=1,
                                    label="Number of Quarters"
                                )
                                
                                quarter_length = gr.Slider(
                                    minimum=6, maximum=12, value=12, step=1,
                                    label="Quarter Length (minutes)"
                                )
                            
                            # Game presets for fun combinations
                            gr.Markdown("### Quick Team Presets")
                            
                            with gr.Row():
                                preset_goats = gr.Button("🐐 All-Time Greats")
                                preset_current = gr.Button("🔥 Today's Superstars")
                                preset_shooters = gr.Button("🎯 3-Point Specialists")
                                
                            with gr.Row():
                                preset_defenders = gr.Button("🛡️ Defensive Stoppers")
                                preset_bigmen = gr.Button("🏔️ Dominant Big Men")
                                preset_playmakers = gr.Button("👀 Elite Playmakers")
                            
                            simulate_btn = gr.Button("Simulate Game!", variant="primary")
                            
                        with gr.Column(scale=1):
                            # Game visualization
                            game_plot = gr.Plot(label="Game Stats", elem_classes=["plot-container"])
                    
                    # Play-by-play results
                    play_by_play = gr.Markdown(label="Play-by-Play")
                    
                    # Preset functions
                    def load_preset_goats():
                        return {
                            team1_players: "Michael Jordan, LeBron James, Kobe Bryant, Magic Johnson, Kareem Abdul-Jabbar",
                            team2_players: "Larry Bird, Wilt Chamberlain, Shaquille O'Neal, Hakeem Olajuwon, Bill Russell",
                            team1_name: "GOAT Squad",
                            team2_name: "Legends"
                        }
                    
                    def load_preset_current():
                        return {
                            team1_players: "Giannis Antetokounmpo, Nikola Jokic, LeBron James, Stephen Curry, Kevin Durant",
                            team2_players: "Joel Embiid, Luka Doncic, Jayson Tatum, Damian Lillard, Anthony Davis",
                            team1_name: "Current Elites",
                            team2_name: "Rising Stars"
                        }
                    
                    def load_preset_shooters():
                        return {
                            team1_players: "Stephen Curry, Klay Thompson, Ray Allen, Reggie Miller, Kyle Korver",
                            team2_players: "Damian Lillard, Trae Young, Duncan Robinson, Buddy Hield, Davis Bertans",
                            team1_name: "Splash Brothers & Co.",
                            team2_name: "New Wave Snipers"
                        }
                    
                    def load_preset_defenders():
                        return {
                            team1_players: "Kawhi Leonard, Rudy Gobert, Draymond Green, Ben Wallace, Gary Payton",
                            team2_players: "Tony Allen, Dikembe Mutombo, Dennis Rodman, Marcus Smart, Jrue Holiday",
                            team1_name: "The Wall",
                            team2_name: "No Entry"
                        }
                    
                    def load_preset_bigmen():
                        return {
                            team1_players: "Shaquille O'Neal, Hakeem Olajuwon, Kareem Abdul-Jabbar, Tim Duncan, David Robinson",
                            team2_players: "Nikola Jokic, Joel Embiid, Anthony Davis, Karl-Anthony Towns, DeMarcus Cousins",
                            team1_name: "Classic Towers",
                            team2_name: "Modern Bigs"
                        }
                    
                    def load_preset_playmakers():
                        return {
                            team1_players: "Magic Johnson, John Stockton, Chris Paul, Jason Kidd, Steve Nash",
                            team2_players: "Luka Doncic, Nikola Jokic, LeBron James, James Harden, Trae Young",
                            team1_name: "Pass First",
                            team2_name: "Point Gods"
                        }
                    
                    preset_goats.click(load_preset_goats, outputs=[team1_players, team2_players, team1_name, team2_name])
                    preset_current.click(load_preset_current, outputs=[team1_players, team2_players, team1_name, team2_name])
                    preset_shooters.click(load_preset_shooters, outputs=[team1_players, team2_players, team1_name, team2_name])
                    preset_defenders.click(load_preset_defenders, outputs=[team1_players, team2_players, team1_name, team2_name])
                    preset_bigmen.click(load_preset_bigmen, outputs=[team1_players, team2_players, team1_name, team2_name])
                    preset_playmakers.click(load_preset_playmakers, outputs=[team1_players, team2_players, team1_name, team2_name])
                    
                    # Connect the simulate button
                    simulate_btn.click(
                        game_simulator,
                        inputs=[team1_players, team2_players, team1_name, team2_name, quarters, quarter_length],
                        outputs=[play_by_play, game_plot]
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