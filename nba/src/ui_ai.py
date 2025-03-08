"""
AI features UI components for Fantasy Basketball Tools
"""

import gradio as gr
from .ai import NBAFantasyAssistant
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .config import NBA_COLORS
from typing import List, Dict, Any

def create_ai_features_interface():
    """Create the UI components for AI-powered features"""
    # Initialize the assistant
    assistant = NBAFantasyAssistant()
    
    # Check if AI is available
    is_ai_available = assistant.is_available()
    
    # Player Analysis Tab
    with gr.Tab("AI Player Analysis"):
        with gr.Row():
            with gr.Column(scale=1):
                player_name_input = gr.Textbox(
                    label="Enter Player Name",
                    placeholder="LeBron James",
                    value="LeBron James" if is_ai_available else ""
                )
                analyze_button = gr.Button("Analyze Player", variant="primary")
                
                # Add some preset player buttons for easy selection
                gr.Markdown("### Quick Player Select")
                with gr.Row():
                    preset_player1 = gr.Button("LeBron James")
                    preset_player2 = gr.Button("Stephen Curry")
                    
                with gr.Row():
                    preset_player3 = gr.Button("Giannis Antetokounmpo")
                    preset_player4 = gr.Button("Nikola Jokić")
                
                ai_status = gr.Markdown(
                    "✅ AI Ready" if is_ai_available else 
                    "⚠️ AI Not Available - Please start Ollama with: `ollama run llama3.2`"
                )
                
            with gr.Column(scale=2):
                # Loading indicator
                analysis_loading = gr.Markdown("", elem_id="analysis_loading_indicator")
                
                # Output components
                player_summary = gr.Markdown(label="Summary")
                
                with gr.Row():
                    with gr.Column():
                        strengths_md = gr.Markdown(label="Strengths")
                    with gr.Column():
                        weaknesses_md = gr.Markdown(label="Weaknesses")
                
                draft_advice = gr.Markdown(label="Draft Advice")
                comparable_players = gr.Markdown(label="Comparable Players")
        
        # Handle preset player buttons the simpler way
        preset_player1.click(lambda: "LeBron James", inputs=[], outputs=[player_name_input])
        preset_player2.click(lambda: "Stephen Curry", inputs=[], outputs=[player_name_input])
        preset_player3.click(lambda: "Giannis Antetokounmpo", inputs=[], outputs=[player_name_input])
        preset_player4.click(lambda: "Nikola Jokić", inputs=[], outputs=[player_name_input])
        
        # Handle analyze button click
        def analyze_player(player_name):
            if not player_name or player_name.strip() == "":
                return (
                    "",  # Clear loading indicator
                    "Please enter a player name",
                    "No strengths to display",
                    "No weaknesses to display",
                    "No draft advice available",
                    "No comparable players available"
                )
            
            if not is_ai_available:
                return (
                    "",  # Clear loading indicator 
                    "⚠️ AI Not Available - Please start Ollama with: `ollama run llama3.2`",
                    "AI Not Available",
                    "AI Not Available",
                    "AI Not Available",
                    "AI Not Available"
                )
            
            # Call the assistant to analyze the player
            analysis = assistant.analyze_player(player_name)
            
            if "error" in analysis:
                return (
                    "",  # Clear loading indicator
                    f"Error: {analysis['error']}",
                    "Error retrieving strengths",
                    "Error retrieving weaknesses",
                    "Error retrieving draft advice",
                    "Error retrieving comparable players"
                )
            
            # Format strengths
            strengths_text = "### Key Strengths\n\n"
            for s in analysis.get("strengths", []):
                strengths_text += f"**{s.get('category', '')}**: {s.get('explanation', '')}\n\n"
                strengths_text += f"*Fantasy Impact: {s.get('fantasy_impact', '')}*\n\n"
            
            # Format weaknesses
            weaknesses_text = "### Key Weaknesses\n\n"
            for w in analysis.get("weaknesses", []):
                weaknesses_text += f"**{w.get('category', '')}**: {w.get('explanation', '')}\n\n"
                weaknesses_text += f"*Fantasy Impact: {w.get('fantasy_impact', '')}*\n\n"
            
            # Format draft advice
            draft_text = f"### Draft Strategy for {analysis.get('player_name', player_name)}\n\n"
            draft_text += analysis.get("draft_advice", "No draft advice available")
            
            # Format comparable players
            comparable_text = "### Similar Players for Comparison\n\n"
            for i, player in enumerate(analysis.get("comparable_players", [])):
                comparable_text += f"{i+1}. {player}\n"
            
            return (
                "",  # Clear loading indicator
                analysis.get("summary", "No summary available"),
                strengths_text,
                weaknesses_text,
                draft_text,
                comparable_text
            )
        
        # Add loading state to analyze button
        analyze_button.click(
            # When button is clicked, show loading indicator first
            fn=lambda name: ("🔄 **Analyzing player...**\n\nThis may take a few moments as the LLM processes the data.", "", "", "", "", ""),
            inputs=[player_name_input],
            outputs=[analysis_loading, player_summary, strengths_md, weaknesses_md, draft_advice, comparable_players]
        ).then(
            # Then run the actual analysis
            fn=analyze_player,
            inputs=[player_name_input],
            outputs=[analysis_loading, player_summary, strengths_md, weaknesses_md, draft_advice, comparable_players]
        )
    
    # Matchup Analysis Tab
    with gr.Tab("AI Matchup Analysis"):
        with gr.Row():
            with gr.Column():
                team1_name = gr.Textbox(label="Team 1 Name", value="Team Alpha")
                team1_players = gr.Textbox(
                    label="Team 1 Players (comma-separated)",
                    placeholder="LeBron James, Stephen Curry, Kevin Durant",
                    value="LeBron James, Giannis Antetokounmpo, Nikola Jokic" if is_ai_available else "",
                    lines=3
                )
                
                # Add preset team buttons
                gr.Markdown("### Team 1: Quick Presets")
                with gr.Row():
                    team1_preset1 = gr.Button("All Stars")
                    team1_preset2 = gr.Button("Elite Guards")
                    team1_preset3 = gr.Button("Defensive Team")
                
            with gr.Column():
                team2_name = gr.Textbox(label="Team 2 Name", value="Team Omega")
                team2_players = gr.Textbox(
                    label="Team 2 Players (comma-separated)",
                    placeholder="Joel Embiid, Luka Doncic, Jayson Tatum",
                    value="Joel Embiid, Luka Doncic, Jayson Tatum" if is_ai_available else "",
                    lines=3
                )
                
                # Add preset team buttons
                gr.Markdown("### Team 2: Quick Presets")
                with gr.Row():
                    team2_preset1 = gr.Button("Young Guns")
                    team2_preset2 = gr.Button("Big Men")
                    team2_preset3 = gr.Button("Offensive Team")
        
        # Preset team handlers
        def get_preset_team(preset_name):
            presets = {
                "All Stars": "LeBron James, Kevin Durant, Stephen Curry, Giannis Antetokounmpo, Nikola Jokic",
                "Young Guns": "Luka Doncic, Trae Young, Ja Morant, Zion Williamson, Anthony Edwards",
                "Big Men": "Joel Embiid, Nikola Jokic, Anthony Davis, Karl-Anthony Towns, Bam Adebayo",
                "Elite Guards": "Stephen Curry, Damian Lillard, Kyrie Irving, Devin Booker, Donovan Mitchell",
                "Defensive Team": "Jrue Holiday, Marcus Smart, Kawhi Leonard, Draymond Green, Rudy Gobert",
                "Offensive Team": "James Harden, Stephen Curry, Kevin Durant, Giannis Antetokounmpo, Joel Embiid"
            }
            return presets.get(preset_name, "")
        
        team1_preset1.click(lambda: get_preset_team("All Stars"), inputs=[], outputs=[team1_players])
        team1_preset2.click(lambda: get_preset_team("Elite Guards"), inputs=[], outputs=[team1_players])
        team1_preset3.click(lambda: get_preset_team("Defensive Team"), inputs=[], outputs=[team1_players])
        
        team2_preset1.click(lambda: get_preset_team("Young Guns"), inputs=[], outputs=[team2_players])
        team2_preset2.click(lambda: get_preset_team("Big Men"), inputs=[], outputs=[team2_players])
        team2_preset3.click(lambda: get_preset_team("Offensive Team"), inputs=[], outputs=[team2_players])
        
        analyze_matchup_button = gr.Button("Analyze Matchup", variant="primary")
        
        # Add loading indicator
        matchup_loading = gr.Markdown("", elem_id="matchup_loading_indicator")
        matchup_analysis = gr.Markdown(label="Matchup Analysis")
        
        def analyze_teams_matchup(team1_name, team1_input, team2_name, team2_input):
            if not is_ai_available:
                return "", "⚠️ AI Not Available - Please start Ollama with: `ollama run llama3.2`"
            
            # Parse team players
            team1_list = [p.strip() for p in team1_input.split(",") if p.strip()]
            team2_list = [p.strip() for p in team2_input.split(",") if p.strip()]
            
            if not team1_list or not team2_list:
                return "", "Please enter players for both teams"
            
            # Call the assistant to analyze the matchup
            analysis = assistant.analyze_matchup(
                team1_list,
                team2_list,
                team1_name,
                team2_name
            )
            
            return "", analysis
        
        # Add loading state to analyze matchup button
        analyze_matchup_button.click(
            # When button is clicked, show loading indicator
            fn=lambda: ("🔄 **Analyzing matchup...**\n\nThis may take a moment while the AI evaluates both teams.", ""),
            inputs=[],
            outputs=[matchup_loading, matchup_analysis]
        ).then(
            # Then run the actual analysis
            fn=analyze_teams_matchup,
            inputs=[team1_name, team1_players, team2_name, team2_players],
            outputs=[matchup_loading, matchup_analysis]
        )
    
    # Fantasy Assistant Tab
    with gr.Tab("Fantasy Assistant"):
        gr.Markdown("### Fantasy Basketball Assistant")
        gr.Markdown("Ask any fantasy basketball question or use one of the preset questions below.")
        
        with gr.Row():
            with gr.Column(scale=2):
                user_question = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask for fantasy advice, player comparisons, strategy suggestions, etc.",
                    lines=3
                )
                
                # Adding preset questions that users can click on
                gr.Markdown("### Quick Questions")
                with gr.Column(elem_classes=["preset-section"]):
                    with gr.Column():
                        with gr.Row():
                            strategy_q1 = gr.Button("Should I prioritize guards or big men in the draft?", elem_classes=["preset-button"])
                            strategy_q2 = gr.Button("What's the optimal draft strategy for a 12-team league?", elem_classes=["preset-button"])
                        
                        with gr.Row():
                            player_q1 = gr.Button("Who are the most consistent fantasy performers?", elem_classes=["preset-button"])
                            player_q2 = gr.Button("Which rookies should I target in my draft?", elem_classes=["preset-button"])
                        
                        with gr.Row():
                            trade_q1 = gr.Button("Is trading LeBron for Giannis a good move?", elem_classes=["preset-button"])
                            trade_q2 = gr.Button("When is the best time to trade for injured stars?", elem_classes=["preset-button"])
                        
                        with gr.Row():
                            league_q1 = gr.Button("What scoring system is most balanced for fantasy?", elem_classes=["preset-button"])
                            league_q2 = gr.Button("How do I counter a team that's dominating in blocks?", elem_classes=["preset-button"])
                
                ask_btn = gr.Button("Ask Assistant", variant="primary")
                
            with gr.Column(scale=3):
                # Add loading indicator
                assistant_loading = gr.Markdown("", elem_id="assistant_loading_indicator", elem_classes=["loading-indicator"])
                assistant_response = gr.Markdown(label="Assistant Response")
        
        # Set up preset question button handlers
        strategy_q1.click(lambda: "Should I prioritize guards or big men in the draft?", inputs=[], outputs=[user_question])
        strategy_q2.click(lambda: "What's the optimal draft strategy for a 12-team league?", inputs=[], outputs=[user_question])
        player_q1.click(lambda: "Who are the most consistent fantasy performers?", inputs=[], outputs=[user_question])
        player_q2.click(lambda: "Which rookies should I target in my draft?", inputs=[], outputs=[user_question])
        trade_q1.click(lambda: "Is trading LeBron for Giannis a good move?", inputs=[], outputs=[user_question])
        trade_q2.click(lambda: "When is the best time to trade for injured stars?", inputs=[], outputs=[user_question])
        league_q1.click(lambda: "What scoring system is most balanced for fantasy?", inputs=[], outputs=[user_question])
        league_q2.click(lambda: "How do I counter a team that's dominating in blocks?", inputs=[], outputs=[user_question])
        
        # Handle ask button click
        def ask_fantasy_assistant(question):
            if not question or question.strip() == "":
                return "", "Please enter a question."
                
            if not is_ai_available:
                return "", "⚠️ AI Not Available - Please start Ollama with: `ollama run llama3.2`"
                
            # Call the assistant to answer the question
            try:
                response = assistant.answer_fantasy_question(question)
                # Response is now a string, return it directly
                return "", response
            except Exception as e:
                return "", f"Error: {str(e)}"
                
        # Add loading state to ask button
        ask_btn.click(
            # When button is clicked, show loading indicator
            fn=lambda: ("🔄 **Thinking...**\n\nThe AI assistant is processing your question.", ""),
            inputs=[],
            outputs=[assistant_loading, assistant_response]
        ).then(
            # Then run the actual query
            fn=ask_fantasy_assistant,
            inputs=[user_question],
            outputs=[assistant_loading, assistant_response]
        )
    
    return is_ai_available