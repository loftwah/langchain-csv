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
                
                ai_status = gr.Markdown(
                    "✅ AI Ready" if is_ai_available else 
                    "⚠️ AI Not Available - Please start Ollama with: `ollama run llama3.2`"
                )
                
            with gr.Column(scale=2):
                # Output components
                player_summary = gr.Markdown(label="Summary")
                
                with gr.Row():
                    with gr.Column():
                        strengths_md = gr.Markdown(label="Strengths")
                    with gr.Column():
                        weaknesses_md = gr.Markdown(label="Weaknesses")
                
                draft_advice = gr.Markdown(label="Draft Advice")
                comparable_players = gr.Markdown(label="Comparable Players")
        
        # Handle analyze button click
        def analyze_player(player_name):
            if not player_name or player_name.strip() == "":
                return (
                    "Please enter a player name",
                    "No strengths to display",
                    "No weaknesses to display",
                    "No draft advice available",
                    "No comparable players available"
                )
            
            if not is_ai_available:
                return (
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
                analysis.get("summary", "No summary available"),
                strengths_text,
                weaknesses_text,
                draft_text,
                comparable_text
            )
        
        analyze_button.click(
            analyze_player,
            inputs=[player_name_input],
            outputs=[player_summary, strengths_md, weaknesses_md, draft_advice, comparable_players]
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
            with gr.Column():
                team2_name = gr.Textbox(label="Team 2 Name", value="Team Omega")
                team2_players = gr.Textbox(
                    label="Team 2 Players (comma-separated)",
                    placeholder="Joel Embiid, Luka Doncic, Jayson Tatum",
                    value="Joel Embiid, Luka Doncic, Jayson Tatum" if is_ai_available else "",
                    lines=3
                )
        
        analyze_matchup_button = gr.Button("Analyze Matchup", variant="primary")
        matchup_analysis = gr.Markdown(label="Matchup Analysis")
        
        def analyze_teams_matchup(team1_name, team1_input, team2_name, team2_input):
            if not is_ai_available:
                return "⚠️ AI Not Available - Please start Ollama with: `ollama run llama3.2`"
            
            # Parse team players
            team1_list = [p.strip() for p in team1_input.split(",") if p.strip()]
            team2_list = [p.strip() for p in team2_input.split(",") if p.strip()]
            
            if not team1_list or not team2_list:
                return "Please enter players for both teams"
            
            # Call the assistant to analyze the matchup
            analysis = assistant.analyze_matchup(
                team1_list,
                team2_list,
                team1_name,
                team2_name
            )
            
            return analysis
        
        analyze_matchup_button.click(
            analyze_teams_matchup,
            inputs=[team1_name, team1_players, team2_name, team2_players],
            outputs=[matchup_analysis]
        )
    
    # Fantasy Assistant Tab (Chat)
    with gr.Tab("Fantasy Assistant"):
        chat_history = gr.Chatbot(
            label="Chat with Fantasy Assistant",
            height=500
        )
        
        with gr.Row():
            question_input = gr.Textbox(
                label="Ask a question about fantasy basketball",
                placeholder="Which category should I prioritize in my draft?",
                lines=2
            )
            send_button = gr.Button("Send", variant="primary")
        
        def chat_with_assistant(message, history):
            if not is_ai_available:
                history.append((message, "⚠️ AI Not Available - Please start Ollama with: `ollama run llama3.2`"))
                return history, ""
            
            # Get answer from assistant
            answer = assistant.answer_fantasy_question(message)
            
            # Update history
            history.append((message, answer))
            
            return history, ""
        
        send_button.click(
            chat_with_assistant,
            inputs=[question_input, chat_history],
            outputs=[chat_history, question_input]
        )
    
    return is_ai_available 