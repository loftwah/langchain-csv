"""
AI integration module for Fantasy Basketball Tools
Using LangChain with Ollama (Llama 3.2)
"""

import os
import pandas as pd
import json
from functools import lru_cache
from typing import List, Dict, Any, Optional

# LangChain imports - updated to use new modules
from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel, Field

# Local imports
from .api import get_players, get_player_stats, get_player_games, get_league_leaders
from .fantasy import calculate_fantasy_points, calculate_consistency

# Initialize Ollama with Llama 3.2
@lru_cache(maxsize=1)
def get_llm():
    """Initialize and return the Ollama LLM instance with Llama 3.2"""
    try:
        # Set the model - default to llama3 but allow override
        model_name = os.environ.get("NBA_LLM_MODEL", "llama3")
        
        # Initialize the LLM
        llm = OllamaLLM(model=model_name)
        
        # Test the LLM connection
        test_response = llm.invoke("Say 'NBA Fantasy Tools Ready'")
        if "NBA Fantasy Tools Ready" in test_response:
            print(f"✅ Successfully connected to Ollama with model: {model_name}")
        else:
            print(f"⚠️ Ollama connected but response unexpected: {test_response[:50]}...")
        
        return llm
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        print("⚠️ Some AI features will be unavailable. Please make sure Ollama is running.")
        print("   Run: ollama run llama3")
        return None

# Player analysis models
class PlayerStrength(BaseModel):
    """Model for player strengths analysis"""
    category: str = Field(description="Statistical category (e.g., 'Scoring', 'Rebounding', 'Passing')")
    explanation: str = Field(description="Brief explanation of the strength")
    fantasy_impact: str = Field(description="How this impacts fantasy value")

class PlayerWeakness(BaseModel):
    """Model for player weaknesses analysis"""
    category: str = Field(description="Statistical category (e.g., 'Turnovers', 'Free throws')")
    explanation: str = Field(description="Brief explanation of the weakness")
    fantasy_impact: str = Field(description="How this negatively impacts fantasy value")

class PlayerAnalysis(BaseModel):
    """Complete player analysis"""
    player_name: str = Field(description="Name of the player")
    summary: str = Field(description="One paragraph summary of the player's fantasy value")
    strengths: List[PlayerStrength] = Field(description="List of player strengths")
    weaknesses: List[PlayerWeakness] = Field(description="List of player weaknesses")
    draft_advice: str = Field(description="Advice for drafting this player")
    comparable_players: List[str] = Field(description="List of 3-5 similar players for comparison")

# Vector database for NBA knowledge
def setup_nba_knowledge_base():
    """Set up a vector database with NBA knowledge"""
    try:
        # Create a simple NBA knowledge base from player stats
        players_df = get_players()
        stats_df = get_player_stats()
        
        # Join player info with stats
        if not stats_df.empty and not players_df.empty:
            # Standardize ID columns
            if 'PERSON_ID' in players_df.columns and 'PLAYER_ID' in stats_df.columns:
                players_df = players_df.rename(columns={'PERSON_ID': 'PLAYER_ID'})
            
            # Merge datasets
            nba_data = pd.merge(
                players_df, 
                stats_df,
                on='PLAYER_ID',
                how='inner'
            )
            
            # Convert to documents
            documents = []
            for _, row in nba_data.iterrows():
                # Create a text representation of player stats
                player_text = f"Player: {row.get('DISPLAY_FIRST_LAST', '')}\n"
                player_text += f"Position: {row.get('POSITION', '')}\n"
                player_text += f"Team: {row.get('TEAM_NAME', '')}\n"
                player_text += f"Stats: {row.get('PTS', 0)} PPG, {row.get('REB', 0)} RPG, {row.get('AST', 0)} APG, "
                player_text += f"{row.get('STL', 0)} SPG, {row.get('BLK', 0)} BPG\n"
                player_text += f"Shooting: {row.get('FG_PCT', 0)*100:.1f}% FG, {row.get('FT_PCT', 0)*100:.1f}% FT, "
                player_text += f"{row.get('FG3_PCT', 0)*100:.1f}% 3PT\n"
                
                # Add to documents
                documents.append(Document(
                    page_content=player_text,
                    metadata={
                        "player_id": str(row.get('PLAYER_ID', '')),
                        "player_name": row.get('DISPLAY_FIRST_LAST', ''),
                        "team": row.get('TEAM_NAME', '')
                    }
                ))
            
            # Initialize text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            
            # Split documents
            splits = text_splitter.split_documents(documents)
            
            # Create embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                cache_folder="./nba_api_cache/embeddings"
            )
            
            # Create vector store
            vector_store = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory="./nba_api_cache/vectordb"
            )
            
            return vector_store
        
        return None
    except Exception as e:
        print(f"Error setting up knowledge base: {e}")
        return None

# NBA Fantasy Assistant using LangChain
class NBAFantasyAssistant:
    """NBA Fantasy Assistant powered by LangChain and Ollama"""
    
    def __init__(self):
        """Initialize the NBA Fantasy Assistant"""
        self.llm = get_llm()
        self.vector_store = None
        
        # Try to initialize vector store but don't block if it fails
        try:
            self.vector_store = setup_nba_knowledge_base()
        except Exception as e:
            print(f"Vector store initialization failed: {e}")
    
    def is_available(self):
        """Check if the LLM is available"""
        return self.llm is not None
    
    def analyze_player(self, player_name):
        """Analyze a player for fantasy basketball"""
        if not self.is_available():
            return {
                "error": "LLM not available. Please ensure Ollama is running with Llama 3.2."
            }
        
        try:
            # Get player data
            player_id = None
            from .api import get_player_id
            player_id = get_player_id(player_name)
            
            if not player_id:
                return {
                    "error": f"Could not find player with name '{player_name}'"
                }
            
            # Get player stats and recent games
            player_stats = get_player_stats()
            player_stats = player_stats[player_stats['PLAYER_ID'] == player_id]
            recent_games = get_player_games(player_id, last_n_games=10)
            
            # Convert stats to text for the LLM
            stats_text = ""
            if not player_stats.empty:
                row = player_stats.iloc[0]
                stats_text = f"Season Stats: {row.get('PTS', 0)} PPG, {row.get('REB', 0)} RPG, "
                stats_text += f"{row.get('AST', 0)} APG, {row.get('STL', 0)} SPG, {row.get('BLK', 0)} BPG, "
                stats_text += f"{row.get('TOV', 0)} TOPG\n"
                stats_text += f"Shooting: {row.get('FG_PCT', 0)*100:.1f}% FG, {row.get('FT_PCT', 0)*100:.1f}% FT, "
                stats_text += f"{row.get('FG3_PCT', 0)*100:.1f}% 3PT\n"
                stats_text += f"Minutes: {row.get('MIN', 0)} MPG, Games: {row.get('GP', 0)}\n"
            
            # Format recent games
            games_text = ""
            if not recent_games.empty:
                games_text = "Recent Games:\n"
                for _, game in recent_games.iterrows():
                    games_text += f"- {game.get('GAME_DATE', '')}: {game.get('PTS', 0)} pts, "
                    games_text += f"{game.get('REB', 0)} reb, {game.get('AST', 0)} ast, "
                    games_text += f"{game.get('STL', 0)} stl, {game.get('BLK', 0)} blk\n"
            
            # Create the prompt template
            template = """
            You are an expert NBA Fantasy Basketball analyst. Analyze the following player for fantasy basketball purposes.
            
            Player: {player_name}
            
            {stats_text}
            
            {games_text}
            
            Provide an analysis of this player's fantasy value, including:
            1. A summary of their overall value
            2. Their key strengths for fantasy
            3. Their key weaknesses for fantasy
            4. Advice for drafting or trading for this player
            5. 3-5 comparable players for comparison
            
            Format your response as a JSON object matching this structure:
            {format_instructions}
            """
            
            # Set up the output parser
            parser = PydanticOutputParser(pydantic_object=PlayerAnalysis)
            
            # Create the prompt
            prompt = PromptTemplate(
                template=template,
                input_variables=["player_name", "stats_text", "games_text"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            
            # Create and run the chain
            chain = LLMChain(llm=self.llm, prompt=prompt)
            response = chain.invoke({
                "player_name": player_name,
                "stats_text": stats_text,
                "games_text": games_text
            })
            
            # Parse the response
            try:
                result = parser.parse(response['text'])
                return result.dict()
            except Exception as e:
                # Fallback to returning raw text if parsing fails
                print(f"Error parsing LLM response: {e}")
                return {"summary": response['text'], "player_name": player_name}
        
        except Exception as e:
            print(f"Error in player analysis: {e}")
            return {"error": str(e)}
    
    def analyze_matchup(self, team1_players, team2_players, team1_name="Team 1", team2_name="Team 2"):
        """Analyze a fantasy basketball matchup with AI insights"""
        if not self.is_available():
            return "LLM not available. Please ensure Ollama is running with Llama 3.2."
        
        try:
            # Get stats for all players
            all_stats = get_player_stats()
            
            # Process team 1
            team1_data = []
            for player in team1_players:
                player_id = get_player_id(player)
                if player_id:
                    player_stats = all_stats[all_stats['PLAYER_ID'] == player_id]
                    if not player_stats.empty:
                        stats = player_stats.iloc[0]
                        team1_data.append({
                            "name": player,
                            "pts": stats.get('PTS', 0),
                            "reb": stats.get('REB', 0),
                            "ast": stats.get('AST', 0),
                            "stl": stats.get('STL', 0),
                            "blk": stats.get('BLK', 0),
                            "tov": stats.get('TOV', 0),
                            "fg_pct": stats.get('FG_PCT', 0),
                            "ft_pct": stats.get('FT_PCT', 0),
                            "fg3m": stats.get('FG3M', 0)
                        })
            
            # Process team 2
            team2_data = []
            for player in team2_players:
                player_id = get_player_id(player)
                if player_id:
                    player_stats = all_stats[all_stats['PLAYER_ID'] == player_id]
                    if not player_stats.empty:
                        stats = player_stats.iloc[0]
                        team2_data.append({
                            "name": player,
                            "pts": stats.get('PTS', 0),
                            "reb": stats.get('REB', 0),
                            "ast": stats.get('AST', 0),
                            "stl": stats.get('STL', 0),
                            "blk": stats.get('BLK', 0),
                            "tov": stats.get('TOV', 0),
                            "fg_pct": stats.get('FG_PCT', 0),
                            "ft_pct": stats.get('FT_PCT', 0),
                            "fg3m": stats.get('FG3M', 0)
                        })
            
            # Calculate team totals
            team1_totals = {
                "pts": sum(p["pts"] for p in team1_data),
                "reb": sum(p["reb"] for p in team1_data),
                "ast": sum(p["ast"] for p in team1_data),
                "stl": sum(p["stl"] for p in team1_data),
                "blk": sum(p["blk"] for p in team1_data),
                "tov": sum(p["tov"] for p in team1_data),
                "fg3m": sum(p["fg3m"] for p in team1_data),
                "fg_pct": sum(p["fg_pct"] * 1 for p in team1_data) / len(team1_data) if team1_data else 0,
                "ft_pct": sum(p["ft_pct"] * 1 for p in team1_data) / len(team1_data) if team1_data else 0
            }
            
            team2_totals = {
                "pts": sum(p["pts"] for p in team2_data),
                "reb": sum(p["reb"] for p in team2_data),
                "ast": sum(p["ast"] for p in team2_data),
                "stl": sum(p["stl"] for p in team2_data),
                "blk": sum(p["blk"] for p in team2_data),
                "tov": sum(p["tov"] for p in team2_data),
                "fg3m": sum(p["fg3m"] for p in team2_data),
                "fg_pct": sum(p["fg_pct"] * 1 for p in team2_data) / len(team2_data) if team2_data else 0,
                "ft_pct": sum(p["ft_pct"] * 1 for p in team2_data) / len(team2_data) if team2_data else 0
            }
            
            # Create the prompt
            template = """
            You are an expert NBA Fantasy Basketball analyst. Analyze the following matchup between two fantasy teams.
            
            Team 1 ({team1_name}): {team1_players}
            Team 1 Stats:
            {team1_stats}
            
            Team 2 ({team2_name}): {team2_players}
            Team 2 Stats:
            {team2_stats}
            
            Analyze this matchup in detail, including:
            1. Which team is likely to win each category (PTS, REB, AST, STL, BLK, TOV, 3PM, FG%, FT%)
            2. The overall strengths and weaknesses of each team
            3. Key players who could swing the matchup
            4. Strategic advice for both team managers
            5. Your prediction for the final outcome with a percentage confidence
            
            Be specific, analytical, and provide insights that would be valuable to fantasy managers.
            """
            
            # Format the data for the prompt
            team1_stats_formatted = "\n".join([
                f"Points: {team1_totals['pts']:.1f}",
                f"Rebounds: {team1_totals['reb']:.1f}",
                f"Assists: {team1_totals['ast']:.1f}",
                f"Steals: {team1_totals['stl']:.1f}",
                f"Blocks: {team1_totals['blk']:.1f}",
                f"Turnovers: {team1_totals['tov']:.1f}",
                f"3-Pointers: {team1_totals['fg3m']:.1f}",
                f"FG%: {team1_totals['fg_pct']*100:.1f}%",
                f"FT%: {team1_totals['ft_pct']*100:.1f}%"
            ])
            
            team2_stats_formatted = "\n".join([
                f"Points: {team2_totals['pts']:.1f}",
                f"Rebounds: {team2_totals['reb']:.1f}",
                f"Assists: {team2_totals['ast']:.1f}",
                f"Steals: {team2_totals['stl']:.1f}",
                f"Blocks: {team2_totals['blk']:.1f}",
                f"Turnovers: {team2_totals['tov']:.1f}",
                f"3-Pointers: {team2_totals['fg3m']:.1f}",
                f"FG%: {team2_totals['fg_pct']*100:.1f}%",
                f"FT%: {team2_totals['ft_pct']*100:.1f}%"
            ])
            
            # Create the prompt
            prompt = PromptTemplate(
                template=template,
                input_variables=[
                    "team1_name", "team1_players", "team1_stats",
                    "team2_name", "team2_players", "team2_stats"
                ]
            )
            
            # Create and run the chain
            chain = LLMChain(llm=self.llm, prompt=prompt)
            response = chain.invoke({
                "team1_name": team1_name,
                "team1_players": ", ".join(team1_players),
                "team1_stats": team1_stats_formatted,
                "team2_name": team2_name,
                "team2_players": ", ".join(team2_players),
                "team2_stats": team2_stats_formatted
            })
            
            return response['text']
        
        except Exception as e:
            print(f"Error in matchup analysis: {e}")
            return f"Error analyzing matchup: {str(e)}"
    
    def answer_fantasy_question(self, question):
        """Answer a natural language question about fantasy basketball"""
        if not self.is_available():
            return "LLM not available. Please ensure Ollama is running with Llama 3.2."
        
        try:
            # If we have a vector store, use it for retrieval
            context = ""
            if self.vector_store is not None:
                # Perform retrieval based on the question
                retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": 5}
                )
                docs = retriever.get_relevant_documents(question)
                if docs:
                    context = "Information from NBA database:\n" + "\n\n".join([doc.page_content for doc in docs])
            
            # Create the prompt
            template = """
            You are an expert NBA Fantasy Basketball analyst and advisor. 
            Answer the following question about fantasy basketball based on your knowledge and the provided context.
            
            {context}
            
            Question: {question}
            
            Provide a detailed, accurate, and helpful answer. If you're unsure or don't have enough information, 
            acknowledge this and suggest what additional information would be helpful.
            """
            
            # Create the prompt
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            # Create and run the chain
            chain = LLMChain(llm=self.llm, prompt=prompt)
            response = chain.invoke({
                "context": context,
                "question": question
            })
            
            return response['text']
        
        except Exception as e:
            print(f"Error answering question: {e}")
            return f"Error: {str(e)}" 