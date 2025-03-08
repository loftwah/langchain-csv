import os
from pathlib import Path

# Create a cache directory
CACHE_DIR = "nba_api_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Color scheme - NBA inspired
NBA_COLORS = {
    'primary': '#17408B',    # NBA blue
    'secondary': '#C9082A',  # NBA red
    'accent': '#FFFFFF',     # White
    'background': '#121212', # Dark background
    'text': '#FFFFFF',       # White text
    'highlight': '#F7B801',  # Gold highlight
    'charts': ['#17408B', '#C9082A', '#F7B801', '#1D428A', '#CE1141', '#552583', '#006BB6', '#007A33', '#860038', '#006BB6']
}

# Scoring system defaults
DEFAULT_SCORING_SYSTEM = 'standard'
DEFAULT_MIN_GAMES = 20
DEFAULT_NUM_GAMES_ANALYSIS = 10

# Default season
DEFAULT_SEASON = "2023-24"

# Game simulator configuration
GAME_CONFIG = {
    'default_quarters': 4,
    'default_quarter_length': 12,
    'possessions_per_minute': 2.2,  # Average possessions per minute
    'shot_types': {
        'three_point': {
            'probability': 0.25,
            'points': 3,
            'default_fg_pct': 0.35
        },
        'mid_range': {
            'probability': 0.25,
            'points': 2,
            'default_fg_pct': 0.45
        },
        'layup': {
            'probability': 0.25,
            'points': 2,
            'default_fg_pct': 0.55
        },
        'dunk': {
            'probability': 0.1,
            'points': 2,
            'default_fg_pct': 0.85
        }
    },
    'play_types': {
        'shot': 0.7,
        'pass': 0.2,
        'turnover': 0.1
    },
    'rebound_chances': {
        'offensive': 0.25,
        'defensive': 0.75
    },
    'and_one_probability': 0.15,
    'foul_probability': {
        'three_point': 0.08,
        'mid_range': 0.08,
        'layup': 0.25,
        'dunk': 0.25
    }
}

# API cache configuration
CACHE_CONFIG = {
    'default_timeout': 7 * 24 * 60 * 60,  # 7 days in seconds
    'player_data_timeout': 7 * 24 * 60 * 60,  # 7 days
    'game_data_timeout': 24 * 60 * 60,  # 1 day
    'seasonal_data_timeout': 24 * 60 * 60,  # 1 day
    'historical_data_timeout': 365 * 24 * 60 * 60,  # 1 year
}

# Consistency tracker configuration
CONSISTENCY_CONFIG = {
    'min_data_points': 5,
    'consistency_thresholds': [
        (0, 20, 'Very Inconsistent', '#FF3030'),
        (20, 40, 'Inconsistent', '#FF8C00'),
        (40, 60, 'Moderate', '#FFFF00'),
        (60, 80, 'Consistent', '#7CFC00'),
        (80, 100, 'Very Consistent', '#00FF7F')
    ]
}

# Notable NBA players for enhanced commentary
NOTABLE_PLAYERS = [
    "Michael Jordan", "Kobe Bryant", "LeBron James", "Magic Johnson", 
    "Larry Bird", "Wilt Chamberlain", "Kareem Abdul-Jabbar", "Hakeem Olajuwon", 
    "Shaquille O'Neal", "Bill Russell", "Tim Duncan", "Julius Erving", 
    "Stephen Curry", "Kevin Durant", "Oscar Robertson", "Jerry West",
    "Allen Iverson", "Charles Barkley", "Dirk Nowitzki", "John Stockton",
    "Giannis Antetokounmpo", "Nikola Jokic", "Luka Doncic", "Joel Embiid"
]

# Preset configurations for various tools
PRESETS = {
    # Player presets for quick selection
    "players": {
        "superstars": ["LeBron James", "Kevin Durant", "Stephen Curry", "Giannis Antetokounmpo", "Nikola Jokić"],
        "guards": ["Stephen Curry", "Damian Lillard", "Kyrie Irving", "Devin Booker", "Donovan Mitchell"],
        "forwards": ["LeBron James", "Kevin Durant", "Kawhi Leonard", "Jayson Tatum", "Jimmy Butler"],
        "centers": ["Nikola Jokić", "Joel Embiid", "Anthony Davis", "Karl-Anthony Towns", "Bam Adebayo"]
    },
    
    # Team presets for simulations and matchups
    "teams": {
        "all_stars": "LeBron James, Kevin Durant, Stephen Curry, Giannis Antetokounmpo, Nikola Jokić",
        "young_guns": "Luka Dončić, Trae Young, Ja Morant, Zion Williamson, Anthony Edwards",
        "big_men": "Joel Embiid, Nikola Jokić, Anthony Davis, Karl-Anthony Towns, Bam Adebayo",
        "guards": "Stephen Curry, Damian Lillard, Kyrie Irving, Devin Booker, Donovan Mitchell",
        "legends": "Michael Jordan, Kobe Bryant, LeBron James, Magic Johnson, Larry Bird",
        "offense": "James Harden, Stephen Curry, Kevin Durant, Giannis Antetokounmpo, Joel Embiid",
        "defense": "Jrue Holiday, Marcus Smart, Kawhi Leonard, Draymond Green, Rudy Gobert",
        "forwards": "LeBron James, Kevin Durant, Kawhi Leonard, Jayson Tatum, Jimmy Butler"
    },
    
    # Draft settings presets
    "draft": {
        "standard_12_team": {"teams": 12, "scoring": "Standard Points", "position": 6},
        "standard_10_team": {"teams": 10, "scoring": "Standard Points", "position": 5},
        "categories_12_team": {"teams": 12, "scoring": "Standard Categories", "position": 6},
        "categories_10_team": {"teams": 10, "scoring": "Standard Categories", "position": 5}
    },
    
    # Consistency tracker presets - popular players to track
    "consistency": [
        "LeBron James",
        "Stephen Curry",
        "Nikola Jokić",
        "Giannis Antetokounmpo",
        "Luka Dončić"
    ]
}

# Custom CSS for Gradio interface
CUSTOM_CSS = """
body, .gradio-container {
    background-color: #121212;
    color: white;
}

.tabs {
    background-color: #1e1e1e;
    border-radius: 8px;
    margin-bottom: 20px;
}

.tab-nav {
    background-color: #1e1e1e;
    border-bottom: 2px solid #333;
}

.tab-nav button {
    color: white;
    background-color: transparent;
    margin: 0 5px;
    padding: 10px 15px;
    font-weight: bold;
}

.tab-nav button.selected {
    color: #F7B801;
    border-bottom: 3px solid #F7B801;
}

label {
    color: #ddd !important;
}

button.primary {
    background-color: #17408B !important;
    border: none !important;
    color: white !important;
    padding: 10px 20px !important;
    border-radius: 5px !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
}

button.primary:hover {
    background-color: #C9082A !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.footer {
    margin-top: 30px;
    text-align: center;
    color: #888;
    padding: 20px;
    border-top: 1px solid #333;
}

a {
    color: #F7B801 !important;
    text-decoration: none !important;
}

a:hover {
    text-decoration: underline !important;
}

.content-block {
    background-color: #1e1e1e;
    border-radius: 8px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

h1, h2, h3 {
    color: white;
}

input, select, textarea {
    background-color: #2d2d2d !important;
    color: white !important;
    border: 1px solid #444 !important;
}

.plot-container {
    background-color: #1e1e1e;
    border-radius: 8px;
    padding: 15px;
}

.full-width-plot {
    width: 100% !important;
    min-height: 500px !important;
}

.gradio-table {
    background-color: #1e1e1e;
}

.highlight {
    color: #F7B801;
    font-weight: bold;
}

.section-header {
    margin-top: 20px !important;
    margin-bottom: 10px !important;
    color: #F7B801 !important;
    border-bottom: 1px solid #333;
    padding-bottom: 8px;
    font-size: 22px !important;
    font-weight: bold !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.preset-button {
    background-color: #f0f0f0;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 6px 12px;
    margin: 4px;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.2s ease;
    color: #333;
}

.preset-button:hover {
    background-color: #e0e0e0;
    border-color: #ccc;
    transform: translateY(-2px);
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

.preset-button:active {
    background-color: #d0d0d0;
    transform: translateY(0);
    box-shadow: none;
}

/* Preset section styling */
.preset-section {
    background-color: #f9f9f9;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 15px;
    border: 1px solid #eee;
}

.preset-section h4 {
    margin-top: 0;
    margin-bottom: 8px;
    font-size: 14px;
    color: #555;
}

.preset-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 8px;
}
""" 