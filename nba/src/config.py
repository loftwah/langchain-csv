import os

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
    margin: 10px 5px !important;
    background-color: #1e1e1e !important;
    border: 2px solid #444 !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 15px !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
    font-size: 16px !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    min-width: 200px !important;
    text-align: center !important;
    cursor: pointer !important;
}

.preset-button:hover {
    background-color: #17408B !important;
    border-color: #F7B801 !important;
    transform: translateY(-3px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    color: #F7B801 !important;
}

.preset-button:active {
    transform: translateY(-1px);
    box-shadow: 0 3px 6px rgba(0,0,0,0.2);
}
""" 