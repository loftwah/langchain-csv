# Fantasy Basketball Tools

A suite of interactive tools for fantasy basketball managers built using the NBA API.

## Features

- **Draft Helper**: Find undervalued players to target in your draft based on statistical analysis
- **Matchup Analyzer**: Compare two fantasy teams to predict head-to-head category winners
- **Consistency Tracker**: Analyze player reliability to identify consistent starters vs. boom/bust players

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and virtual environments. Make sure you have uv installed before proceeding.

### Setup

1. Clone the repository (if you haven't already):

```bash
git clone https://github.com/loftwah/langchain-csv.git
cd langchain-csv/nba
```

2. Create and activate a virtual environment with uv:

```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# OR
.venv\Scripts\activate     # On Windows
```

3. Install dependencies:

```bash
uv pip install nba_api pandas numpy matplotlib seaborn gradio
```

## Usage

Run the Fantasy Basketball Tools:

```bash
uv run fantasy-tools.py
```

This will start a Gradio web interface accessible at http://localhost:7860 in your browser.

### Draft Helper

1. Select your league's scoring system (standard, points, or categories)
2. Set minimum games played requirement
3. Choose a statistical category to emphasize
4. Click "Find Value Players" to get a ranked list of draft targets

### Matchup Analyzer

1. Enter your team's players (comma-separated)
2. Enter your opponent's players (comma-separated)
3. Select your league's scoring system
4. Click "Compare Teams" to see projected category winners and matchup outcome

### Consistency Tracker

1. Enter a player's name
2. Select number of recent games to analyze
3. Select your league's scoring system
4. Click "Analyze Player" to see consistency metrics and performance trend

## Technical Details

- Uses the `nba_api` package to fetch data from official NBA.com endpoints
- Implements caching to reduce API calls and improve performance
- Visualizes results using matplotlib and seaborn
- Provides an interactive interface via Gradio

## Troubleshooting

If you encounter issues with the NBA API:

1. Check your internet connection
2. The NBA API occasionally gets rate limited - wait and try again later
3. Make sure player names are spelled correctly (or use partial names)
4. For persistent errors, check [nba_api issues](https://github.com/swar/nba_api/issues) for known problems

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is part of [loftwah/langchain-csv](https://github.com/loftwah/langchain-csv) and is available under its license terms.
