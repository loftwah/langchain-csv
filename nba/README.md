# Fantasy Basketball Tools

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![NBA API](https://img.shields.io/badge/NBA_API-1.3.1-red.svg)](https://github.com/swar/nba_api)
[![Gradio](https://img.shields.io/badge/gradio-4.12.0-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

A suite of interactive tools for fantasy basketball managers built using the NBA API.

![Fantasy Basketball Tools](https://via.placeholder.com/800x400?text=Fantasy+Basketball+Tools)

> **Demo-Ready**: This application includes an offline mode, animated startup, and professional UI for smooth presentations.

## Key Features

- **Draft Helper**: Find undervalued players to target in your draft based on statistical analysis
  - Automatically identifies players outperforming their draft position
  - Visualizes value metrics to spot hidden gems
  - Customizable by scoring system and statistical categories

- **Matchup Analyzer**: Compare two fantasy teams to predict head-to-head category winners
  - Head-to-head category prediction with confidence scores
  - Visual breakdown of team strengths and weaknesses
  - Preset team matchups for quick demonstrations

- **Consistency Tracker**: Analyze player reliability to identify consistent starters vs. boom/bust players
  - Game-by-game performance visualization
  - Consistency metrics for key fantasy statistics
  - Detect trends in player performance

- **Game Simulator**: Simulate fantasy matchups to project outcomes 
  - Quarter-by-quarter fantasy game simulation
  - Based on real player performance data
  - Visualize game flow and key statistics

## Requirements

- Python 3.9 or higher
- Internet connection for NBA API data (or pre-cached data for offline mode)
- 4GB RAM recommended
- Modern web browser
- Dependencies listed in `requirements.txt`

## Quick Start

```bash
# Clone repository
git clone https://github.com/loftwah/langchain-csv.git
cd langchain-csv/nba

# Setup with uv
uv venv
source .venv/bin/activate  # On Unix/macOS
# OR
.venv\Scripts\activate     # On Windows

# Install dependencies
uv pip install -r requirements.txt

# Run the application
python fantasy-tools.py
```

Visit http://localhost:7860 in your browser to use the tools.

## Demo Preparation

If you're preparing for a demo:

1. Run the application ahead of time to cache NBA data
2. Consult the [Demo Guide](./docs/DEMO.md) for a step-by-step presentation plan
3. For offline demos, set the environment variable:
   ```bash
   export NBA_OFFLINE_MODE=1  # Unix/macOS
   # OR
   set NBA_OFFLINE_MODE=1     # Windows
   ```
4. Generate screenshots for your presentation:
   ```bash
   # Install screenshot requirements
   uv pip install selenium webdriver-manager
   
   # Capture screenshots of all tools
   python tools/capture_screenshots.py
   ```
   This will automatically start the app, capture all screens, and save them to a timestamped folder.

### Offline Mode Features

The application now includes full offline support:
- Automatically caches NBA data after first run
- Falls back to cached data when NBA API is unavailable
- Environment variable toggle for guaranteed offline operation
- Transparent to end-users with appropriate fallback messages

## Documentation

- [Installation Guide](./docs/INSTALLATION.md) - Detailed setup instructions
- [Usage Guide](./docs/USAGE.md) - How to use each fantasy basketball tool
- [Demo Guide](./docs/DEMO.md) - Step-by-step guide for demonstrations
- [Troubleshooting](./docs/TROUBLESHOOTING.md) - Common issues and solutions
- [Contributing](./docs/CONTRIBUTING.md) - Guidelines for contributors

## Technical Architecture

The application is built using:
- NBA API for official basketball statistics
- Pandas and NumPy for data analysis
- Matplotlib and Seaborn for data visualization
- Gradio for the interactive web interface

## Code Structure

- `fantasy-tools.py` - Main entry point with animated startup
- `src/api.py` - NBA API interaction with caching
- `src/api_cache.py` - Enhanced API caching with decorator pattern
- `src/ui.py` - Gradio UI components
- `src/tools.py` - Core analysis tools wrappers
- `src/game.py` - Game simulation engine with object-oriented design
- `src/tracker.py` - Consistency tracking with object-oriented design
- `src/config.py` - Configuration and styling with centralized parameters
- `src/fantasy.py` - Fantasy basketball calculations
- `tools/` - Utility scripts for documentation and demos

### Architectural Improvements

The codebase follows these design principles:
- **Modularity**: Components are broken down into smaller, focused modules
- **Object-Oriented Design**: Classes represent core domain concepts like Players and Teams
- **Separation of Concerns**: UI, business logic, and data access are separated
- **Configuration Management**: Centralized configuration with sensible defaults
- **Consistent Error Handling**: Robust error handling and graceful fallbacks
- **Clean API**: Well-documented interfaces between components

#### Object Model

The simulator uses a clean object model with classes including:
- `Player`: Represents NBA players with stats and game performance
- `Team`: Collection of players with team-level statistics
- `GameState`: Tracks the state of an ongoing simulation
- `PlayGenerator`: Generates realistic basketball plays
- `GameRenderer`: Formats output in different presentations
- `GameSimulator`: Coordinates the simulation process

The consistency tracker uses:
- `ConsistencyTracker`: Analyzes player fantasy performance patterns

## Roadmap

Planned features for future releases:
- Fantasy platform integrations (ESPN, Yahoo, etc.)
- Live draft assistant mode
- Player injury impact analysis
- Trade value calculator
- Custom league settings support

## Contributing

We welcome contributions! Please see our [Contributing Guide](./docs/CONTRIBUTING.md) for details.

Some ways to contribute:
- Add new fantasy basketball tools
- Enhance existing statistical models
- Improve UI/UX design
- Fix bugs and issues
- Extend documentation
- Add tests

## License

This project is part of [loftwah/langchain-csv](https://github.com/loftwah/langchain-csv) and is available under its license terms.
