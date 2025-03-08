# Fantasy Basketball Tools

A suite of interactive tools for fantasy basketball managers built using the NBA API.

## Features

- **Draft Helper**: Find undervalued players to target in your draft based on statistical analysis
- **Matchup Analyzer**: Compare two fantasy teams to predict head-to-head category winners
- **Consistency Tracker**: Analyze player reliability to identify consistent starters vs. boom/bust players

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
uv pip install nba_api pandas numpy matplotlib seaborn gradio

# Run the application
uv run fantasy-tools.py
```

Visit http://localhost:7860 in your browser to use the tools.

## Documentation

- [Installation Guide](./docs/INSTALLATION.md) - Detailed setup instructions
- [Usage Guide](./docs/USAGE.md) - How to use each fantasy basketball tool
- [Troubleshooting](./docs/TROUBLESHOOTING.md) - Common issues and solutions
- [Contributing](./docs/CONTRIBUTING.md) - Guidelines for contributors

## License

This project is part of [loftwah/langchain-csv](https://github.com/loftwah/langchain-csv) and is available under its license terms.
