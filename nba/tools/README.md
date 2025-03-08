# Fantasy Basketball Tools - Utilities

This directory contains utility scripts for the Fantasy Basketball Tools application.

## Screenshot Capture Tool

`capture_screenshots.py` - Automatically captures screenshots of the application for documentation and demo materials.

### Usage

```bash
# Install required dependencies
uv pip install selenium webdriver-manager

# Run the script
uv run tools/capture_screenshots.py

# Options
uv run tools/capture_screenshots.py --output custom_folder --base-url http://localhost:7860
```

### Features

- Automatically starts the Fantasy Basketball Tools application
- Captures screenshots of each tab/tool
- Organizes screenshots in a timestamped directory
- Supports custom output directories
- Works with different application URLs

### Requirements

- Python 3.9+
- Selenium
- Chrome or Chromium browser installed
- webdriver-manager

## Example Output

When you run the screenshot capture tool, it will create a directory structure like:

```
screenshots/
  ├── 20240308-153022/
  │   ├── main_interface.png
  │   ├── main_dark.png
  │   ├── draft_helper.png
  │   ├── matchup_analyzer.png
  │   ├── consistency_tracker.png
  │   └── game_simulator.png
```

You can use these images in your documentation, presentations, or marketing materials. 