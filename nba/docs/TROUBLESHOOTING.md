# Troubleshooting Guide

This document provides solutions to common issues you might encounter when using the Fantasy Basketball Tools.

## Connection Issues

### NBA API Connection Errors

**Symptoms**: 
- Error messages about connection timeouts
- Warnings about failed API requests
- Empty results when querying player data

**Solutions**:
1. **Enable Offline Mode**:
   ```bash
   export NBA_OFFLINE_MODE=1  # Unix/macOS
   # OR
   set NBA_OFFLINE_MODE=1     # Windows
   ```
   
2. **Check Network Connection**:
   - Verify you have a stable internet connection
   - NBA API can be rate-limited, so try again in a few minutes
   
3. **Use Cached Data**:
   - The application automatically caches responses
   - Run queries once before your demo to populate the cache
   - Look for data in the `nba_api_cache` directory

### Gradio Interface Not Loading

**Symptoms**:
- The terminal shows the application is running but no browser opens
- You see an error message about port conflicts

**Solutions**:
1. **Open Browser Manually**:
   - Navigate to http://localhost:7860 in your browser
   
2. **Change Ports**:
   - Edit `fantasy-tools.py` and modify the launch call:
   ```python
   demo.launch(share=False, server_port=7861)  # Try a different port
   ```

## Data Issues

### Missing Player Data

**Symptoms**:
- Player searches return no results
- Analytics tools show incomplete data

**Solutions**:
1. **Check Player Name Spelling**:
   - Ensure correct spelling, including apostrophes (e.g., "D'Angelo Russell")
   
2. **Run in Debug Mode**:
   - Set the environment variable `NBA_DEBUG=1` before running the application
   - Check console output for specific API errors

3. **Update Player Cache**:
   - Delete the `nba_api_cache/all_players.csv` file to force a refresh
   - Run the application again to rebuild the cache

### Visualization Problems

**Symptoms**:
- Charts not displaying properly
- Error messages related to matplotlib or seaborn

**Solutions**:
1. **Check Dependencies**:
   ```bash
   pip install -r requirements.txt --upgrade
   ```
   
2. **Reset Matplotlib Cache**:
   - Delete `~/.matplotlib/fontList.cache` (location may vary by OS)
   
3. **Adjust Display Settings**:
   - For high-DPI displays, add to the beginning of your script:
   ```python
   import matplotlib as mpl
   mpl.rcParams['figure.dpi'] = 100  # Adjust as needed
   ```

## Performance Issues

### Slow Application Startup

**Symptoms**:
- Application takes more than 30 seconds to start
- Console shows long waits for NBA API responses

**Solutions**:
1. **Pre-cache Data**:
   - Run the application once before your demo
   - Use the offline mode for actual presentations
   
2. **Limit Data Requests**:
   - Focus on specific players rather than league-wide queries
   - Use preset team matchups which load faster
   
3. **Optimize Dependencies**:
   ```bash
   pip install orjson  # Faster JSON processing
   pip install polars  # Alternative to pandas for faster data processing
   ```

### Memory Usage Too High

**Symptoms**:
- Application crashes with memory errors
- System becomes unresponsive when running the tool

**Solutions**:
1. **Limit Data Loading**:
   - Edit `src/config.py` to reduce `DEFAULT_NUM_GAMES_ANALYSIS`
   
2. **Clear Cache Between Sessions**:
   - Delete large files in the `nba_api_cache` directory
   
3. **Run with Memory Profiling**:
   ```bash
   python -m memory_profiler fantasy-tools.py
   ```

## Miscellaneous Issues

### Tool Crashes During Demo

**Solutions**:
1. **Have Backup Demo Content**:
   - Prepare screenshots of the tool in action
   - Have analysis results saved and ready to present
   
2. **Run in Simpler Mode**:
   - Edit `src/config.py` to disable advanced features

3. **Check Gradio Version**:
   ```bash
   pip install gradio==4.12.0  # Use a specific, stable version
   ```

### File Not Found Errors

**Solutions**:
1. **Check Working Directory**:
   ```python
   import os
   print(os.getcwd())  # Ensure you're in the right directory
   ```
   
2. **Use Absolute Paths**:
   - Edit scripts to use absolute paths for data files

## Getting Help

If you encounter issues not covered in this guide:

1. Check the [NBA API Documentation](https://github.com/swar/nba_api)
2. Examine terminal output for specific error messages
3. Submit an issue on the GitHub repository with steps to reproduce 