#!/usr/bin/env python3
"""
Screenshot Capture Tool for Fantasy Basketball Tools
Creates documentation images and demo materials
"""

import os
import time
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import shutil

def create_directory(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def run_app_in_background():
    """Start the fantasy basketball tools app in the background"""
    print("Starting Fantasy Basketball Tools application...")
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(script_dir, "fantasy-tools.py")
    
    # Run the application in the background
    process = subprocess.Popen(
        ["python", app_path],
        cwd=script_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for the app to start (adjust as needed)
    time.sleep(5)
    return process

def capture_screenshot(url, output_path, filename, wait_time=2):
    """Capture a screenshot of the specified URL"""
    try:
        # Try to import the required libraries
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
    except ImportError:
        print("Error: Required libraries not found. Please install them with:")
        print("uv pip install selenium webdriver-manager")
        return False

    try:
        from webdriver_manager.chrome import ChromeDriverManager
        
        # Set up Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--window-size=1920,1080")  # Set window size
        
        # Set up the Chrome driver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Navigate to the URL
        driver.get(url)
        
        # Wait for the page to load
        time.sleep(wait_time)
        
        # Create the full output path
        full_path = os.path.join(output_path, filename)
        
        # Take a screenshot
        driver.save_screenshot(full_path)
        
        print(f"Screenshot saved to: {full_path}")
        driver.quit()
        return True
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Capture screenshots of Fantasy Basketball Tools")
    parser.add_argument("--output", default="screenshots", help="Output directory for screenshots")
    parser.add_argument("--base-url", default="http://localhost:7860", help="Base URL of the application")
    args = parser.parse_args()
    
    # Create screenshots directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = create_directory(f"{args.output}/{timestamp}")
    
    print(f"Screenshots will be saved to: {output_dir}")
    
    # Start the application
    app_process = run_app_in_background()
    
    try:
        # Try to install required packages if needed
        try:
            from selenium import webdriver
        except ImportError:
            print("Installing required packages...")
            # Check if uv is available
            uv_path = shutil.which("uv")
            if uv_path:
                subprocess.run(["uv", "pip", "install", "selenium", "webdriver-manager"])
            else:
                print("UV not found. Falling back to pip...")
                subprocess.run(["pip", "install", "selenium", "webdriver-manager"])
        
        # Define the pages to capture
        pages = [
            {"url": f"{args.base_url}", "filename": "main_interface.png", "wait": 4},
            {"url": f"{args.base_url}/?__theme=dark", "filename": "main_dark.png", "wait": 3},
            # Add tabs with #tab= suffix
            {"url": f"{args.base_url}/#tab=1", "filename": "draft_helper.png", "wait": 5},
            {"url": f"{args.base_url}/#tab=2", "filename": "matchup_analyzer.png", "wait": 5},
            {"url": f"{args.base_url}/#tab=3", "filename": "consistency_tracker.png", "wait": 5},
            {"url": f"{args.base_url}/#tab=4", "filename": "game_simulator.png", "wait": 5}
        ]
        
        # Capture screenshots
        for page in pages:
            capture_screenshot(
                page["url"], 
                output_dir, 
                page["filename"],
                wait_time=page.get("wait", 3)
            )
            
        print(f"All screenshots captured in: {output_dir}")
        print("You can use these images in your documentation or presentations.")
        
    finally:
        # Clean up: Terminate the app process
        if app_process:
            app_process.terminate()
            print("Application terminated.")

if __name__ == "__main__":
    main() 