#!/usr/bin/env python3
"""
Simple wrapper to run the Streamlit app with the correct environment.
"""
import subprocess
import sys
import os

def main():
    # Activate virtual environment and run Streamlit
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Windows version (adjust for your OS)
    venv_path = os.path.join(project_root, "venv")  # Adjust based on your venv name
    
    # Construct the command
    streamlit_cmd = [
        sys.executable,  # Use current Python interpreter
        "-m", "streamlit", "run",
        os.path.join(script_dir, "app.py"),
        "--server.port=8501"
    ]
    
    print(f"Starting Streamlit app on http://localhost:8501")
    subprocess.run(streamlit_cmd)

if __name__ == "__main__":
    main()
