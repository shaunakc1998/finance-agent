# run_training_background.py
# Script to run the model training in the background

import subprocess
import os
import sys
import time
from datetime import datetime

def run_training_in_background():
    """Run the train_all_models.py script in the background and log output to a file"""
    print("Starting model training in the background...")
    
    # Create logs directory if it doesn't exist
    logs_dir = "training_logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"training_run_{timestamp}.log")
    
    # Determine the Python executable to use
    python_executable = sys.executable
    
    # Command to run
    cmd = [python_executable, "train_all_models.py"]
    
    # Open log file for writing
    with open(log_file, "w") as f:
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        print(f"Training process started with PID: {process.pid}")
        print(f"Output is being logged to: {log_file}")
        print("You can continue using the chat agent while models are being trained.")
        print("To check training progress, use: tail -f " + log_file)

if __name__ == "__main__":
    run_training_in_background()
