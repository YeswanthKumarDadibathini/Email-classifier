import os
import sys
import subprocess
import time

def check_file_exists(file_path):
    return os.path.isfile(file_path)

def run_command(command, description):
    print(f"\n{'=' * 50}")
    print(f"  {description}")
    print(f"{'=' * 50}\n")
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    if process.returncode != 0:
        print("\nERROR: Command failed with the following error:")
        for line in process.stderr:
            print(line, end='')
        return False
    
    return True

def main():
    print("\n📧 Email Classifier Setup and Run Tool 📧\n")
    
    # Check if model and tokenizer already exist
    model_exists = check_file_exists('email_classifier_model.h5')
    tokenizer_exists = check_file_exists('tokenizer.pickle')
    
    # Step 1: Install dependencies if requirements.txt exists
    if check_file_exists('requirements.txt'):
        if run_command('pip install -r requirements.txt', "Installing dependencies"):
            print("✅ Dependencies installed successfully")
        else:
            print("❌ Failed to install dependencies")
            return
    
    # Step 2: Run preprocessing if tokenizer doesn't exist
    if not tokenizer_exists:
        if run_command('python preprocess.py', "Running data preprocessing"):
            print("✅ Data preprocessing completed")
        else:
            print("❌ Data preprocessing failed")
            return
    else:
        print("✅ Tokenizer already exists, skipping preprocessing")
    
    # Step 3: Train model if it doesn't exist
    if not model_exists:
        if run_command('python train_model.py', "Training neural network model"):
            print("✅ Model training completed")
        else:
            print("❌ Model training failed")
            return
    else:
        print("✅ Model already exists, skipping training")
    
    # Step 4: Run the Flask application
    if run_command('python app.py', "Starting Flask server"):
        print("✅ Server started successfully")
    else:
        print("❌ Failed to start server")
        return

if __name__ == "__main__":
    main() 