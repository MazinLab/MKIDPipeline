
# Run script for mkidpipeline
# Execute in command line with: python3 run_mkidpipeline.py
import subprocess
def run_mkidpipe():
    command = ['mkidpipe', '--make-dir', '--make-outputs']
    try:
        # Run the command
        subprocess.run(command, check=True)
        print("Command executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the command: {e}")
if __name__ == '__main__':
    run_mkidpipe()
