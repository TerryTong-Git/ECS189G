import sys
from pathlib import Path

# Add the directory containing the 'script' package to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import and execute the desired module
import script.stage_2_script.script_mlp

if __name__ == "__main__":
    script.stage_2_script.script_mlp