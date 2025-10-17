#!/bin/bash
# Helper script to activate virtual environment

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating one..."
    uv venv
    uv sync
fi

# Source the activation script
source .venv/bin/activate

echo "✓ Virtual environment activated"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo ""
echo "To deactivate, run: deactivate"
