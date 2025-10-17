#!/bin/bash
# Test runner script - always activates venv before running tests

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Activating virtual environment...${NC}"

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating one..."
    uv venv
fi

# Activate virtual environment
source .venv/bin/activate

echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo -e "${BLUE}Running tests...${NC}\n"

# Run pytest with all arguments passed to this script
python -m pytest "$@"

# Deactivate virtual environment
deactivate
