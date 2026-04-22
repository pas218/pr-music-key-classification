#!/bin/bash

# Define environment name
VENV_PATH=".venv"

# 1. Create virtual environment if it doesn't exist
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment..."
    python3.11 -m venv "$VENV_PATH"
fi

# 2. Activate the virtual environment
# NOTE: You MUST use 'source' or '.' to apply changes to the current shell
source "$VENV_PATH/bin/activate"

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi
