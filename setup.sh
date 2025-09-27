#!/bin/bash
# Setup script for DistilGPT-2 LLM Server

echo "Setting up DistilGPT-2 LLM Server..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
echo "Python version: $python_version"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Running basic tests..."
python test_basic.py

echo ""
echo "Setup complete! To start the server:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Start the server: python server.py"
echo "3. Open http://localhost:8082/docs in your browser"