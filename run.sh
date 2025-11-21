#!/bin/bash
# Startup script for Python AI Service

echo "Starting YourCare AI Service (Python)..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if dependencies are installed
if ! python -c "import flask" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the service
python app.py

