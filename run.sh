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

# Check if gunicorn is available
if command -v gunicorn &> /dev/null; then
    echo "Starting with gunicorn (production mode)..."
    # Production settings for large payloads:
    # -w 2: 2 worker processes (adjust based on CPU cores)
    # --timeout 300: 5 minute timeout for large video processing
    # --max-requests 1000: Restart workers after 1000 requests to prevent memory leaks
    # --max-requests-jitter 100: Randomize restart to avoid all workers restarting at once
    # --worker-class sync: Use sync workers (better for CPU-intensive tasks)
    # --worker-connections 1000: Max connections per worker
    gunicorn -w 2 \
        -b 0.0.0.0:3001 \
        --timeout 300 \
        --max-requests 1000 \
        --max-requests-jitter 100 \
        --worker-class sync \
        --worker-connections 1000 \
        --log-level info \
        app:app
else
    echo "Gunicorn not found. Installing..."
    pip install gunicorn
    
    echo "Starting with gunicorn..."
    gunicorn -w 2 \
        -b 0.0.0.0:3001 \
        --timeout 300 \
        --max-requests 1000 \
        --max-requests-jitter 100 \
        --worker-class sync \
        app:app
fi

