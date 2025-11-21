#!/bin/bash
set -e
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
    # Production settings for large payloads with memory optimization:
    # -w 1: 1 worker process (reduces memory usage, prevents OOM kills)
    # --timeout 300: 5 minute timeout for large video processing
    # --max-requests 20: Restart workers more frequently to prevent memory leaks (reduced from 30)
    # --max-requests-jitter 3: Randomize restart to avoid all workers restarting at once
    # --worker-class sync: Use sync workers (better for CPU-intensive tasks)
    # --worker-connections 1000: Max connections per worker
    # --limit-request-line 8190: Increase max request line size
    # --limit-request-fields 32768: Increase max request fields
    # --graceful-timeout 30: Time to wait for workers to finish before killing
    # --preload: Preload app to share memory between workers (not used with -w 1, but good practice)
    # Note: Memory limits should be set via PM2 max_memory_restart
    gunicorn -w 1 \
        -b 0.0.0.0:3001 \
        --timeout 300 \
        --graceful-timeout 30 \
        --max-requests 20 \
        --max-requests-jitter 3 \
        --worker-class sync \
        --worker-connections 1000 \
        --limit-request-line 8190 \
        --limit-request-fields 32768 \
        --log-level info \
        --access-logfile - \
        --error-logfile - \
        app:app
else
    echo "Gunicorn not found. Installing..."
    pip install gunicorn
    
    echo "Starting with gunicorn..."
    gunicorn -w 1 \
        -b 0.0.0.0:3001 \
        --timeout 300 \
        --graceful-timeout 30 \
        --max-requests 20 \
        --max-requests-jitter 3 \
        --worker-class sync \
        --limit-request-line 8190 \
        --limit-request-fields 32768 \
        --log-level info \
        --access-logfile - \
        --error-logfile - \
        app:app
fi
