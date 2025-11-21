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
    # Production settings optimized for t2.micro (1GB RAM) with aggressive memory management:
    # -w 1: 1 worker process (reduces memory usage, prevents OOM kills)
    # --timeout 300: 5 minute timeout for large video processing
    # --max-requests 5: Very frequent restarts to prevent memory leaks (critical for 1GB RAM)
    # --max-requests-jitter 2: Randomize restart to avoid all workers restarting at once
    # --worker-class sync: Use sync workers (better for CPU-intensive tasks, lower memory)
    # --worker-connections 100: Reduced connections for memory-constrained environments
    # --limit-request-line 8190: Increase max request line size
    # --limit-request-fields 32768: Increase max request fields
    # --graceful-timeout 30: Time to wait for workers to finish before killing
    # Note: For t2.micro, PM2 max_memory_restart is set to 400M to leave room for OS
    # For larger instances (2GB+), increase --max-requests to 10-20 and max_memory_restart to 1G
    gunicorn -w 1 \
        -b 0.0.0.0:3001 \
        --timeout 300 \
        --graceful-timeout 30 \
        --max-requests 5 \
        --max-requests-jitter 2 \
        --worker-class sync \
        --worker-connections 100 \
        --limit-request-line 8190 \
        --limit-request-fields 32768 \
        --log-level warning \
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
        --max-requests 5 \
        --max-requests-jitter 2 \
        --worker-class sync \
        --worker-connections 100 \
        --limit-request-line 8190 \
        --limit-request-fields 32768 \
        --log-level warning \
        --access-logfile - \
        --error-logfile - \
        app:app
fi
