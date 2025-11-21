#!/bin/bash
cd /srv/siha/ai-service-python   # <-- set your project path

echo "Activating venv…"
source venv/bin/activate

echo "Starting Python app…"
exec python app.py
