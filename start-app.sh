#!/bin/bash

# Go to the scoutjar-ai project
cd ~/projects/scoutjar/scoutjar-ai

# Activate the Python virtual environment
source venv/bin/activate

# Pull latest code
git fetch origin
git reset --hard origin/mvp0.1

# Show current branch and commit
echo "🛠 Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "🔖 Commit: $(git rev-parse --short HEAD)"

# Install Python dependencies if needed
pip install -r requirements.txt

# Kill anything using port 5001
fuser -k 5001/tcp || true
# Stop any existing scoutjar-ai process
pm2 delete scoutjar-ai || true

# Start the Flask app with pm2
pm2 start "python3 app.py" --name "scoutjar-ai-mvp0.1"

# Save the pm2 process list
pm2 save
