#!/bin/bash

# Go to the scoutjar-ai project
cd ~/projects/scoutjar/scoutjar-ai

# Activate the Python virtual environment
source venv/bin/activate

# Pull latest code
git fetch origin
# git reset --hard origin/mvp0.1

# Show current branch and commit
echo "🛠 Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "🔖 Commit: $(git rev-parse --short HEAD)"

# Install Python dependencies if needed
pip install -r requirements.txt

# Stop any existing scoutjar-ai process
pm2 delete scoutjar-ai-mvp0.1 || true

# Kill any manual python3 process or leftovers on port 5001
echo "🔪 Killing any process using port 5001..."
kill -9 $(lsof -t -i :5001) || true



# Start the Flask app with pm2
# pm2 start "python3 app.py" --name "scoutjar-ai-mvp0.1" --no-autorestart --time
pm2 start "gunicorn -b 0.0.0.0:5001 app:app --timeout 180 --workers 2" --name "scoutjar-ai-mvp0.1" --time

# Save the pm2 process list
pm2 save
