cd /home/serverdeploy/projects/scoutjar/scoutjar-ai
pm2 delete scoutjar-ai || true
pm2 start venv/bin/python3 --name scoutjar-ai -- app.py
pm2 save