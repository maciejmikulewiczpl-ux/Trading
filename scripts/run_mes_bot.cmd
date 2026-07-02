@echo off
REM Surface scheduled-task wrapper for the MES momentum paper bot. Fired every 5 min on weekdays
REM during a local window that covers US RTH (bot self-guards to ET market hours, so off-hours
REM passes exit instantly). Needs TWS running + logged into the PAPER account with the API enabled.
cd /d C:\Users\macie\VSC\Trading
if not exist logs mkdir logs
.venv-openbb\Scripts\python.exe futures\run_mes_bot.py >> logs\mes_bot.log 2>&1
REM push the live status to the VM dashboard (Futures tab reads it there)
if exist futures\status.json scp -o BatchMode=yes -o ConnectTimeout=15 futures\status.json trading-vm:/home/ubuntu/trading/futures/status.json >> logs\mes_bot.log 2>&1
