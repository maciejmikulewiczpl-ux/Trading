@echo off
REM Surface scheduled-task wrapper for the MES momentum paper bot. Fired every 5 min on weekdays
REM during a local window that covers US RTH (bot self-guards to ET market hours, so off-hours
REM passes exit instantly). Needs TWS running + logged into the PAPER account with the API enabled.
cd /d C:\Users\macie\VSC\Trading
if not exist logs mkdir logs
.venv-openbb\Scripts\python.exe futures\run_mes_bot.py >> logs\mes_bot.log 2>&1
