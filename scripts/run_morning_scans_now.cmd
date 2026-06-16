@echo off
REM One-click: run the early morning scans (news-edge + hype) on a travel day.
REM Double-click this file. It calls run_morning_scans_now.ps1 (sitting next to it).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_morning_scans_now.ps1"
