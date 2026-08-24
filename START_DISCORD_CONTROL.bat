@echo off
setlocal
cd /d "%~dp0"
python -m flattrade_bot.discord_control
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
