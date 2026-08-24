@echo off
title Flattrade B07 Trading Bot
cd /d "%~dp0"

echo.
echo  ========================================================
echo     FLATTRADE B07 GRAND CHAMPION TRADING BOT
echo     3-Minute Bidirectional CE+PE  -  Zero-Touch Launch
echo  ========================================================
echo.
echo   [1]  LIVE TRADING MODE   (Real Flattrade Orders)
echo   [2]  PAPER / SIM MODE    (No Real Money Risk)
echo   [3]  EXIT
echo.
set /p mode="  Enter choice [1/2/3] (default=2 Paper): "

if "%mode%"=="3" goto :eof
if "%mode%"=="1" goto :LIVE
goto :PAPER

:LIVE
echo.
echo  [LIVE MODE] Starting with real Flattrade orders...
echo  --------------------------------------------------------
python -m flattrade_bot.main --auto-login --live
goto :DONE

:PAPER
echo.
echo  [PAPER MODE] Starting in simulation (no real orders)...
echo  --------------------------------------------------------
python -m flattrade_bot.main --auto-login
goto :DONE

:DONE
echo.
echo  ========================================================
echo  Bot has stopped. See output above for details.
echo  ========================================================
echo.
pause
