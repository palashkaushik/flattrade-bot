@echo off
shutdown.exe /a
if errorlevel 1 (
    echo No shutdown is currently pending.
) else (
    echo Weekend shutdown cancelled.
)
pause
