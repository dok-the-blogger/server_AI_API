@echo off
setlocal
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -STA -ExecutionPolicy Bypass -File "%~dp0run-pilot.ps1" -Action inspect
set "pilotExit=%ERRORLEVEL%"
echo.
echo Inspection mode uses GET requests only. Saved reports are identified above.
pause
exit /b %pilotExit%
