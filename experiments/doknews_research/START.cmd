@echo off
setlocal
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -STA -ExecutionPolicy Bypass -File "%~dp0run-pilot.ps1"
set "pilotExit=%ERRORLEVEL%"
echo.
echo Keep receipt.json and dossier.md for reviewing the result. Do not share the key file.
pause
exit /b %pilotExit%
