@echo off
setlocal
cd /d "%~dp0"
call "%~dp0START-UNLEASHED.bat" --wizard %*
exit /b %errorlevel%
