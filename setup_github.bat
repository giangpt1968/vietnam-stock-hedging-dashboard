@echo off
echo 🇻🇳 Vietnam Stock Hedging Dashboard - GitHub Setup
echo ==============================================
echo.
echo This script will help you upload your project to GitHub
echo.

set /p username="Enter your GitHub username: "

if "%username%"=="" (
    echo Error: GitHub username is required
    pause
    exit /b 1
)

echo.
echo Starting GitHub setup for user: %username%
echo.

powershell -ExecutionPolicy Bypass -File "setup_github_repo.ps1" -GitHubUsername "%username%"

pause 