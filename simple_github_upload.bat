@echo off
echo Simple GitHub Upload Script
echo ===========================

set /p username="Enter your GitHub username: "
set REPO_NAME=vietnam-stock-hedging-dashboard

echo.
echo Please make sure you have:
echo 1. Created the repository '%REPO_NAME%' on GitHub
echo 2. Made it public or private as desired
echo 3. Added a README file when creating it
echo.
pause

echo Initializing git repository...
git init

echo Adding files...
git add .

echo Committing files...
git commit -m "Initial commit: Vietnam Stock Hedging Dashboard

Complete Professional Trading Platform:
- Enhanced Streamlit dashboard with interactive UI
- Advanced hedging algorithms and risk management
- Comprehensive backtesting framework
- Real-time monitoring and analytics
- Machine learning integration
- Vietnam market data integration

Ready for production deployment!"

echo Adding remote repository...
git remote add origin https://github.com/%username%/%REPO_NAME%.git

echo Setting main branch...
git branch -M main

echo Pushing to GitHub...
git push -u origin main

echo.
echo Upload complete!
echo Repository URL: https://github.com/%username%/%REPO_NAME%
echo.
pause 