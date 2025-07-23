# Vietnam Stock Hedging Dashboard - GitHub Setup Script
# PowerShell script to automate repository creation and file upload

param(
    [Parameter(Mandatory=$true)]
    [string]$GitHubUsername,
    
    [Parameter(Mandatory=$false)]
    [string]$RepositoryName = "vietnam-stock-hedging-dashboard",
    
    [Parameter(Mandatory=$false)]
    [string]$DestinationPath = "$env:USERPROFILE\Documents\$RepositoryName"
)

Write-Host "Vietnam Stock Hedging Dashboard - GitHub Setup" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green

# Step 1: Check if Git is installed
Write-Host "`nStep 1: Checking Git installation..." -ForegroundColor Yellow
try {
    $gitVersion = git --version
    Write-Host "Git is installed: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "Git is not installed. Please install Git first:" -ForegroundColor Red
    Write-Host "   Download from: https://git-scm.com/download/win" -ForegroundColor Red
    exit 1
}

# Step 2: Get source path
$SourcePath = $PWD.Path
Write-Host "`nStep 2: Source path identified" -ForegroundColor Yellow
Write-Host "   Source: $SourcePath" -ForegroundColor Cyan
Write-Host "   Destination: $DestinationPath" -ForegroundColor Cyan

# Step 3: Display GitHub instructions
Write-Host "`nStep 3: Create GitHub Repository" -ForegroundColor Yellow
Write-Host "   Please follow these steps manually:" -ForegroundColor White
Write-Host "   1. Go to https://github.com" -ForegroundColor White
Write-Host "   2. Click 'New repository'" -ForegroundColor White
Write-Host "   3. Repository name: $RepositoryName" -ForegroundColor Green
Write-Host "   4. Description: Professional Vietnam Stock Hedging Dashboard" -ForegroundColor White
Write-Host "   5. Choose Public or Private" -ForegroundColor White
Write-Host "   6. Add a README file" -ForegroundColor White
Write-Host "   7. Add .gitignore: Python" -ForegroundColor White
Write-Host "   8. Choose a license: MIT" -ForegroundColor White
Write-Host "   9. Click 'Create repository'" -ForegroundColor White

$confirm = Read-Host "`n   Have you created the repository? (y/n)"
if ($confirm -ne 'y') {
    Write-Host "Please create the repository first, then run this script again." -ForegroundColor Red
    exit 1
}

# Step 4: Clone repository
Write-Host "`nStep 4: Cloning repository..." -ForegroundColor Yellow
$repoUrl = "https://github.com/$GitHubUsername/$RepositoryName.git"

try {
    if (Test-Path $DestinationPath) {
        Write-Host "   Destination directory exists. Removing..." -ForegroundColor Orange
        Remove-Item $DestinationPath -Recurse -Force
    }
    
    Write-Host "   Cloning from: $repoUrl" -ForegroundColor Cyan
    git clone $repoUrl $DestinationPath
    
    # Verify the clone was successful
    if (Test-Path $DestinationPath) {
        Write-Host "Repository cloned successfully" -ForegroundColor Green
    } else {
        Write-Host "Clone appeared to succeed but destination directory not found" -ForegroundColor Red
        Write-Host "This might be a permissions issue or the repository doesn't exist" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Failed to clone repository. Check the URL and try again:" -ForegroundColor Red
    Write-Host "   $repoUrl" -ForegroundColor Red
    Write-Host "   Make sure you have created the repository on GitHub first" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Step 5: Copy project files
Write-Host "`nStep 5: Copying project files..." -ForegroundColor Yellow

# List of files to copy (excluding large data files for now)
$FilesToCopy = @(
    "vietnam_hedge_dashboard.py",
    "vietnam_hedging_engine.py",
    "vietnam_hedge_pipeline.py",
    "vietnam_hedge_monitor.py",
    "vietnam_backtesting_engine.py",
    "vietnam_advanced_analytics.py",
    "vietnam_realtime_system.py",
    "advanced_demo.py",
    "run_dashboard.py",
    "simple_demo.py",
    "vietnam_hedge_app.py",
    "data.py",
    "README.md",
    "requirements.txt",
    ".gitignore",
    "ENHANCED_INTERACTIVITY_SUMMARY.md",
    "ADVANCED_FEATURES_SUMMARY.md",
    "BACKTESTING_TAB_FEATURES.md",
    "10_YEAR_ANALYSIS_UPGRADE.md",
    "GITHUB_SETUP_INSTRUCTIONS.md",
    "Vietnam_Stock_Hedging_Application_Documentation.ipynb",
    "Bai39_CauHoi_HedgingBetaVaSector.ipynb"
)

$copiedFiles = 0
$totalFiles = $FilesToCopy.Count

foreach ($file in $FilesToCopy) {
    $sourceFile = Join-Path $SourcePath $file
    $destFile = Join-Path $DestinationPath $file
    
    if (Test-Path $sourceFile) {
        Copy-Item $sourceFile $destFile -Force
        $copiedFiles++
        Write-Host "   Copied: $file" -ForegroundColor Green
    } else {
        Write-Host "   Not found: $file" -ForegroundColor Orange
    }
}

Write-Host "`nCopied $copiedFiles out of $totalFiles files" -ForegroundColor Cyan

# Step 6: Handle large files
Write-Host "`nStep 6: Large file handling..." -ForegroundColor Yellow
$largeFiles = @("CafeF.feather.zip", "Copy of 39-why-hedge.ipynb", "Bai39_TraLoi_HedgingBetaVaSector.ipynb")

foreach ($file in $largeFiles) {
    $largeSrcFile = Join-Path $SourcePath $file
    if (Test-Path $largeSrcFile) {
        $fileSize = (Get-Item $largeSrcFile).Length / 1MB
        if ($fileSize -gt 25) {
            Write-Host "   Large file detected: $file ($([math]::Round($fileSize, 1)) MB)" -ForegroundColor Orange
            Write-Host "      Consider using Git LFS or uploading separately" -ForegroundColor Orange
        } else {
            Copy-Item $largeSrcFile (Join-Path $DestinationPath $file) -Force
            Write-Host "   Copied large file: $file" -ForegroundColor Green
        }
    }
}

# Step 7: Git operations
Write-Host "`nStep 7: Committing and pushing files..." -ForegroundColor Yellow

Set-Location $DestinationPath

try {
    # Add all files
    git add .
    Write-Host "   Files staged for commit" -ForegroundColor Green
    
    # Commit with detailed message
    $commitMessage = @"
Initial commit: Vietnam Stock Hedging Dashboard

Complete Professional Trading Platform:
- Enhanced Streamlit dashboard with interactive UI
- Advanced hedging algorithms and risk management
- Comprehensive backtesting framework
- Real-time monitoring and analytics
- Machine learning integration
- Vietnam market data integration
- Professional documentation and examples

Key Features:
- Multi-strategy hedging (Beta, Sector, Dynamic)
- Advanced portfolio analytics
- Interactive charts with range selectors
- Risk alert system
- Performance attribution analysis
- Monte Carlo simulations
- Professional reporting

Technical Stack:
- Python 3.8+
- Streamlit for web interface
- Plotly for interactive visualizations
- Pandas/NumPy for data processing
- Scikit-learn for ML models
- Comprehensive test suite

Ready for production deployment and collaboration!
"@
    
    git commit -m $commitMessage
    Write-Host "   Files committed successfully" -ForegroundColor Green
    
    # Push to GitHub
    git push origin main
    Write-Host "   Files pushed to GitHub successfully" -ForegroundColor Green
    
} catch {
    Write-Host "   Git operation failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "   You may need to configure Git credentials or check repository permissions" -ForegroundColor Orange
}

# Step 8: Final instructions
Write-Host "`nStep 8: Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Write-Host "`nNext Steps:" -ForegroundColor Yellow
Write-Host "1. Repository URL: https://github.com/$GitHubUsername/$RepositoryName" -ForegroundColor Cyan
Write-Host "2. Add repository topics/tags for discoverability" -ForegroundColor White
Write-Host "3. Update README.md with deployment instructions" -ForegroundColor White
Write-Host "4. Set up GitHub Secrets for sensitive data" -ForegroundColor White
Write-Host "5. Consider setting up GitHub Actions for CI/CD" -ForegroundColor White
Write-Host "6. Add collaborators if working in a team" -ForegroundColor White

Write-Host "`nRepository Statistics:" -ForegroundColor Yellow
Write-Host "   Total files copied: $copiedFiles" -ForegroundColor Cyan

if (Test-Path $DestinationPath) {
    try {
        $pyFiles = (Get-ChildItem $DestinationPath -Filter '*.py' -ErrorAction SilentlyContinue).Count
        $ipynbFiles = (Get-ChildItem $DestinationPath -Filter '*.ipynb' -ErrorAction SilentlyContinue).Count
        $mdFiles = (Get-ChildItem $DestinationPath -Filter '*.md' -ErrorAction SilentlyContinue).Count
        
        Write-Host "   Python files: $pyFiles" -ForegroundColor Cyan
        Write-Host "   Jupyter notebooks: $ipynbFiles" -ForegroundColor Cyan
        Write-Host "   Documentation files: $mdFiles" -ForegroundColor Cyan
    } catch {
        Write-Host "   Unable to get detailed file statistics" -ForegroundColor Orange
    }
} else {
    Write-Host "   Destination directory not accessible for statistics" -ForegroundColor Orange
}

Write-Host "`nYour Vietnam Stock Hedging Dashboard is now on GitHub!" -ForegroundColor Green
Write-Host "   Ready for collaboration, deployment, and sharing!" -ForegroundColor Green

# Return to original directory
Set-Location $SourcePath 