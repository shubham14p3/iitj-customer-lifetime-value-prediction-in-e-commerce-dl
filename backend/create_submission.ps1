# PowerShell script to create submission zip file for Group 8
# This script creates a clean submission package excluding unnecessary files

Write-Host "`n=== Creating Submission Package for Group 8 ===" -ForegroundColor Green

# Remove existing zip if present
$zipFile = "group_08.zip"
if (Test-Path $zipFile) {
    Remove-Item $zipFile -Force
    Write-Host "Removed existing $zipFile" -ForegroundColor Yellow
}

# Files and directories to include
$includeItems = @(
    "models",
    "data",
    "utils",
    "scripts",
    "templates",
    "static",
    "README.md",
    "report.md",
    "requirements.txt",
    "verify_setup.py",
    "app.py"
)

# Files and directories to exclude
$excludePatterns = @(
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".git",
    ".gitignore",
    "uploads",
    "*.log",
    "*.tmp",
    "olist_customers_dataset.csv"  # Large file, not needed for submission
)

Write-Host "`nIncluding files and directories:" -ForegroundColor Cyan
foreach ($item in $includeItems) {
    if (Test-Path $item) {
        Write-Host "  + $item" -ForegroundColor Green
    } else {
        Write-Host "  - $item (not found)" -ForegroundColor Red
    }
}

Write-Host "`nExcluding:" -ForegroundColor Yellow
foreach ($pattern in $excludePatterns) {
    Write-Host "  - $pattern" -ForegroundColor Gray
}

# Create temporary directory for clean submission
$tempDir = "submission_temp"
if (Test-Path $tempDir) {
    Remove-Item $tempDir -Recurse -Force
}
New-Item -ItemType Directory -Path $tempDir | Out-Null

Write-Host "`nCopying files..." -ForegroundColor Cyan

# Copy included items
foreach ($item in $includeItems) {
    if (Test-Path $item) {
        Copy-Item -Path $item -Destination $tempDir -Recurse -Force
        Write-Host "  Copied: $item" -ForegroundColor Gray
    }
}

# Remove excluded patterns from temp directory
Write-Host "`nCleaning up excluded files..." -ForegroundColor Cyan
Get-ChildItem -Path $tempDir -Recurse -Force | Where-Object {
    $exclude = $false
    foreach ($pattern in $excludePatterns) {
        if ($_.FullName -like "*$pattern*") {
            $exclude = $true
            break
        }
    }
    if ($exclude) {
        Remove-Item $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# Create zip file
Write-Host "`nCreating zip file: $zipFile" -ForegroundColor Cyan
Compress-Archive -Path "$tempDir\*" -DestinationPath $zipFile -Force

# Clean up temp directory
Remove-Item $tempDir -Recurse -Force

# Check zip file
if (Test-Path $zipFile) {
    $zipSize = (Get-Item $zipFile).Length / 1MB
    Write-Host "`n✓ Successfully created $zipFile" -ForegroundColor Green
    Write-Host "  Size: $([math]::Round($zipSize, 2)) MB" -ForegroundColor Gray
    
    Write-Host "`n=== Submission Package Contents ===" -ForegroundColor Green
    Write-Host "The zip file contains:" -ForegroundColor Cyan
    Write-Host '  - All source code (models, data, utils, scripts)' -ForegroundColor White
    Write-Host '  - Trained model (models/saved_model.pth)' -ForegroundColor White
    Write-Host '  - Sample data (data/clv_features_sample.csv)' -ForegroundColor White
    Write-Host '  - README.md (complete documentation)' -ForegroundColor White
    Write-Host '  - report.md (project report)' -ForegroundColor White
    Write-Host '  - requirements.txt (dependencies)' -ForegroundColor White
    Write-Host '  - Flask web application (app.py, templates, static)' -ForegroundColor White
    Write-Host '  - Verification script (verify_setup.py)' -ForegroundColor White
    
    Write-Host "`n✓ Ready for submission!" -ForegroundColor Green
} else {
    Write-Host "`n✗ Failed to create zip file" -ForegroundColor Red
    exit 1
}

