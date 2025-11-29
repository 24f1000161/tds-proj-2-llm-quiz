# Deploy to Hugging Face Spaces (PowerShell)
# Usage: .\deploy_hf.ps1 [space-name]

param(
    [string]$SpaceName = "llm-quiz-solver"
)

Write-Host "=== Hugging Face Spaces Deployment ===" -ForegroundColor Cyan

# Check if git is installed
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Error: git is not installed" -ForegroundColor Red
    exit 1
}

# Check if huggingface-cli is installed
$hfCli = Get-Command huggingface-cli -ErrorAction SilentlyContinue
if (-not $hfCli) {
    Write-Host "Installing huggingface_hub..." -ForegroundColor Yellow
    pip install huggingface_hub
}

# Check login status
$whoami = huggingface-cli whoami 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Please login to Hugging Face:" -ForegroundColor Yellow
    huggingface-cli login
    $whoami = huggingface-cli whoami
}

$username = ($whoami | Select-Object -First 1).Trim()
Write-Host "Logged in as: $username" -ForegroundColor Green

# Full space path
$spacePath = "$username/$SpaceName"
Write-Host "Deploying to: $spacePath" -ForegroundColor Cyan

# Create temporary directory
$tempDir = Join-Path $env:TEMP "hf-deploy-$(Get-Random)"
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

try {
    # Try to clone existing space
    Write-Host "Cloning space..." -ForegroundColor Yellow
    $cloneResult = git clone "https://huggingface.co/spaces/$spacePath" "$tempDir/space" 2>&1
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Creating new space..." -ForegroundColor Yellow
        huggingface-cli repo create $SpaceName --type space --space_sdk docker 2>&1 | Out-Null
        git clone "https://huggingface.co/spaces/$spacePath" "$tempDir/space"
    }
    
    $spaceDir = "$tempDir/space"
    
    # Copy files
    Write-Host "Copying files..." -ForegroundColor Yellow
    Copy-Item -Path "quiz_solver" -Destination $spaceDir -Recurse -Force
    Copy-Item -Path "pyproject.toml" -Destination $spaceDir -Force
    Copy-Item -Path "uv.lock" -Destination $spaceDir -Force
    Copy-Item -Path "Dockerfile" -Destination $spaceDir -Force
    Copy-Item -Path ".dockerignore" -Destination $spaceDir -Force
    Copy-Item -Path "README_HF.md" -Destination "$spaceDir/README.md" -Force
    
    # Git operations
    Set-Location $spaceDir
    git add .
    git commit -m "Deploy quiz solver" 2>&1 | Out-Null
    
    Write-Host "Pushing to Hugging Face..." -ForegroundColor Yellow
    git push origin main
    
    Write-Host ""
    Write-Host "=== Deployment Complete ===" -ForegroundColor Green
    Write-Host ""
    Write-Host "Space URL: https://huggingface.co/spaces/$spacePath" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "IMPORTANT: Set these secrets in HF Space settings:" -ForegroundColor Yellow
    Write-Host "  - STUDENT_SECRETS=your-email:your-secret" -ForegroundColor White
    Write-Host "  - AIPIPE_TOKEN=your-aipipe-token" -ForegroundColor White
    Write-Host ""
}
finally {
    # Cleanup
    Set-Location $PSScriptRoot
    Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
}
