# Get the folder where this script is located
$ProjectDir = $PSScriptRoot

# Define the virtual environment directory
$VenvDir = Join-Path $ProjectDir ".venv"

Write-Host "Creating virtual environment..." -ForegroundColor Cyan
python -m venv $VenvDir

# Activate the virtual environment
. "$VenvDir\Scripts\Activate.ps1"

Write-Host "Upgrading pip and installing requirements..." -ForegroundColor Cyan
python -m pip install --upgrade pip
pip install -r (Join-Path $ProjectDir "requirements.txt")

Write-Host ""
Write-Host "Done! Virtualenv created at $VenvDir and dependencies installed." -ForegroundColor Green
Write-Host "To activate it manually later, run: .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow