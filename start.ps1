# Neural Canvas - Start Script
# This script starts both the backend and frontend servers

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Neural Canvas - Starting Servers    " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get the script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Start Backend
Write-Host "[1/2] Starting Backend Server..." -ForegroundColor Yellow
$backendPath = Join-Path $scriptDir "backend"

# Check if venv exists
$venvPath = Join-Path $backendPath "venv"
if (-Not (Test-Path $venvPath)) {
    Write-Host "  Creating Python virtual environment..." -ForegroundColor Gray
    Push-Location $backendPath
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    Pop-Location
}

# Start backend in a new PowerShell window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$backendPath'; .\venv\Scripts\Activate.ps1; Write-Host 'Backend running on http://localhost:8000' -ForegroundColor Green; python -m uvicorn server:app --reload --host 0.0.0.0 --port 8000"

Write-Host "  Backend starting on http://localhost:8000" -ForegroundColor Green
Start-Sleep -Seconds 2

# Start Frontend
Write-Host "[2/2] Starting Frontend Server..." -ForegroundColor Yellow
$frontendPath = Join-Path $scriptDir "frontend"

# Check if node_modules exists
$nodeModulesPath = Join-Path $frontendPath "node_modules"
if (-Not (Test-Path $nodeModulesPath)) {
    Write-Host "  Installing npm dependencies..." -ForegroundColor Gray
    Push-Location $frontendPath
    npm install
    Pop-Location
}

# Start frontend in a new PowerShell window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$frontendPath'; Write-Host 'Frontend running on http://localhost:3000' -ForegroundColor Green; npm start"

Write-Host "  Frontend starting on http://localhost:3000" -ForegroundColor Green
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Both servers are starting!          " -ForegroundColor Cyan
Write-Host "   Backend:  http://localhost:8000     " -ForegroundColor White
Write-Host "   Frontend: http://localhost:3000     " -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to exit this window..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
