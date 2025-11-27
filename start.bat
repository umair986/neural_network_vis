@echo off
REM Neural Canvas - Start Script (Batch version)
REM Double-click this file to start both servers

echo ========================================
echo    Neural Canvas - Starting Servers
echo ========================================
echo.

cd /d "%~dp0"

echo [1/2] Starting Backend Server...
start "Neural Canvas - Backend" cmd /k "cd backend && venv\Scripts\activate.bat && echo Backend running on http://localhost:8000 && python -m uvicorn server:app --reload --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak > nul

echo [2/2] Starting Frontend Server...
start "Neural Canvas - Frontend" cmd /k "cd frontend && echo Frontend running on http://localhost:3000 && npm start"

echo.
echo ========================================
echo    Both servers are starting!
echo    Backend:  http://localhost:8000
echo    Frontend: http://localhost:3000
echo ========================================
echo.
echo You can close this window now.
pause
