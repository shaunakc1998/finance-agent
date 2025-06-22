@echo off
setlocal enabledelayedexpansion

echo 🚀 FinanceGPT Setup Script (Windows)
echo ===================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH.
    echo Please install Python 3.9-3.11 from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo 📋 Detected Python version: %PYTHON_VERSION%

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo 📦 Virtual environment already exists
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

REM Install dependencies
echo 📚 Installing dependencies (this may take 5-10 minutes)...
pip install -r requirements.txt

REM Check if .env file exists
if not exist ".env" (
    echo ⚙️  Setting up environment file...
    copy .env.example .env >nul
    echo ✅ Created .env file from template
    echo.
    echo 🔑 IMPORTANT: You need to add your API keys to the .env file
    echo    Edit .env and add your:
    echo    - OpenAI API Key (get from: https://platform.openai.com)
    echo    - Alpha Vantage API Key (get from: https://www.alphavantage.co/support/#api-key)
    echo.
) else (
    echo ⚙️  .env file already exists
)

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist "web\database" mkdir web\database
if not exist "models" mkdir models
if not exist "vectors" mkdir vectors
if not exist "visualizations" mkdir visualizations
if not exist "profiles" mkdir profiles
if not exist "cache" mkdir cache
if not exist "test_results" mkdir test_results

echo.
echo 🎉 Setup completed successfully!
echo.
echo 📝 Next steps:
echo 1. Edit the .env file with your API keys:
echo    notepad .env
echo.
echo 2. Start the application:
echo    venv\Scripts\activate.bat
echo    python web\app.py
echo.
echo 3. Open your browser and go to:
echo    http://localhost:5001
echo.
echo 💡 Need help? Check the README.md file for detailed instructions.
echo.
pause
