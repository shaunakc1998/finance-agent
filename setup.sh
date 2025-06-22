#!/bin/bash

# FinanceGPT Setup Script
# This script automates the installation process for FinanceGPT

set -e  # Exit on any error

echo "🚀 FinanceGPT Setup Script"
echo "=========================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9-3.11 first."
    echo "Visit: https://www.python.org/downloads/"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "📋 Detected Python version: $PYTHON_VERSION"

if [ "$MAJOR_VERSION" -ne 3 ] || [ "$MINOR_VERSION" -lt 9 ] || [ "$MINOR_VERSION" -gt 11 ]; then
    echo "⚠️  Warning: Python $PYTHON_VERSION detected. Recommended: Python 3.9-3.11"
    echo "   Some dependencies may not work correctly with Python 3.12+"
    read -p "   Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "📦 Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "📚 Installing dependencies (this may take 5-10 minutes)..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚙️  Setting up environment file..."
    cp .env.example .env
    echo "✅ Created .env file from template"
    echo ""
    echo "🔑 IMPORTANT: You need to add your API keys to the .env file"
    echo "   Edit .env and add your:"
    echo "   - OpenAI API Key (get from: https://platform.openai.com)"
    echo "   - Alpha Vantage API Key (get from: https://www.alphavantage.co/support/#api-key)"
    echo ""
else
    echo "⚙️  .env file already exists"
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p web/database
mkdir -p models
mkdir -p vectors
mkdir -p visualizations
mkdir -p profiles
mkdir -p cache
mkdir -p test_results

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📝 Next steps:"
echo "1. Edit the .env file with your API keys:"
echo "   nano .env"
echo ""
echo "2. Start the application:"
echo "   source venv/bin/activate"
echo "   python web/app.py"
echo ""
echo "3. Open your browser and go to:"
echo "   http://localhost:5001"
echo ""
echo "💡 Need help? Check the README.md file for detailed instructions."
echo ""
