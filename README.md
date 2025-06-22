# FinanceGPT - Advanced Financial Research Agent

[![Python Lint and Test](https://github.com/yourusername/finance-agent/actions/workflows/python-lint-test.yml/badge.svg)](https://github.com/yourusername/finance-agent/actions/workflows/python-lint-test.yml)
[![Docker Build](https://github.com/yourusername/finance-agent/actions/workflows/docker-build.yml/badge.svg)](https://github.com/yourusername/finance-agent/actions/workflows/docker-build.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive financial research agent with advanced price forecasting capabilities using machine learning models and in-depth financial analysis with vector search. Features a modern web interface for easy interaction and API key management.

## 🚀 Quick Start

> **🎯 Super Fast Setup**: Use our automated setup scripts!
> - **Linux/macOS**: Run `./setup.sh`
> - **Windows**: Run `setup.bat`
> - **Need help?**: See [QUICK_START.md](QUICK_START.md)

### Prerequisites

- **Python 3.9 - 3.11** (Required - Python 3.12+ may have compatibility issues with some dependencies)
- **pip** (Python package manager)
- **Git** (for cloning the repository)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/finance-agent.git
cd finance-agent
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your API keys
# You can use any text editor, for example:
nano .env
```

Add your API keys to the `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
```

### 5. Run the Application

```bash
# Start the web application
python web/app.py
```

Open your browser and navigate to **http://localhost:5001**

## 📋 System Requirements

### Python Version
- **Required**: Python 3.9, 3.10, or 3.11
- **Not supported**: Python 3.12+ (due to dependency compatibility issues)

### Operating System
- **Windows**: Windows 10/11
- **macOS**: macOS 10.15+
- **Linux**: Ubuntu 18.04+, CentOS 7+, or equivalent

### Hardware Requirements
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: 2GB free space for dependencies and models
- **CPU**: Any modern CPU (multi-core recommended for ML training)

## 🔧 Detailed Installation Guide

### Step 1: Check Python Version

```bash
python --version
```

If you don't have Python 3.9-3.11, install it:

**Windows:**
- Download from [python.org](https://www.python.org/downloads/)
- Or use [pyenv-win](https://github.com/pyenv-win/pyenv-win)

**macOS:**
```bash
# Using Homebrew
brew install python@3.11

# Using pyenv
brew install pyenv
pyenv install 3.11.7
pyenv global 3.11.7
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-pip
```

### Step 2: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/finance-agent.git
cd finance-agent

# Create virtual environment with specific Python version
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# Verify you're using the correct Python
which python  # Should show path to venv
python --version  # Should show Python 3.9-3.11
```

### Step 3: Install Dependencies

```bash
# Upgrade pip to latest version
pip install --upgrade pip setuptools wheel

# Install dependencies (this may take 5-10 minutes)
pip install -r requirements.txt
```

### Step 4: Get API Keys

#### OpenAI API Key (Required)
1. Go to [OpenAI Platform](https://platform.openai.com)
2. Sign up or log in
3. Navigate to API Keys section
4. Click "Create new secret key"
5. Copy the key (starts with `sk-`)

#### Alpha Vantage API Key (Required)
1. Go to [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Click "Get Free API Key"
3. Fill out the form
4. Check your email for the API key

### Step 5: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit with your preferred editor
nano .env
# OR
code .env  # If you have VS Code
# OR
notepad .env  # Windows
```

Your `.env` file should look like:
```env
# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-key-here

# Alpha Vantage Configuration  
ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key-here

# Optional: Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
```

### Step 6: Run the Application

```bash
# Start the web server
python web/app.py
```

You should see output like:
```
Database initialized at /path/to/finance-agent/web/database/finance_chat.db
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5001
 * Running on http://[your-ip]:5001
```

Open your browser and go to **http://localhost:5001**

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. Python Version Issues
**Error**: `ModuleNotFoundError` or compatibility warnings

**Solution**: Ensure you're using Python 3.9-3.11:
```bash
python --version
# If wrong version, create new venv with correct Python
python3.11 -m venv venv_new
source venv_new/bin/activate
pip install -r requirements.txt
```

#### 2. Dependency Installation Failures
**Error**: `Failed building wheel` or compilation errors

**Solution**: Install build tools:
```bash
# Windows
pip install --upgrade setuptools wheel

# macOS
xcode-select --install
brew install cmake

# Linux (Ubuntu/Debian)
sudo apt install build-essential python3-dev
```

#### 3. PyTorch Installation Issues
**Error**: PyTorch installation fails or is very slow

**Solution**: Install PyTorch separately:
```bash
# CPU-only version (faster download)
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Then install other requirements
pip install -r requirements.txt
```

#### 4. Memory Issues During Installation
**Error**: `MemoryError` during pip install

**Solution**: Install packages one by one:
```bash
pip install --no-cache-dir -r requirements.txt
# OR install in smaller batches
pip install Flask==3.0.0 openai==1.51.2 langchain==0.3.7
pip install pandas==2.1.4 numpy==1.24.4 scikit-learn==1.3.2
# Continue with remaining packages
```

#### 5. Port Already in Use
**Error**: `Address already in use` on port 5001

**Solution**: Use a different port:
```bash
# Edit web/app.py and change the last line to:
app.run(debug=True, port=5002, host='0.0.0.0')
```

#### 6. API Key Issues
**Error**: OpenAI or Alpha Vantage API errors

**Solution**: 
- Verify API keys are correct in `.env` file
- Check API key quotas and billing
- Test API keys independently:
```bash
python -c "
import openai
import os
from dotenv import load_dotenv
load_dotenv()
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
print('OpenAI API key is valid')
"
```

#### 7. Database Issues
**Error**: SQLite database errors

**Solution**: Delete and recreate database:
```bash
rm -rf web/database/
python web/app.py  # Will recreate database
```

### Performance Optimization

#### For Faster Startup
```bash
# Install only essential packages for basic functionality
pip install Flask==3.0.0 openai==1.51.2 langchain==0.3.7 langchain-openai==0.2.8 python-dotenv==1.0.0 requests==2.31.0
```

#### For Better ML Performance
```bash
# Install with GPU support (if you have NVIDIA GPU)
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

## 🐳 Docker Installation (Alternative)

If you prefer Docker or have dependency issues:

```bash
# Clone repository
git clone https://github.com/yourusername/finance-agent.git
cd finance-agent

# Create .env file with your API keys
cp .env.example .env
# Edit .env with your API keys

# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f
```

Access the application at **http://localhost:5001**

## 🌟 Features

### Web Interface
- **Modern UI**: Responsive design with glassmorphism effects
- **Chat History**: Save and revisit previous conversations
- **API Key Management**: Secure storage and management of API keys
- **Guided Mode**: Step-by-step financial planning assistance
- **Settings Page**: Easy configuration management

### Financial Analysis
- **Fundamental Analysis**: P/E ratio, EPS, market cap, financial ratios
- **Technical Analysis**: RSI, SMA, MACD, Bollinger Bands
- **Price Forecasting**: ML-based predictions with confidence intervals
- **Investment Strategy**: Buy/sell/hold recommendations
- **ETF Analysis**: Portfolio recommendations and SIP planning
- **Economic Factors**: Macro-economic analysis and impact assessment

### Machine Learning Models
- **Random Forest**: Non-linear pattern recognition
- **Gradient Boost**: High accuracy predictions
- **Ensemble Models**: Combined model predictions
- **Neural Networks**: Deep learning approaches
- **ARIMA**: Time series statistical modeling

## 📊 Usage Examples

### Web Interface
1. Open http://localhost:5001
2. Go to Settings and add your API keys
3. Start chatting with the financial assistant
4. Try guided mode for structured financial planning

### Example Queries
- "Analyze AAPL stock fundamentals"
- "What's the technical analysis for TSLA?"
- "Create a diversified ETF portfolio for $10,000"
- "Forecast NVDA price for next 20 days"
- "Compare MSFT vs GOOGL investment potential"

### CLI Usage (Alternative)
```bash
python chat_agent.py
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest

# Format code
black .

# Lint code
flake8 .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

If you encounter issues:

1. Check the [Troubleshooting](#-troubleshooting) section above
2. Search existing [GitHub Issues](https://github.com/yourusername/finance-agent/issues)
3. Create a new issue with:
   - Your Python version (`python --version`)
   - Your operating system
   - Full error message
   - Steps to reproduce

## 🔄 Updates

To update to the latest version:

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart the application
python web/app.py
```

---

**Happy Financial Analysis! 📈💰**
