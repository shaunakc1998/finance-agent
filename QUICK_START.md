# 🚀 Quick Start Guide for FinanceGPT

This guide will help you get FinanceGPT running in just a few minutes!

## 📋 Before You Start

Make sure you have:
- **Python 3.9, 3.10, or 3.11** installed (NOT 3.12+)
- **Git** installed
- An **OpenAI API key** (required)
- An **Alpha Vantage API key** (required, free)

## 🎯 Super Quick Setup (Automated)

### Option 1: Linux/macOS (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/finance-agent.git
cd finance-agent

# Run the automated setup script
./setup.sh

# Edit your API keys
nano .env

# Start the application
source venv/bin/activate
python web/app.py
```

### Option 2: Windows
```cmd
# Clone the repository
git clone https://github.com/yourusername/finance-agent.git
cd finance-agent

# Run the automated setup script
setup.bat

# Edit your API keys (will open automatically)
# Add your OpenAI and Alpha Vantage API keys

# Start the application
venv\Scripts\activate.bat
python web\app.py
```

## 🔑 Getting API Keys

### OpenAI API Key (Required)
1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up/login
3. Go to "API Keys" section
4. Click "Create new secret key"
5. Copy the key (starts with `sk-`)

### Alpha Vantage API Key (Required, Free)
1. Go to [alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
2. Click "Get Free API Key"
3. Fill out the form
4. Check your email for the key

## 🌐 Access the Application

Once running, open your browser and go to:
**http://localhost:5001**

## 🆘 Having Issues?

### Common Problems:

**Python Version Issues:**
```bash
python --version  # Should show 3.9, 3.10, or 3.11
```

**Permission Issues (Linux/macOS):**
```bash
chmod +x setup.sh
```

**Dependencies Failing:**
```bash
# Try installing PyTorch separately first
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Port Already in Use:**
- Change port in `web/app.py` from 5001 to 5002

### Need More Help?
- Check the detailed [README.md](README.md) file
- Look at the troubleshooting section
- Create an issue on GitHub

## 🎉 You're Ready!

Once the application is running:
1. Go to **Settings** and add your API keys
2. Start chatting with the financial assistant
3. Try the **Guided Mode** for structured planning
4. Explore different financial analysis features

**Happy Financial Analysis! 📈💰**
