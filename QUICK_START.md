# Quick Start Guide - FinanceGPT

Get up and running with FinanceGPT in under 5 minutes!

## 🚀 One-Command Setup

### Linux/Mac
```bash
chmod +x setup.sh && ./setup.sh
```

### Windows
```cmd
setup.bat
```

## 📋 Manual Setup (if scripts don't work)

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables
```bash
# Copy the example file
cp .env.example .env

# Edit .env with your API keys
nano .env  # or use any text editor
```

Add your API keys to `.env`:
```
OPENAI_API_KEY=sk-your-openai-key-here
ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key-here
```

### 3. Start the Application
```bash
python web/app.py
```

### 4. Open Your Browser
Navigate to: `http://localhost:5001`

## 🔑 Getting API Keys

### OpenAI API Key (Required)
1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up/login
3. Go to "API Keys" section
4. Click "Create new secret key"
5. Copy the key (starts with `sk-`)
6. **Important**: Add billing information to your OpenAI account

### Alpha Vantage API Key (Required)
1. Go to [alphavantage.co](https://www.alphavantage.co)
2. Click "Get Free API Key"
3. Fill out the form
4. Check your email for the key
5. Free tier: 25 requests/day

## 🎯 First Steps

1. **Choose a conversation type**:
   - Company Research
   - ETF Portfolio Planning
   - Savings Goal Planning

2. **Try these example queries**:
   - "Analyze Apple stock"
   - "Help me build a retirement portfolio"
   - "I want to save $50,000 for a house down payment"

3. **Configure settings**:
   - Click "Settings" in the top navigation
   - Enter your API keys if not done via `.env`

## 🔧 Troubleshooting

### Common Issues

**"No module named 'xyz'"**
```bash
pip install -r requirements.txt
```

**"OpenAI API key not found"**
- Check your `.env` file has the correct key
- Or add keys via the Settings page
- Ensure no extra spaces or quotes

**"Alpha Vantage API limit exceeded"**
- Free tier has 25 requests/day
- Wait 24 hours or upgrade to paid plan

**"Port 5001 already in use"**
```bash
# Kill the process using port 5001
lsof -ti:5001 | xargs kill -9

# Or change the port in web/app.py
```

**Database errors**
```bash
# Remove the database file to reset
rm web/finance_chat.db
```

### Getting Help

1. Check the full [README.md](README.md) for detailed documentation
2. Look at the error messages in the terminal
3. Ensure all API keys are valid and have billing set up
4. Try restarting the application

## 💡 Usage Tips

- **Start simple**: Try basic queries before complex ones
- **Be specific**: "Analyze AAPL for long-term investment" vs "Tell me about Apple"
- **Provide context**: Mention your investment timeline and risk tolerance
- **Use the chat history**: Previous conversations are saved automatically
- **Explore features**: Try different conversation types to see all capabilities

## 🎉 You're Ready!

Once you see the FinanceGPT interface at `http://localhost:5001`, you're all set! Start with a simple company analysis or ETF recommendation to test everything is working.

---

**Need more help?** Check the full [README.md](README.md) or review the code documentation.
