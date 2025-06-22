# FinanceGPT - AI-Powered Financial Planning Assistant

A comprehensive financial planning and investment research tool powered by AI. Get personalized investment advice, company analysis, ETF recommendations, and savings goal planning.

## 🚀 Features

- **Company Research**: Deep analysis of stocks with fundamentals, technicals, and earnings insights
- **ETF Portfolio Planning**: Personalized ETF recommendations and portfolio optimization
- **Savings Goal Planning**: Comprehensive financial planning for specific goals
- **Real-time Market Data**: Live stock prices and technical indicators
- **AI-Powered Analysis**: GPT-4 powered financial insights and recommendations
- **Interactive Web Interface**: Clean, modern web interface with chat history
- **Personalized Recommendations**: Tailored advice based on your financial situation

## 📁 Project Structure

```
finance-agent/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── setup.sh                 # Linux/Mac setup script
├── setup.bat                # Windows setup script
├── QUICK_START.md           # Quick start guide
├── LICENSE                  # MIT License
│
├── chat_agent.py            # Core AI agent with financial tools
│
├── tools/                   # Financial analysis tools
│   ├── economic_factors.py  # Economic indicators and analysis
│   ├── financial_analysis.py # Company financial insights
│   ├── forecast.py          # Price forecasting models
│   ├── fundamentals.py      # Fundamental analysis
│   ├── portfolio.py         # Portfolio optimization
│   ├── product_search.py    # Product pricing research
│   ├── strategy.py          # Investment strategy recommendations
│   ├── technicals.py        # Technical analysis
│   ├── user_profile.py      # User personalization
│   └── visualization.py     # Data visualization
│
└── web/                     # Web application
    ├── app.py               # Flask web server
    ├── finance_chat.db      # SQLite database (created on first run)
    ├── static/
    │   ├── css/
    │   │   ├── guided_styles.css  # Main app styles
    │   │   └── styles.css         # Settings page styles
    │   └── js/
    │       └── guided_chat.js     # Frontend JavaScript
    └── templates/
        ├── guided_chat.html       # Main chat interface
        └── settings.html          # API key configuration
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Alpha Vantage API key (free)

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd finance-agent
   ```

2. **Run the setup script**
   
   **Linux/Mac:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```
   
   **Windows:**
   ```cmd
   setup.bat
   ```

3. **Configure API keys**
   - Copy `.env.example` to `.env`
   - Add your API keys to `.env` file
   - Or configure them through the web interface at `/settings`

4. **Start the application**
   ```bash
   python web/app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:5001`

### Manual Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run the application**
   ```bash
   python web/app.py
   ```

## 🔑 API Keys Setup

### OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new secret key
5. Add billing information (required for API usage)

### Alpha Vantage API Key
1. Visit [Alpha Vantage](https://www.alphavantage.co)
2. Click "Get Free API Key"
3. Fill out the registration form
4. Check your email for the API key
5. Free tier includes 25 requests per day

## 💡 Usage

### Web Interface

1. **Start a conversation**: Choose from company research, ETF exploration, or savings goals
2. **Get personalized advice**: Provide your financial context for tailored recommendations
3. **Explore features**: Use the chat interface to ask about any financial topic
4. **Manage conversations**: Save, rename, and organize your chat history

### Command Line Interface

Run the chat agent directly:
```bash
python chat_agent.py
```

### Example Queries

- "Analyze Apple stock for a long-term investment"
- "Help me build a diversified ETF portfolio with $10,000"
- "I want to save for a Tesla Model 3, help me create a plan"
- "What are the best dividend ETFs for retirement?"
- "Compare Microsoft vs Google for investment"

## 🧰 Core Tools

- **Fundamentals Analysis**: P/E ratios, EPS, market cap, financial health
- **Technical Analysis**: RSI, moving averages, trend analysis
- **Earnings Insights**: Recent earnings calls, SEC filings, management commentary
- **Industry Comparison**: Peer analysis and sector performance
- **Economic Factors**: Interest rates, inflation, macroeconomic context
- **Portfolio Optimization**: Modern portfolio theory, risk-adjusted returns
- **Price Forecasting**: Machine learning models for price predictions
- **Savings Planning**: Goal-based financial planning with investment strategies

## 🔧 Configuration

### Environment Variables

Create a `.env` file with:
```
OPENAI_API_KEY=your_openai_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
```

### Web Interface Settings

Configure API keys through the web interface:
1. Navigate to `/settings`
2. Enter your API keys
3. Keys are stored securely in the local database

## 🚀 Deployment

### Local Development
```bash
python web/app.py
```

### Production Deployment
- Use a production WSGI server like Gunicorn
- Set up environment variables securely
- Configure database backups
- Use HTTPS in production

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- Check the [QUICK_START.md](QUICK_START.md) for common issues
- Review the code documentation in each module
- Open an issue for bugs or feature requests

## 🔮 Roadmap

- [ ] Additional data sources integration
- [ ] Advanced portfolio analytics
- [ ] Mobile-responsive design improvements
- [ ] Export functionality for reports
- [ ] Integration with brokerage APIs
- [ ] Advanced charting and visualization
- [ ] Multi-user support with authentication

---

**Disclaimer**: This tool is for educational and informational purposes only. Always consult with qualified financial advisors before making investment decisions.
