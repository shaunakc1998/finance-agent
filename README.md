# Finance Agent with Advanced Price Forecasting and Financial Analysis

[![Python Lint and Test](https://github.com/yourusername/finance-agent/actions/workflows/python-lint-test.yml/badge.svg)](https://github.com/yourusername/finance-agent/actions/workflows/python-lint-test.yml)
[![CodeQL](https://github.com/yourusername/finance-agent/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/yourusername/finance-agent/actions/workflows/codeql-analysis.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive financial research agent with advanced price forecasting capabilities using machine learning models and in-depth financial analysis with vector search.

## Features

- **Fundamental Analysis**: Get key financial metrics like P/E ratio, EPS, market cap, and more
- **Technical Analysis**: Access technical indicators like RSI, SMA50, SMA200, and trend signals
- **Investment Strategy**: Receive buy/sell/hold recommendations with confidence levels and price targets
- **ETF Investment Plans**: Get SIP (Systematic Investment Plan) recommendations for ETFs
- **Price Forecasting**: ML-based price predictions with confidence intervals and feature importance
- **Financial Insights**: Comprehensive financial analysis including earnings call transcripts, SEC filings, and financial statements
- **Vector Search**: Ask specific questions about a company's financials and get relevant information

## Price Forecasting Module

The price forecasting module uses machine learning models to predict future price movements for stocks and ETFs. It provides:

- Predicted price and return for a specified number of days in the future
- Confidence intervals (90% and 95%) for the prediction
- Forecast strength and direction (bullish/bearish)
- Historical comparison with average returns
- Feature importance analysis

### Available Models

- **Random Forest**: Good for capturing non-linear patterns (recommended default)
- **Gradient Boost**: Often provides the best accuracy
- **Ensemble**: Combines multiple models for better accuracy
- **Linear**: Simple and interpretable
- **Ridge**: Linear model with regularization
- **Neural Net**: Deep learning approach
- **ARIMA**: Time series statistical model

## Usage

### Chat Agent

Run the chat agent to interact with the financial research assistant:

```bash
python chat_agent.py
```

Example commands:
- `forecast AAPL 10 days random_forest`
- `forecast MSFT 20 days gradient_boost`
- `fundamentals NVDA`
- `technicals AMZN`
- `strategy TSLA`
- `etf SPY`
- `financial insights AAPL What was the revenue growth?`
- `financial insights MSFT What did the CEO say about AI?`

### Training Models

#### Train a Single Model

To train a model for a specific stock:

```bash
python auto_train_model.py AAPL random_forest 10
```

Arguments:
- Ticker symbol (required)
- Model type (optional, default: random_forest)
- Forecast days (optional, default: 10)

#### Train Multiple Models

To train models for a comprehensive list of stocks:

```bash
python run_training_background.py
```

This will start the training process in the background and log the output to a file in the `training_logs` directory.

## Scripts

- **chat_agent.py**: Main chat interface for interacting with the financial research assistant
- **auto_train_model.py**: Script to check if a model exists for a stock and train it if it doesn't
- **train_all_models.py**: Script to train models for a comprehensive list of popular stocks
- **run_training_background.py**: Script to run the model training in the background
- **test_financial_analysis.py**: Script to test the financial analysis module
- **tools/financial_analysis.py**: Module for comprehensive financial analysis with vector search
- **tools/fundamentals.py**: Module for fundamental analysis
- **tools/technicals.py**: Module for technical analysis
- **tools/strategy.py**: Module for investment strategy recommendations
- **tools/forecast.py**: Module for price forecasting

## Model Training

The system automatically checks if a model exists for a stock when a forecast is requested. If no model is found, it will train one on the fly.

For better performance, you can pre-train models for stocks you're interested in using the provided scripts.

## Forecast Accuracy

Forecasts are estimates with inherent uncertainty. The system provides confidence intervals to help understand the range of possible outcomes.

For long-term investors, short-term forecasts are less relevant. Focus on fundamentals, dividend history, and compound growth potential over 5-10+ years.

## Financial Analysis Module

The financial analysis module provides comprehensive insights into a company's financial health and performance. It includes:

- **Financial Statements**: Access to income statements, balance sheets, and cash flow statements
- **Earnings Information**: EPS estimates vs. actual, upcoming earnings dates, and historical performance
- **SEC Filings**: Recent 10-K, 10-Q, and 8-K filings with direct links
- **Earnings Call Transcripts**: Text from recent earnings calls for qualitative analysis
- **Key Financial Metrics**: Comprehensive set of valuation, profitability, growth, and debt metrics
- **Vector Search**: Ask specific questions about a company's financials and get relevant information

### Vector Database

The system uses a vector database to store and retrieve financial information:

- Financial data is processed and stored as vector embeddings
- Semantic search allows for natural language queries about a company's financials
- Information is automatically cached and refreshed weekly
- All data is stored locally, with no external API dependencies for privacy and performance

### Example Queries

- `financial insights AAPL What was the revenue growth?`
- `financial insights MSFT What did the CEO say about AI in the last earnings call?`
- `financial insights AMZN What are the debt metrics?`
- `financial insights TSLA What were the recent earnings?`

## Testing

To test the financial analysis module:

```bash
python test_financial_analysis.py AAPL
```

This will run a comprehensive test of all financial analysis features and save the results to the `test_results` directory.

## Installation

### From Source

Clone the repository and install the package:

```bash
git clone https://github.com/yourusername/finance-agent.git
cd finance-agent
pip install -e .
```

### Using pip

```bash
pip install finance-agent
```

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/finance-agent.git
   cd finance-agent
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   pip install -r requirements.txt
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

5. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

## Contributing

We welcome contributions to the Finance Agent project! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute.

### Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

### Branch Protection

The `master` branch is protected. All changes must be made through pull requests that pass all checks and receive approval.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
