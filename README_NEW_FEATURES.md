# Finance Agent New Features

This document outlines the new features added to the Finance Agent to enhance its capabilities for long-term investing research.

## Table of Contents

1. [Portfolio Management](#portfolio-management)
2. [User Personalization](#user-personalization)
3. [Macroeconomic Context](#macroeconomic-context)
4. [Running the Test Scripts](#running-the-test-scripts)
5. [Integration with Chat Agent](#integration-with-chat-agent)

## Portfolio Management

The Portfolio Management module (`tools/portfolio.py`) provides advanced portfolio optimization and analysis capabilities:

### Key Features

- **Portfolio Optimization**: Find the optimal asset allocation to maximize Sharpe ratio
- **Efficient Frontier Calculation**: Generate the efficient frontier for a set of assets
- **Portfolio Analysis**: Analyze existing portfolios with detailed performance metrics
- **Rebalancing Recommendations**: Get recommendations for rebalancing a portfolio
- **Stress Testing**: Test how a portfolio would perform under historical stress scenarios

### Example Usage

```python
from tools.portfolio import (
    get_optimal_portfolio,
    analyze_existing_portfolio,
    get_efficient_frontier,
    get_portfolio_rebalance,
    stress_test_portfolio
)

# Optimize a portfolio
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
optimal_portfolio = get_optimal_portfolio(tickers, period="1y")

# Analyze an existing portfolio
current_tickers = ["AAPL", "MSFT", "GOOGL"]
current_weights = [0.4, 0.3, 0.3]
analysis = analyze_existing_portfolio(current_tickers, current_weights)

# Get rebalancing recommendations
rebalance = get_portfolio_rebalance(
    current_tickers, 
    current_weights, 
    optimize=True
)

# Perform stress testing
stress_test = stress_test_portfolio(
    current_tickers, 
    current_weights, 
    scenarios=["financial_crisis_2008", "covid_crash_2020"]
)
```

## User Personalization

The User Personalization module (`tools/user_profile.py`) enables personalized investment recommendations based on user profiles:

### Key Features

- **User Profile Management**: Create and manage user profiles
- **Investment Goals**: Track progress toward financial goals
- **Portfolio Tracking**: Manage multiple portfolios and holdings
- **Personalized Recommendations**: Get investment recommendations based on risk tolerance and investment horizon
- **Price Alerts**: Set and manage price alerts for stocks

### Example Usage

```python
from tools.user_profile import (
    create_user_profile,
    get_user_profile,
    update_user_preferences,
    add_investment_goal,
    get_personalized_recommendations
)

# Create a user profile
user = create_user_profile("John Doe", "john.doe@example.com")
user_id = user["user_id"]

# Update user preferences
preferences = {
    "investment_profile": {
        "risk_tolerance": "aggressive",
        "investment_horizon": "long_term",
        "investment_experience": "advanced"
    }
}
update_user_preferences(user_id, preferences)

# Add an investment goal
add_investment_goal(
    user_id,
    "Retirement",
    2000000.0,  # $2M target
    "2055-01-01",  # Target date
    current_amount=100000.0,
    priority=1
)

# Get personalized recommendations
recommendations = get_personalized_recommendations(user_id)
```

## Macroeconomic Context

The Macroeconomic Context module (`tools/economic_factors.py`) provides analysis of economic factors affecting investments:

### Key Features

- **Economic Indicators**: Track key economic indicators like interest rates, inflation, and economic growth
- **Sector Performance**: Analyze performance of different market sectors
- **Interest Rate Sensitivity**: Analyze a stock's sensitivity to interest rate changes
- **Inflation Sensitivity**: Analyze a stock's sensitivity to inflation
- **Comprehensive Macroeconomic Context**: Get a complete macroeconomic picture for a stock

### Example Usage

```python
from tools.economic_factors import (
    get_economic_indicators,
    get_sector_performance,
    analyze_interest_rate_sensitivity,
    analyze_inflation_sensitivity,
    get_macroeconomic_context
)

# Get economic indicators
indicators = get_economic_indicators(period="6mo")

# Get sector performance
sectors = get_sector_performance(period="1y")

# Analyze interest rate sensitivity
ir_sensitivity = analyze_interest_rate_sensitivity("AAPL", period="1y")

# Analyze inflation sensitivity
inf_sensitivity = analyze_inflation_sensitivity("AAPL", period="1y")

# Get comprehensive macroeconomic context
context = get_macroeconomic_context("AAPL", period="1y")
```

## Running the Test Scripts

Two test scripts are provided to demonstrate the new features:

### Portfolio and User Profile Test

```bash
python test_portfolio_user.py
```

This script demonstrates:
- Portfolio optimization
- Portfolio analysis
- User profile creation
- Investment goal tracking
- Personalized recommendations

### Economic Factors Test

```bash
python test_economic_factors.py
```

This script demonstrates:
- Economic indicators tracking
- Sector performance analysis
- Interest rate sensitivity analysis
- Inflation sensitivity analysis
- Comprehensive macroeconomic context

## Integration with Chat Agent

To integrate these new features with the chat agent, you can modify the `chat_agent.py` file to include the new tools in the tools list and tool descriptions.

### Example Integration

```python
# In chat_agent.py

# Import new tools
from tools.portfolio import get_optimal_portfolio, analyze_existing_portfolio
from tools.user_profile import get_personalized_recommendations
from tools.economic_factors import get_macroeconomic_context

# Add to tools list
tools = [
    # Existing tools...
    "get_optimal_portfolio",
    "analyze_existing_portfolio",
    "get_personalized_recommendations",
    "get_macroeconomic_context",
]

# Add to tool_descriptions
tool_descriptions = {
    # Existing descriptions...
    "get_optimal_portfolio": "Optimize a portfolio of stocks to maximize Sharpe ratio",
    "analyze_existing_portfolio": "Analyze an existing portfolio with detailed performance metrics",
    "get_personalized_recommendations": "Get personalized investment recommendations based on user profile",
    "get_macroeconomic_context": "Get comprehensive macroeconomic context for a stock",
}
```

## Future Enhancements

1. **Enhanced Visualization**: When ready to reintegrate visualization, consider adding interactive charts and PDF report generation.

2. **Sentiment Analysis**: Add news sentiment analysis to gauge market perception and incorporate earnings call sentiment analysis.

3. **Performance Monitoring**: Add a system to track forecast accuracy over time and implement automated model retraining based on performance.
