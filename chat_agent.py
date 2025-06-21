# chat_agent.py

import os
import time
import functools
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
import re
from colorama import Fore, Style, init
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.agents.agent_types import AgentType
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache
from tools.fundamentals import get_fundamentals
from tools.technicals import get_technicals
from tools.strategy import make_strategy_decision
from tools.forecast import get_forecast, train_model
from tools.financial_analysis import get_financial_insights, store_financial_analysis_in_vector_db, get_industry_comparison
from tools.portfolio import get_optimal_portfolio, analyze_existing_portfolio, get_efficient_frontier, get_portfolio_rebalance, stress_test_portfolio
from tools.user_profile import create_user_profile, get_user_profile, update_user_preferences, add_investment_goal, get_personalized_recommendations
from tools.economic_factors import get_economic_indicators, get_sector_performance, analyze_interest_rate_sensitivity, analyze_inflation_sensitivity, get_macroeconomic_context

# Set up LLM caching to avoid redundant API calls
set_llm_cache(InMemoryCache())

# Load .env variables
load_dotenv()

# Initialize colorama for colored terminal output
init(autoreset=True)

# Simple cache implementation for financial data
class FinanceDataCache:
    def __init__(self, ttl_minutes: int = 15):
        self.cache = {}
        self.ttl = timedelta(minutes=ttl_minutes)
    
    def get(self, key: str) -> Optional[Dict]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                return data
            # Expired data
            del self.cache[key]
        return None
    
    def set(self, key: str, data: Dict) -> None:
        self.cache[key] = (data, datetime.now())

# Initialize cache
finance_cache = FinanceDataCache()

# Rate limiter decorator
def rate_limit(calls_per_minute: int = 10):
    min_interval = 60.0 / calls_per_minute
    last_call_time = {}
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            current_time = time.time()
            
            if func_name in last_call_time:
                elapsed = current_time - last_call_time[func_name]
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
            
            result = func(*args, **kwargs)
            last_call_time[func_name] = time.time()
            return result
        return wrapper
    return decorator

# Wrap each tool using LangChain's @tool decorator with caching and rate limiting
@tool
@rate_limit(calls_per_minute=20)
def fundamentals_tool(ticker: str) -> str:
    """Returns P/E ratio, PEG ratio, EPS, and sector for a given ticker."""
    ticker = ticker.upper().strip()
    cache_key = f"fundamentals_{ticker}"
    
    # Check cache first
    cached_data = finance_cache.get(cache_key)
    if cached_data:
        return str(cached_data)
    
    # If not in cache, fetch and store
    data = get_fundamentals(ticker)
    finance_cache.set(cache_key, data)
    return str(data)

@tool
@rate_limit(calls_per_minute=20)
def technicals_tool(ticker: str) -> str:
    """Returns RSI, SMA50, SMA200, current price, and trend signals."""
    ticker = ticker.upper().strip()
    cache_key = f"technicals_{ticker}"
    
    # Check cache first
    cached_data = finance_cache.get(cache_key)
    if cached_data:
        return str(cached_data)
    
    # If not in cache, fetch and store
    data = get_technicals(ticker)
    finance_cache.set(cache_key, data)
    return str(data)

@tool
@rate_limit(calls_per_minute=10)
def strategy_tool(ticker: str) -> str:
    """Provides comprehensive investment strategy with buy/sell/hold recommendation, confidence level, 
    price targets, and detailed reasoning based on technical and fundamental analysis."""
    return get_strategy(ticker, "standard")

def get_strategy(ticker: str, investment_type: str = "standard") -> str:
    """Helper function to get strategy with different investment types"""
    ticker = ticker.upper().strip()
    cache_key = f"strategy_{ticker}_{investment_type}"
    
    # Check cache first
    cached_data = finance_cache.get(cache_key)
    if cached_data:
        return str(cached_data)
    
    # Get fresh fundamental and technical data
    fundamentals = get_fundamentals(ticker)
    technicals = get_technicals(ticker)
    
    # Apply strategy logic with specified investment type
    result = make_strategy_decision(fundamentals, technicals, investment_type)
    
    # Add ticker symbol for reference
    result["symbol"] = ticker
    
    # Cache the result
    finance_cache.set(cache_key, result)
    
    return str(result)

@tool
@rate_limit(calls_per_minute=5)
def forecast_tool(ticker: str, days: int = 5, model_type: str = "random_forest") -> str:
    """Provides price forecasts for stocks and ETFs using machine learning models.
    
    Returns predicted price movement with confidence intervals for the specified number of days
    in the future. Includes forecast strength, direction, and historical comparison.
    
    Args:
        ticker: Stock or ETF symbol (e.g., AAPL, VTI)
        days: Number of days to forecast into the future (default: 5)
        model_type: Type of model to use (default: random_forest)
            - random_forest: Good for capturing non-linear patterns (recommended default)
            - gradient_boost: Often provides the best accuracy
            - ensemble: Combines multiple models for better accuracy
            - linear: Simple and interpretable
            - ridge: Linear model with regularization
            - neural_net: Deep learning approach
            - arima: Time series statistical model
    """
    try:
        # Convert days to int if it's a string
        if isinstance(days, str):
            days = int(days)
        
        # Limit days to reasonable range
        days = max(1, min(days, 30))
        
        # Validate model type
        valid_models = ["ensemble", "random_forest", "gradient_boost", "linear", "ridge", "neural_net", "arima"]
        if model_type not in valid_models:
            model_type = "random_forest"  # Default to random_forest if invalid
        
        ticker = ticker.upper().strip()
        cache_key = f"forecast_{ticker}_{days}_{model_type}"
        
        # Check cache first
        cached_data = finance_cache.get(cache_key)
        if cached_data:
            return str(cached_data)
        
        # Check if model exists and train if needed
        from tools.forecast import PriceForecaster
        forecaster = PriceForecaster(ticker, model_type)
        if forecaster.model is None:
            print(f"No model found for {ticker}, training a new one...")
            
            # First, try to get historical data to check if we have enough
            import yfinance as yf
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1y")
                if len(hist) < 100:
                    print(f"Warning: Limited historical data for {ticker} ({len(hist)} samples)")
                    if len(hist) < 30:
                        return str({
                            "error": f"Insufficient historical data for {ticker} (only {len(hist)} samples available). Need at least 30 days of data for forecasting.",
                            "ticker": ticker
                        })
            except Exception as e:
                print(f"Error checking historical data: {e}")
            
            # Train the model with appropriate parameters
            train_result = train_model(ticker, model_type=model_type, forecast_days=days)
            if "error" in train_result:
                return str({"error": f"Error training model: {train_result['error']}", "ticker": ticker})
            
            # For newly trained models, add a note to the result
            print(f"Successfully trained new model for {ticker}")
        
        # Get forecast
        result = get_forecast(ticker, model_type=model_type, forecast_days=days)
        
        # Add ticker symbol for reference
        if "ticker" not in result:
            result["ticker"] = ticker
        
        # Cache the result
        finance_cache.set(cache_key, result)
        
        return str(result)
    except Exception as e:
        return str({"error": f"Error generating forecast: {str(e)}", "ticker": ticker})

@tool
@rate_limit(calls_per_minute=3)
def train_forecast_model(ticker: str, model_type: str = "random_forest") -> str:
    """Trains a new forecasting model for a specific stock or ETF.
    
    This tool is useful when you want to create a more accurate model for a specific ticker
    before generating forecasts. Training a new model can take some time but provides
    better accuracy.
    
    Args:
        ticker: Stock or ETF symbol (e.g., AAPL, VTI)
        model_type: Type of model to train (default: ensemble)
            - ensemble: Combines multiple models for better accuracy
            - random_forest: Good for capturing non-linear patterns
            - gradient_boost: Often provides the best accuracy
            - linear: Simple and interpretable
            - ridge: Linear model with regularization
            - neural_net: Deep learning approach
            - arima: Time series statistical model
    """
    try:
        ticker = ticker.upper().strip()
        
        # Validate model type
        valid_models = ["ensemble", "random_forest", "gradient_boost", "linear", "ridge", "neural_net", "arima"]
        if model_type not in valid_models:
            model_type = "ensemble"  # Default to ensemble if invalid
        
        # Train the model
        result = train_model(ticker, model_type=model_type)
        
        # Add ticker symbol for reference
        if "ticker" not in result:
            result["ticker"] = ticker
        
        # Clear any existing forecast cache for this ticker
        for key in list(finance_cache.cache.keys()):
            if key.startswith(f"forecast_{ticker}_"):
                del finance_cache.cache[key]
        
        return str(result)
    except Exception as e:
        return str({"error": f"Error training model: {str(e)}", "ticker": ticker})

@tool
@rate_limit(calls_per_minute=5)
def financial_insights_tool(ticker: str, query: str = None) -> str:
    """Provides comprehensive financial insights including earnings call transcripts, SEC filings, 
    financial statements, and key metrics. Can answer specific questions about a company's financials.
    
    This tool retrieves detailed financial information from multiple sources and uses semantic search
    to find the most relevant information for your query.
    
    Args:
        ticker: Stock symbol (e.g., AAPL, MSFT)
        query: Optional specific question about the company's financials (e.g., "What was revenue growth?", 
               "What did the CEO say about AI in the last earnings call?")
    """
    try:
        ticker = ticker.upper().strip()
        cache_key = f"financial_insights_{ticker}_{query}"
        
        # Check cache first
        cached_data = finance_cache.get(cache_key)
        if cached_data:
            return str(cached_data)
        
        # Get financial insights
        insights = get_financial_insights(ticker, query)
        
        # Cache the result
        finance_cache.set(cache_key, insights)
        
        return str(insights)
    except Exception as e:
        return str({"error": f"Error getting financial insights: {str(e)}", "ticker": ticker})

@tool
@rate_limit(calls_per_minute=10)
def etf_investment_tool(ticker: str) -> str:
    """Provides long-term investment analysis for ETFs, including SIP (Systematic Investment Plan) 
    recommendations, historical performance, and projected returns. Specifically designed for 
    long-term ETF investors rather than traders."""
    
    # Get fresh fundamental and technical data
    ticker = ticker.upper().strip()
    cache_key = f"strategy_{ticker}_etf_sip"
    
    # Check cache first
    cached_data = finance_cache.get(cache_key)
    if cached_data:
        return str(cached_data)
    
    try:
        # Get fresh fundamental and technical data
        fundamentals = get_fundamentals(ticker)
        technicals = get_technicals(ticker)
        
        # Ensure we have a price value for calculations
        if technicals and "current_price" not in technicals or technicals["current_price"] is None:
            # Set a default price if none is available
            if "symbol" in fundamentals:
                print(f"Warning: No price data for {fundamentals['symbol']}, using default price")
            technicals["current_price"] = 100.0  # Default price for calculations
        
        # Apply strategy logic with ETF SIP investment type
        result = make_strategy_decision(fundamentals, technicals, "etf_sip")
        
        # Add ticker symbol for reference
        result["symbol"] = ticker
        
        # Cache the result
        finance_cache.set(cache_key, result)
        
        return str(result)
    except Exception as e:
        # Return a default response if there's an error
        return str({
            "symbol": ticker,
            "action": "SIP_INVEST",
            "reason": "Default recommendation due to data processing error.",
            "confidence": 0.7,
            "timeframe": "long_term",
            "investment_plan": {
                "recommended_frequency": "monthly",
                "monthly_investment_options": {
                    "conservative": 500.0,
                    "moderate": 1000.0,
                    "aggressive": 2000.0
                },
                "estimated_annual_return": 9.5,
                "compound_growth": {
                    "5_year": 57.3,
                    "10_year": 147.1,
                    "20_year": 511.4
                }
            }
        })

# Initialize the LLM (OpenAI GPT-4o) with streaming and optimized settings
llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    streaming=True,
    max_tokens=1500,  # Increased token limit to allow for longer responses
    request_timeout=30  # Set timeout to avoid hanging
)

# Create a message history for storing conversation
message_history = ChatMessageHistory()

# Create an optimized prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the Finance Planning Agent - a structured financial research assistant focused on long-term investment. You provide accurate information about stocks and ETFs using a clear, organized approach.

    # COMMUNICATION STRUCTURE
    All your responses must follow this tag-based structure:
    - <thinking> for your reasoning, analysis, and planning
    - <action> for using financial tools and retrieving data
    - <conclusion> for your final investment advice and summary

    # TAG FLOW RULES
    You must follow this tag flow exactly:
    -> <thinking> -> <action> -> <thinking> -> <action> -> ... -> <conclusion>

    # TAG GUIDELINES
    
    <thinking>
    - Start with a numbered plan outlining your analysis strategy
    - Explain your reasoning in natural, human-like language
    - Justify which financial tools you'll use and why
    - Analyze results from previous actions before proceeding
    - Be thorough but concise
    
    <action>
    - Use exactly one financial tool per <action> block
    - Specify the tool name and parameters clearly
    - Format as: "Using [tool_name] with parameters: [parameters]"
    - Example: "Using fundamentals_tool with parameters: ticker='AAPL'"
    
    <conclusion>
    - Provide a clear, actionable investment recommendation
    - Summarize key findings from your analysis
    - Include relevant metrics that support your conclusion
    - Acknowledge limitations and risks
    - Format numbers clearly (e.g., $1.2B not $1,200,000,000)
    - Emphasize long-term perspective (5-10+ years)
    
    # TOOL SELECTION GUIDELINES
    
    Basic Analysis Tools:
    - fundamentals_tool: For P/E ratio, EPS, market cap, and sector
    - technicals_tool: For RSI, moving averages, and trend signals
    - strategy_tool: For buy/sell/hold recommendations with confidence levels
    
    Advanced Analysis Tools:
    - forecast_tool: For price forecasts and predictions
    - financial_insights_tool: For earnings calls, SEC filings, and key metrics
    - industry_comparison_tool: For peer and sector comparison
    - etf_investment_tool: For ETF analysis and SIP recommendations
    
    Portfolio Management Tools:
    - portfolio_optimization_tool: For optimal asset allocation
    - portfolio_analysis_tool: For evaluating existing portfolios
    - portfolio_rebalance_tool: For rebalancing recommendations
    - portfolio_stress_test_tool: For testing performance in stress scenarios
    
    Personalization Tools:
    - user_profile_tool, user_preferences_tool, investment_goal_tool
    - personalized_recommendations_tool
    
    Macroeconomic Tools:
    - economic_indicators_tool, sector_performance_tool
    - interest_rate_sensitivity_tool, inflation_sensitivity_tool
    - macroeconomic_context_tool
    
    # CONTENT GUIDELINES
    - Explain technical terms briefly when they might not be familiar
    - Highlight the most important metrics for decision-making
    - Consider macroeconomic factors in your recommendations
    - Personalize based on user's risk tolerance when available
    - Emphasize compound growth and dollar-cost averaging
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

@tool
@rate_limit(calls_per_minute=5)
def industry_comparison_tool(ticker: str) -> str:
    """Provides industry comparison metrics for a stock, including peer analysis, 
    sector performance, and relative valuation metrics. This helps investors understand 
    how a company performs compared to its industry peers.
    
    Args:
        ticker: Stock symbol (e.g., AAPL, MSFT)
    """
    try:
        ticker = ticker.upper().strip()
        cache_key = f"industry_comparison_{ticker}"
        
        # Check cache first
        cached_data = finance_cache.get(cache_key)
        if cached_data:
            return str(cached_data)
        
        # Get industry comparison
        comparison = get_industry_comparison(ticker)
        
        # Cache the result
        finance_cache.set(cache_key, comparison)
        
        return str(comparison)
    except Exception as e:
        return str({"error": f"Error getting industry comparison: {str(e)}", "ticker": ticker})


@tool
@rate_limit(calls_per_minute=5)
def portfolio_optimization_tool(tickers: str, period: str = "1y", max_weight: float = 0.4, min_assets: int = 2) -> str:
    """Optimizes a portfolio of stocks to maximize the Sharpe ratio (risk-adjusted return).
    
    This tool finds the optimal asset allocation that maximizes the Sharpe ratio based on
    historical performance. It provides detailed weights for each asset along with expected
    return, volatility, and Sharpe ratio metrics.
    
    Args:
        tickers: Comma-separated list of stock symbols (e.g., "AAPL,MSFT,GOOGL")
        period: Time period for historical data (default: "1y", options: "1mo", "3mo", "6mo", "1y", "2y", "5y")
        max_weight: Maximum weight for any single asset (default: 0.4 or 40%)
        min_assets: Minimum number of assets with non-zero weights (default: 2)
    """
    try:
        # Parse tickers from string to list
        ticker_list = [t.strip().upper() for t in tickers.split(',')]
        
        # Validate inputs
        if len(ticker_list) < min_assets:
            return str({"error": f"Need at least {min_assets} tickers for proper diversification", "tickers": tickers})
        
        if max_weight <= 0 or max_weight > 1:
            max_weight = 0.4  # Reset to default if invalid
        
        # Check cache
        cache_key = f"portfolio_optimization_{'_'.join(ticker_list)}_{period}_{max_weight}_{min_assets}"
        cached_data = finance_cache.get(cache_key)
        if cached_data:
            return str(cached_data)
        
        # Get optimal portfolio with diversification constraints
        result = get_optimal_portfolio(ticker_list, period=period)
        
        # If there was an error in the optimization, try to provide a helpful message
        if "error" in result:
            # Check if we can get historical data at all
            from tools.portfolio import PortfolioManager
            pm = PortfolioManager()
            prices = pm.get_historical_data(ticker_list, period)
            
            if prices.empty:
                return str({"error": f"Could not retrieve historical data for the specified tickers. Please check that all symbols are valid.", "tickers": ticker_list})
            
            # If we have data but optimization failed, return the original error
            return str(result)
        
        # Cache the result
        finance_cache.set(cache_key, result)
        
        return str(result)
    except Exception as e:
        return str({"error": f"Error optimizing portfolio: {str(e)}", "tickers": tickers})

@tool
@rate_limit(calls_per_minute=5)
def portfolio_analysis_tool(tickers: str, weights: str, period: str = "1y") -> str:
    """Analyzes an existing portfolio with detailed performance metrics.
    
    This tool evaluates the performance of a portfolio with specified weights, providing
    metrics like expected return, volatility, Sharpe ratio, and improvement potential.
    
    Args:
        tickers: Comma-separated list of stock symbols (e.g., "AAPL,MSFT,GOOGL")
        weights: Comma-separated list of weights corresponding to each ticker (e.g., "0.4,0.3,0.3")
        period: Time period for historical data (default: "1y")
    """
    try:
        # Parse tickers and weights
        ticker_list = [t.strip().upper() for t in tickers.split(',')]
        weight_list = [float(w.strip()) for w in weights.split(',')]
        
        # Validate inputs
        if len(ticker_list) != len(weight_list):
            return str({"error": "Number of tickers must match number of weights", "tickers": ticker_list, "weights": weight_list})
        
        # Normalize weights if they don't sum to 1
        weight_sum = sum(weight_list)
        if abs(weight_sum - 1.0) > 0.01:
            weight_list = [w / weight_sum for w in weight_list]
        
        # Check cache
        cache_key = f"portfolio_analysis_{'_'.join(ticker_list)}_{period}"
        cached_data = finance_cache.get(cache_key)
        if cached_data:
            return str(cached_data)
        
        # Analyze portfolio
        result = analyze_existing_portfolio(ticker_list, weight_list, period=period)
        
        # Cache the result
        finance_cache.set(cache_key, result)
        
        return str(result)
    except Exception as e:
        return str({"error": f"Error analyzing portfolio: {str(e)}", "tickers": tickers})

@tool
@rate_limit(calls_per_minute=5)
def portfolio_rebalance_tool(tickers: str, weights: str, optimize: bool = True, period: str = "1y") -> str:
    """Provides recommendations for rebalancing a portfolio.
    
    This tool analyzes an existing portfolio and suggests trades to optimize performance
    or rebalance to target weights. It provides detailed trade recommendations and
    improvement metrics.
    
    Args:
        tickers: Comma-separated list of stock symbols (e.g., "AAPL,MSFT,GOOGL")
        weights: Comma-separated list of weights corresponding to each ticker (e.g., "0.4,0.3,0.3")
        optimize: Whether to optimize the portfolio (True) or just rebalance to target weights (False)
        period: Time period for historical data (default: "1y")
    """
    try:
        # Parse tickers and weights
        ticker_list = [t.strip().upper() for t in tickers.split(',')]
        weight_list = [float(w.strip()) for w in weights.split(',')]
        
        # Validate inputs
        if len(ticker_list) != len(weight_list):
            return str({"error": "Number of tickers must match number of weights", "tickers": ticker_list, "weights": weight_list})
        
        # Normalize weights if they don't sum to 1
        weight_sum = sum(weight_list)
        if abs(weight_sum - 1.0) > 0.01:
            weight_list = [w / weight_sum for w in weight_list]
        
        # Check cache
        cache_key = f"portfolio_rebalance_{'_'.join(ticker_list)}_{optimize}_{period}"
        cached_data = finance_cache.get(cache_key)
        if cached_data:
            return str(cached_data)
        
        # Get rebalance recommendations
        result = get_portfolio_rebalance(ticker_list, weight_list, optimize=optimize, period=period)
        
        # Cache the result
        finance_cache.set(cache_key, result)
        
        return str(result)
    except Exception as e:
        return str({"error": f"Error generating rebalance recommendations: {str(e)}", "tickers": tickers})

@tool
@rate_limit(calls_per_minute=3)
def portfolio_stress_test_tool(tickers: str, weights: str, scenarios: str = "financial_crisis_2008,covid_crash_2020") -> str:
    """Tests how a portfolio would perform under historical stress scenarios.
    
    This tool simulates how a portfolio would have performed during historical market
    stress events, providing metrics like drawdown, return, and worst day performance.
    
    Args:
        tickers: Comma-separated list of stock symbols (e.g., "AAPL,MSFT,GOOGL")
        weights: Comma-separated list of weights corresponding to each ticker (e.g., "0.4,0.3,0.3")
        scenarios: Comma-separated list of scenarios to test (default: "financial_crisis_2008,covid_crash_2020")
                  Options: financial_crisis_2008, covid_crash_2020, dot_com_bubble, black_monday,
                  taper_tantrum, china_slowdown_2015, inflation_scare_2022
    """
    try:
        # Parse tickers, weights, and scenarios
        ticker_list = [t.strip().upper() for t in tickers.split(',')]
        weight_list = [float(w.strip()) for w in weights.split(',')]
        scenario_list = [s.strip() for s in scenarios.split(',')]
        
        # Validate inputs
        if len(ticker_list) != len(weight_list):
            return str({"error": "Number of tickers must match number of weights", "tickers": ticker_list, "weights": weight_list})
        
        # Normalize weights if they don't sum to 1
        weight_sum = sum(weight_list)
        if abs(weight_sum - 1.0) > 0.01:
            weight_list = [w / weight_sum for w in weight_list]
        
        # Check cache
        cache_key = f"portfolio_stress_test_{'_'.join(ticker_list)}_{'_'.join(scenario_list)}"
        cached_data = finance_cache.get(cache_key)
        if cached_data:
            return str(cached_data)
        
        # Perform stress test
        result = stress_test_portfolio(ticker_list, weight_list, scenarios=scenario_list)
        
        # Cache the result
        finance_cache.set(cache_key, result)
        
        return str(result)
    except Exception as e:
        return str({"error": f"Error performing stress test: {str(e)}", "tickers": tickers})

@tool
@rate_limit(calls_per_minute=5)
def user_profile_tool(name: str, email: str = None) -> str:
    """Creates a new user profile for personalized investment recommendations.
    
    This tool creates a user profile with default preferences that can be updated later.
    The profile is used to provide personalized investment recommendations based on
    risk tolerance, investment horizon, and other factors.
    
    Args:
        name: User's name
        email: User's email (optional)
    """
    try:
        # Create user profile
        result = create_user_profile(name, email)
        return str(result)
    except Exception as e:
        return str({"error": f"Error creating user profile: {str(e)}", "name": name})

@tool
@rate_limit(calls_per_minute=5)
def user_preferences_tool(user_id: str, risk_tolerance: str = None, investment_horizon: str = None, 
                         investment_experience: str = None) -> str:
    """Updates user preferences for personalized investment recommendations.
    
    This tool updates a user's investment profile preferences, which are used to provide
    personalized investment recommendations.
    
    Args:
        user_id: User ID from user_profile_tool
        risk_tolerance: Risk tolerance level (optional, options: "conservative", "moderate", "aggressive")
        investment_horizon: Investment time horizon (optional, options: "short_term", "medium_term", "long_term")
        investment_experience: Investment experience level (optional, options: "beginner", "intermediate", "advanced")
    """
    try:
        # Build preferences dictionary
        preferences = {"investment_profile": {}}
        if risk_tolerance:
            preferences["investment_profile"]["risk_tolerance"] = risk_tolerance
        if investment_horizon:
            preferences["investment_profile"]["investment_horizon"] = investment_horizon
        if investment_experience:
            preferences["investment_profile"]["investment_experience"] = investment_experience
        
        # Update preferences
        result = update_user_preferences(user_id, preferences)
        return str(result)
    except Exception as e:
        return str({"error": f"Error updating user preferences: {str(e)}", "user_id": user_id})

@tool
@rate_limit(calls_per_minute=5)
def investment_goal_tool(user_id: str, name: str, target_amount: float, target_date: str, 
                        current_amount: float = 0.0, priority: int = 1) -> str:
    """Adds an investment goal for a user.
    
    This tool adds a financial goal for a user, such as retirement, home purchase,
    or education funding. Goals are tracked and used to provide personalized
    investment recommendations.
    
    Args:
        user_id: User ID from user_profile_tool
        name: Goal name (e.g., "Retirement", "House Down Payment")
        target_amount: Target amount to reach
        target_date: Target date to reach the goal (YYYY-MM-DD)
        current_amount: Current amount saved (default: 0)
        priority: Goal priority (1-5, 1 being highest)
    """
    try:
        # Add investment goal
        result = add_investment_goal(user_id, name, target_amount, target_date, current_amount, priority)
        return str(result)
    except Exception as e:
        return str({"error": f"Error adding investment goal: {str(e)}", "user_id": user_id})

@tool
@rate_limit(calls_per_minute=5)
def personalized_recommendations_tool(user_id: str) -> str:
    """Gets personalized investment recommendations based on user profile.
    
    This tool provides personalized investment recommendations based on a user's
    risk tolerance, investment horizon, and other preferences. It includes
    asset allocation, suggested ETFs, and investment strategy recommendations.
    
    Args:
        user_id: User ID from user_profile_tool
    """
    try:
        # Get personalized recommendations
        result = get_personalized_recommendations(user_id)
        return str(result)
    except Exception as e:
        return str({"error": f"Error getting personalized recommendations: {str(e)}", "user_id": user_id})

@tool
@rate_limit(calls_per_minute=5)
def economic_indicators_tool(period: str = "6mo") -> str:
    """Gets economic indicators data including interest rates, inflation, economic growth, and more.
    
    This tool provides comprehensive economic indicators data that can be used to
    understand the macroeconomic environment and its potential impact on investments.
    
    Args:
        period: Time period for historical data (default: "6mo", options: "1mo", "3mo", "6mo", "1y", "2y", "5y")
    """
    try:
        # Check cache
        cache_key = f"economic_indicators_{period}"
        cached_data = finance_cache.get(cache_key)
        if cached_data:
            return str(cached_data)
        
        # Get economic indicators
        result = get_economic_indicators(period)
        
        # Cache the result
        finance_cache.set(cache_key, result)
        
        return str(result)
    except Exception as e:
        return str({"error": f"Error getting economic indicators: {str(e)}"})

@tool
@rate_limit(calls_per_minute=5)
def sector_performance_tool(period: str = "6mo") -> str:
    """Gets sector performance data including returns, volatility, and relative strength.
    
    This tool provides performance metrics for different market sectors, which can be
    used to understand sector trends and identify potential investment opportunities.
    
    Args:
        period: Time period for historical data (default: "6mo", options: "1mo", "3mo", "6mo", "1y", "2y", "5y")
    """
    try:
        # Check cache
        cache_key = f"sector_performance_{period}"
        cached_data = finance_cache.get(cache_key)
        if cached_data:
            return str(cached_data)
        
        # Get sector performance
        result = get_sector_performance(period)
        
        # Cache the result
        finance_cache.set(cache_key, result)
        
        return str(result)
    except Exception as e:
        return str({"error": f"Error getting sector performance: {str(e)}"})

@tool
@rate_limit(calls_per_minute=5)
def interest_rate_sensitivity_tool(ticker: str, period: str = "1y") -> str:
    """Analyzes a stock's sensitivity to interest rate changes.
    
    This tool provides a detailed analysis of how a stock has historically responded
    to changes in interest rates, including correlation, regression analysis, and
    interpretation of the results.
    
    Args:
        ticker: Stock symbol (e.g., "AAPL", "MSFT")
        period: Time period for historical data (default: "1y", options: "6mo", "1y", "2y", "5y")
    """
    try:
        ticker = ticker.upper().strip()
        
        # Check cache
        cache_key = f"interest_rate_sensitivity_{ticker}_{period}"
        cached_data = finance_cache.get(cache_key)
        if cached_data:
            return str(cached_data)
        
        # Analyze interest rate sensitivity
        result = analyze_interest_rate_sensitivity(ticker, period)
        
        # Cache the result
        finance_cache.set(cache_key, result)
        
        return str(result)
    except Exception as e:
        return str({"error": f"Error analyzing interest rate sensitivity: {str(e)}", "ticker": ticker})

@tool
@rate_limit(calls_per_minute=5)
def inflation_sensitivity_tool(ticker: str, period: str = "1y") -> str:
    """Analyzes a stock's sensitivity to inflation.
    
    This tool provides a detailed analysis of how a stock has historically responded
    to changes in inflation indicators, including correlation, regression analysis,
    and interpretation of the results.
    
    Args:
        ticker: Stock symbol (e.g., "AAPL", "MSFT")
        period: Time period for historical data (default: "1y", options: "6mo", "1y", "2y", "5y")
    """
    try:
        ticker = ticker.upper().strip()
        
        # Check cache
        cache_key = f"inflation_sensitivity_{ticker}_{period}"
        cached_data = finance_cache.get(cache_key)
        if cached_data:
            return str(cached_data)
        
        # Analyze inflation sensitivity
        result = analyze_inflation_sensitivity(ticker, period)
        
        # Cache the result
        finance_cache.set(cache_key, result)
        
        return str(result)
    except Exception as e:
        return str({"error": f"Error analyzing inflation sensitivity: {str(e)}", "ticker": ticker})

@tool
@rate_limit(calls_per_minute=3)
def macroeconomic_context_tool(ticker: str, period: str = "1y") -> str:
    """Gets comprehensive macroeconomic context for a stock.
    
    This tool provides a complete macroeconomic picture for a stock, including
    interest rate sensitivity, inflation sensitivity, sector performance, and
    relevant economic indicators. It also includes a summary of the macroeconomic
    context and its implications for the stock.
    
    Args:
        ticker: Stock symbol (e.g., "AAPL", "MSFT")
        period: Time period for historical data (default: "1y", options: "6mo", "1y", "2y", "5y")
    """
    try:
        ticker = ticker.upper().strip()
        
        # Check cache
        cache_key = f"macroeconomic_context_{ticker}_{period}"
        cached_data = finance_cache.get(cache_key)
        if cached_data:
            return str(cached_data)
        
        # Get macroeconomic context
        result = get_macroeconomic_context(ticker, period)
        
        # Cache the result
        finance_cache.set(cache_key, result)
        
        return str(result)
    except Exception as e:
        return str({"error": f"Error getting macroeconomic context: {str(e)}", "ticker": ticker})

# Initialize LangChain agent with tools
tools = [
    fundamentals_tool, 
    technicals_tool, 
    strategy_tool, 
    etf_investment_tool, 
    forecast_tool, 
    train_forecast_model, 
    financial_insights_tool, 
    industry_comparison_tool,
    # Portfolio management tools
    portfolio_optimization_tool,
    portfolio_analysis_tool,
    portfolio_rebalance_tool,
    portfolio_stress_test_tool,
    # User personalization tools
    user_profile_tool,
    user_preferences_tool,
    investment_goal_tool,
    personalized_recommendations_tool,
    # Macroeconomic context tools
    economic_indicators_tool,
    sector_performance_tool,
    interest_rate_sensitivity_tool,
    inflation_sensitivity_tool,
    macroeconomic_context_tool
]
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=False,  # Hide verbose output for cleaner CLI
    handle_parsing_errors=True,  # More robust error handling
    max_iterations=20,  # Increased iterations for complex portfolio queries
    max_execution_time=120  # Increased maximum execution time to 120 seconds
)

# Wrap the executor with message history
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# Helper functions for formatting output
def format_stock_data(data_str):
    """Format stock data from string representation to readable format"""
    try:
        # Extract data from string representation of dict
        data_str = data_str.replace("'", "\"")
        data = json.loads(data_str)
        
        if 'symbol' not in data and 'ticker' not in data:
            return data_str
            
        # Format based on data type
        if 'pe_ratio' in data:  # Fundamentals
            return format_fundamentals(data)
        elif 'current_price' in data and 'action' not in data and 'predicted_return' not in data:  # Technicals
            return format_technicals(data)
        elif 'action' in data:  # Strategy
            return format_strategy(data)
        elif 'predicted_return' in data:  # Forecast
            return format_forecast(data)
        else:
            return data_str
    except Exception as e:
        return f"Error formatting data: {str(e)}\n{data_str}"

def format_fundamentals(data):
    """Format fundamental data in a readable way"""
    result = []
    result.append(f"{Fore.CYAN}===== {data['symbol']} Fundamentals ====={Style.RESET_ALL}")
    
    if 'error' in data:
        result.append(f"{Fore.RED}Error: {data['error']}{Style.RESET_ALL}")
        return "\n".join(result)
        
    # Format financial metrics
    if data.get('pe_ratio'):
        result.append(f"P/E Ratio: {Fore.YELLOW}{data['pe_ratio']}{Style.RESET_ALL}")
    if data.get('eps'):
        result.append(f"EPS: {Fore.YELLOW}${data['eps']}{Style.RESET_ALL}")
    if data.get('market_cap'):
        market_cap = data['market_cap']
        if market_cap > 1_000_000_000_000:
            formatted = f"${market_cap/1_000_000_000_000:.2f}T"
        elif market_cap > 1_000_000_000:
            formatted = f"${market_cap/1_000_000_000:.2f}B"
        else:
            formatted = f"${market_cap/1_000_000:.2f}M"
        result.append(f"Market Cap: {Fore.YELLOW}{formatted}{Style.RESET_ALL}")
    if data.get('sector'):
        result.append(f"Sector: {Fore.YELLOW}{data['sector']}{Style.RESET_ALL}")
    if data.get('dividend_yield'):
        result.append(f"Dividend Yield: {Fore.YELLOW}{data['dividend_yield']}%{Style.RESET_ALL}")
        
    return "\n".join(result)

def format_technicals(data):
    """Format technical data in a readable way"""
    result = []
    result.append(f"{Fore.CYAN}===== {data['symbol']} Technicals ====={Style.RESET_ALL}")
    
    if 'error' in data:
        result.append(f"{Fore.RED}Error: {data['error']}{Style.RESET_ALL}")
        return "\n".join(result)
        
    # Format price and indicators
    if data.get('current_price'):
        result.append(f"Current Price: {Fore.GREEN}${data['current_price']}{Style.RESET_ALL}")
    if data.get('rsi'):
        rsi = data['rsi']
        color = Fore.RED if rsi > 70 else Fore.GREEN if rsi < 30 else Fore.YELLOW
        result.append(f"RSI (14): {color}{rsi}{Style.RESET_ALL}")
    if data.get('sma_50'):
        result.append(f"50-day SMA: {Fore.YELLOW}${data['sma_50']}{Style.RESET_ALL}")
    if data.get('sma_200'):
        result.append(f"200-day SMA: {Fore.YELLOW}${data['sma_200']}{Style.RESET_ALL}")
        
    # Add trend signals
    if data.get('price_above_sma_50') is not None and data.get('price_above_sma_200') is not None:
        if data['price_above_sma_50'] and data['price_above_sma_200']:
            trend = f"{Fore.GREEN}Bullish (above both SMAs){Style.RESET_ALL}"
        elif data['price_above_sma_50'] and not data['price_above_sma_200']:
            trend = f"{Fore.YELLOW}Mixed (above 50-day, below 200-day){Style.RESET_ALL}"
        elif not data['price_above_sma_50'] and data['price_above_sma_200']:
            trend = f"{Fore.YELLOW}Mixed (below 50-day, above 200-day){Style.RESET_ALL}"
        else:
            trend = f"{Fore.RED}Bearish (below both SMAs){Style.RESET_ALL}"
        result.append(f"Trend: {trend}")
        
    return "\n".join(result)

def format_strategy(data):
    """Format strategy data in a readable way"""
    result = []
    symbol = data.get('symbol', 'UNKNOWN')
    action = data.get('action', 'UNKNOWN')
    confidence = data.get('confidence', 0)
    reason = data.get('reason', 'No reason provided.')
    timeframe = data.get('timeframe', 'unknown')
    investment_type = data.get('investment_type', 'standard')
    
    # Determine color based on action
    action_color = Fore.YELLOW  # Default
    if action in ["BUY", "BUY_LONG", "SIP_INVEST"]:
        action_color = Fore.GREEN
    elif action in ["SELL", "AVOID"]:
        action_color = Fore.RED
    elif action == "HOLD_BULLISH":
        action_color = Fore.CYAN
    elif action == "HOLD_BEARISH":
        action_color = Fore.MAGENTA
    
    # Header based on investment type
    if investment_type == "etf_sip":
        result.append(f"{Fore.CYAN}{'='*20} {symbol} ETF INVESTMENT PLAN {'='*20}{Style.RESET_ALL}")
    elif investment_type == "long_term":
        result.append(f"{Fore.CYAN}{'='*20} {symbol} LONG-TERM INVESTMENT {'='*20}{Style.RESET_ALL}")
    else:
        result.append(f"{Fore.CYAN}{'='*20} {symbol} STRATEGY {'='*20}{Style.RESET_ALL}")
    
    # Action and confidence
    result.append(f"Recommendation: {action_color}{action}{Style.RESET_ALL} with {Fore.YELLOW}{confidence*100:.0f}%{Style.RESET_ALL} confidence")
    result.append(f"Timeframe: {Fore.YELLOW}{timeframe.replace('_', ' ').title()}{Style.RESET_ALL}")
    
    # Reason
    result.append(f"Reason: {reason}")
    
    # Historical performance if available
    if 'historical_performance' in data and data['historical_performance']:
        perf = data['historical_performance']
        result.append(f"\n{Fore.CYAN}Historical Performance:{Style.RESET_ALL}")
        for period, return_val in perf.items():
            period_str = period.replace('_', ' ').title()
            result.append(f"{period_str}: {Fore.GREEN}+{return_val}%{Style.RESET_ALL}")
    
    # Investment plan for ETFs
    if 'investment_plan' in data and data['investment_plan']:
        plan = data['investment_plan']
        result.append(f"\n{Fore.CYAN}SIP Investment Plan:{Style.RESET_ALL}")
        
        if 'recommended_frequency' in plan:
            result.append(f"Recommended Frequency: {Fore.YELLOW}{plan['recommended_frequency'].title()}{Style.RESET_ALL}")
        
        if 'monthly_investment_options' in plan:
            options = plan['monthly_investment_options']
            result.append(f"Recommended Monthly Investment:")
            result.append(f"  Conservative: {Fore.GREEN}${options['conservative']:.2f}{Style.RESET_ALL}")
            result.append(f"  Moderate: {Fore.GREEN}${options['moderate']:.2f}{Style.RESET_ALL}")
            result.append(f"  Aggressive: {Fore.GREEN}${options['aggressive']:.2f}{Style.RESET_ALL}")
        
        if 'estimated_annual_return' in plan:
            result.append(f"Estimated Annual Return: {Fore.GREEN}{plan['estimated_annual_return']}%{Style.RESET_ALL}")
        
        if 'compound_growth' in plan:
            growth = plan['compound_growth']
            result.append(f"Projected Growth (with consistent investment):")
            result.append(f"  5 Years: {Fore.GREEN}+{growth['5_year']}%{Style.RESET_ALL}")
            result.append(f"  10 Years: {Fore.GREEN}+{growth['10_year']}%{Style.RESET_ALL}")
            result.append(f"  20 Years: {Fore.GREEN}+{growth['20_year']}%{Style.RESET_ALL}")
    
    # Price targets for standard investment
    elif 'price_targets' in data and data['price_targets'] and investment_type == "standard":
        targets = data['price_targets']
        result.append(f"\n{Fore.CYAN}Price Targets:{Style.RESET_ALL}")
        
        if targets.get('entry'):
            result.append(f"Entry: {Fore.GREEN}${targets['entry']}{Style.RESET_ALL}")
        
        if targets.get('stop_loss'):
            result.append(f"Stop Loss: {Fore.RED}${targets['stop_loss']}{Style.RESET_ALL}")
        
        if targets.get('take_profit'):
            result.append(f"Take Profit: {Fore.GREEN}${targets['take_profit']}{Style.RESET_ALL}")
    
    # Signals
    if 'signals' in data and data['signals']:
        result.append(f"\n{Fore.CYAN}Key Signals:{Style.RESET_ALL}")
        for i, signal in enumerate(data['signals'][:5], 1):  # Show top 5 signals
            result.append(f"{i}. {signal}")
    
    # Scores for standard investment
    if 'buy_score' in data and 'sell_score' in data and investment_type == "standard":
        buy_score = data['buy_score']
        sell_score = data['sell_score']
        result.append(f"\nSignal Strength: Buy ({Fore.GREEN}{buy_score:.1f}{Style.RESET_ALL}) vs Sell ({Fore.RED}{sell_score:.1f}{Style.RESET_ALL})")
    
    return "\n".join(result)

def format_forecast(data):
    """Format forecast data in a readable way"""
    result = []
    ticker = data.get('ticker', 'UNKNOWN')
    
    # Header
    result.append(f"{Fore.CYAN}{'='*20} {ticker} PRICE FORECAST {'='*20}{Style.RESET_ALL}")
    
    # Current price and forecast days
    current_price = data.get('current_price')
    forecast_days = data.get('forecast_days', 5)
    if current_price:
        result.append(f"Current Price: {Fore.GREEN}${current_price}{Style.RESET_ALL}")
    result.append(f"Forecast Period: {Fore.YELLOW}{forecast_days} days{Style.RESET_ALL}")
    
    # Prediction
    predicted_return = data.get('predicted_return')
    predicted_price = data.get('predicted_price')
    if predicted_return is not None and predicted_price is not None:
        return_color = Fore.GREEN if predicted_return > 0 else Fore.RED
        result.append(f"Predicted Move: {return_color}{'+' if predicted_return > 0 else ''}{predicted_return}%{Style.RESET_ALL}")
        result.append(f"Predicted Price: {Fore.GREEN}${predicted_price}{Style.RESET_ALL}")
    
    # Forecast metrics
    if 'forecast_metrics' in data:
        metrics = data['forecast_metrics']
        direction = metrics.get('direction', 'neutral')
        strength = metrics.get('strength', 'unknown')
        z_score = metrics.get('z_score')
        
        direction_color = Fore.GREEN if direction == 'bullish' else Fore.RED if direction == 'bearish' else Fore.YELLOW
        result.append(f"Forecast: {direction_color}{direction.title()}{Style.RESET_ALL} ({strength.title()} confidence)")
        if z_score is not None:
            result.append(f"Z-Score: {Fore.YELLOW}{z_score}{Style.RESET_ALL}")
    
    # Confidence intervals
    if 'confidence_intervals' in data:
        intervals = data['confidence_intervals']
        result.append(f"\n{Fore.CYAN}Confidence Intervals:{Style.RESET_ALL}")
        
        if '90%' in intervals:
            interval_90 = intervals['90%']
            result.append(f"90% Confidence: {Fore.GREEN}${interval_90['lower_price']}{Style.RESET_ALL} to {Fore.GREEN}${interval_90['upper_price']}{Style.RESET_ALL}")
            result.append(f"  Return Range: {Fore.YELLOW}{interval_90['lower_return']}%{Style.RESET_ALL} to {Fore.YELLOW}{interval_90['upper_return']}%{Style.RESET_ALL}")
        
        if '95%' in intervals:
            interval_95 = intervals['95%']
            result.append(f"95% Confidence: {Fore.GREEN}${interval_95['lower_price']}{Style.RESET_ALL} to {Fore.GREEN}${interval_95['upper_price']}{Style.RESET_ALL}")
    
    # Historical metrics
    if 'historical_metrics' in data:
        hist = data['historical_metrics']
        result.append(f"\n{Fore.CYAN}Historical Performance:{Style.RESET_ALL}")
        if 'avg_return_5d' in hist:
            result.append(f"Avg 5-Day Return: {Fore.YELLOW}{hist['avg_return_5d']}%{Style.RESET_ALL}")
        if 'avg_return_10d' in hist:
            result.append(f"Avg 10-Day Return: {Fore.YELLOW}{hist['avg_return_10d']}%{Style.RESET_ALL}")
        if 'return_volatility' in hist:
            result.append(f"Return Volatility: {Fore.YELLOW}{hist['return_volatility']}%{Style.RESET_ALL}")
    
    # Model info
    if 'model_info' in data:
        model = data['model_info']
        result.append(f"\n{Fore.CYAN}Model Information:{Style.RESET_ALL}")
        result.append(f"Model Type: {Fore.YELLOW}{model.get('model_type', 'unknown')}{Style.RESET_ALL}")
        result.append(f"Last Trained: {Fore.YELLOW}{model.get('last_trained', 'unknown')}{Style.RESET_ALL}")
        
        # Feature importance
        if 'feature_importance' in model and model['feature_importance']:
            result.append(f"\n{Fore.CYAN}Top Features:{Style.RESET_ALL}")
            for i, (feature, importance) in enumerate(list(model['feature_importance'].items())[:5]):
                result.append(f"{i+1}. {feature}: {Fore.YELLOW}{importance}{Style.RESET_ALL}")
    
    return "\n".join(result)

def format_agent_response(response):
    """Format the agent's response for better readability"""
    # Highlight action recommendations
    for action in ["BUY", "SELL", "HOLD", "HOLD_BULLISH", "HOLD_BEARISH", "BUY_LONG", "AVOID", "SIP_INVEST"]:
        color = Fore.YELLOW
        if action in ["BUY", "BUY_LONG", "SIP_INVEST"]:
            color = Fore.GREEN
        elif action in ["SELL", "AVOID"]:
            color = Fore.RED
        response = re.sub(f"\\b{action}\\b", f"{color}{action}{Style.RESET_ALL}", response)
    
    # Add formatting to stock tickers
    response = re.sub(r'\b([A-Z]{2,5})\b', f"{Fore.CYAN}\\1{Style.RESET_ALL}", response)
    
    # Add formatting to dollar amounts
    response = re.sub(r'\$(\d+(\.\d+)?)', f"{Fore.GREEN}$\\1{Style.RESET_ALL}", response)
    
    # Add formatting to percentages
    response = re.sub(r'(\d+(\.\d+)?)%', f"{Fore.YELLOW}\\1%{Style.RESET_ALL}", response)
    
    # Highlight confidence levels
    response = re.sub(r'\b(confidence:?\s*)([\d\.]+)%?', f"\\1{Fore.YELLOW}\\2%{Style.RESET_ALL}", response, flags=re.IGNORECASE)
    
    # Highlight price targets
    for target in ["entry", "stop loss", "take profit", "target", "monthly investment", "conservative", "moderate", "aggressive"]:
        response = re.sub(f"\\b({target}:?\\s*)(\\$?\\d+\\.?\\d*)", f"{Fore.CYAN}\\1{Fore.GREEN}\\2{Style.RESET_ALL}", response, flags=re.IGNORECASE)
    
    # Highlight SIP and investment terms
    for term in ["sip", "systematic investment plan", "dollar cost averaging", "dca", "compound growth", "long-term"]:
        response = re.sub(f"\\b({term})\\b", f"{Fore.CYAN}\\1{Style.RESET_ALL}", response, flags=re.IGNORECASE)
    
    # Highlight time periods
    for period in ["5 years", "10 years", "20 years", "long term", "short term", "medium term"]:
        response = re.sub(f"\\b({period})\\b", f"{Fore.YELLOW}\\1{Style.RESET_ALL}", response, flags=re.IGNORECASE)
    
    # Highlight forecast terms
    for term in ["forecast", "prediction", "projected", "bullish", "bearish", "confidence interval"]:
        response = re.sub(f"\\b({term})\\b", f"{Fore.CYAN}\\1{Style.RESET_ALL}", response, flags=re.IGNORECASE)
    
    return response

# Command-line chat loop
if __name__ == "__main__":
    print(f"\n{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}💬 Welcome to your Financial Research Agent{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Type 'exit' to quit{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}\n")

    while True:
        user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}")
        if user_input.lower() in ["exit", "quit"]:
            print(f"\n{Fore.CYAN}👋 Goodbye!{Style.RESET_ALL}\n")
            break

        print(f"{Fore.YELLOW}Processing...{Style.RESET_ALL}")
        
        try:
            # Create a custom callback handler to intercept and format tool outputs
            class ToolOutputFormatter:
                def __init__(self):
                    self.tool_outputs = []
                
                def on_tool_start(self, tool_name, input_str):
                    # Show which tool is being used and why
                    tool_descriptions = {
                        "fundamentals_tool": "Getting fundamental financial data",
                        "technicals_tool": "Analyzing technical indicators",
                        "strategy_tool": "Formulating investment strategy",
                        "etf_investment_tool": "Creating ETF investment plan",
                        "forecast_tool": "Generating price forecast",
                        "train_forecast_model": "Training a new forecasting model",
                        "financial_insights_tool": "Retrieving comprehensive financial insights",
                        "industry_comparison_tool": "Comparing company to industry peers and sector",
                        # Portfolio management tools
                        "portfolio_optimization_tool": "Optimizing portfolio asset allocation",
                        "portfolio_analysis_tool": "Analyzing portfolio performance metrics",
                        "portfolio_rebalance_tool": "Generating portfolio rebalancing recommendations",
                        "portfolio_stress_test_tool": "Testing portfolio in stress scenarios",
                        # User personalization tools
                        "user_profile_tool": "Creating user investment profile",
                        "user_preferences_tool": "Updating user investment preferences",
                        "investment_goal_tool": "Adding user investment goals",
                        "personalized_recommendations_tool": "Getting personalized investment recommendations",
                        # Macroeconomic context tools
                        "economic_indicators_tool": "Retrieving economic indicators data",
                        "sector_performance_tool": "Analyzing sector performance",
                        "interest_rate_sensitivity_tool": "Analyzing interest rate sensitivity",
                        "inflation_sensitivity_tool": "Analyzing inflation sensitivity",
                        "macroeconomic_context_tool": "Getting comprehensive macroeconomic context"
                    }
                    
                    description = tool_descriptions.get(tool_name, "Processing")
                    print(f"\n{Fore.MAGENTA}💡 {description}...{Style.RESET_ALL}")
                    
                    # For specific tools, provide more details
                    if tool_name == "forecast_tool":
                        try:
                            params = eval(input_str)
                            ticker = params.get("ticker", "").upper()
                            days = params.get("days", 5)
                            model_type = params.get("model_type", "random_forest")
                            print(f"{Fore.MAGENTA}   Forecasting {ticker} for {days} days using {model_type} model{Style.RESET_ALL}")
                        except:
                            pass
                    elif tool_name == "financial_insights_tool":
                        try:
                            params = eval(input_str)
                            ticker = params.get("ticker", "").upper()
                            query = params.get("query", "")
                            if query:
                                print(f"{Fore.MAGENTA}   Analyzing {ticker} with focus on: {query}{Style.RESET_ALL}")
                            else:
                                print(f"{Fore.MAGENTA}   Performing comprehensive analysis of {ticker}{Style.RESET_ALL}")
                        except:
                            pass
                    
                def on_tool_end(self, output):
                    formatted = format_stock_data(output)
                    if formatted:
                        print(f"\n{formatted}\n")
                    self.tool_outputs.append(output)
                    
                    # Add a summary of what was found
                    print(f"{Fore.MAGENTA}✓ Data retrieved successfully{Style.RESET_ALL}")
                    
                def on_tool_error(self, error):
                    print(f"\n{Fore.RED}⚠️ Tool Error: {error}{Style.RESET_ALL}\n")
            
            # Initialize the formatter
            formatter = ToolOutputFormatter()
            
            # Invoke the agent with the formatter
            response = agent_with_chat_history.invoke(
                {"input": user_input},
                {"configurable": {"session_id": "default", "callbacks": [formatter]}}
            )
            
            # Format and print the agent's response
            formatted_response = format_agent_response(response['output'])
            print(f"\n{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}🧠 Agent Response:{Style.RESET_ALL}")
            print(f"{formatted_response}\n")
            print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}\n")

        except Exception as e:
            print(f"\n{Fore.RED}⚠️ Error: {str(e)}{Style.RESET_ALL}\n")
