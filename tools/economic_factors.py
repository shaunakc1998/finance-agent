# tools/economic_factors.py

import os
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Union, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import json
import requests
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache directory for economic data
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Economic indicator tickers
ECONOMIC_INDICATORS = {
    "interest_rates": {
        "fed_funds_rate": "^FVX",  # 5-Year Treasury Rate
        "treasury_10y": "^TNX",    # 10-Year Treasury Rate
        "treasury_30y": "^TYX"     # 30-Year Treasury Rate
    },
    "inflation": {
        "tips": "TIP",             # iShares TIPS Bond ETF
        "inflation_expectation": "RINF"  # ProShares Inflation Expectations ETF
    },
    "economic_growth": {
        "gdp_proxy": "SPY",        # S&P 500 as GDP proxy
        "industrial": "XLI",       # Industrial Sector ETF
        "consumer": "XLY"          # Consumer Discretionary ETF
    },
    "commodities": {
        "oil": "CL=F",             # Crude Oil Futures
        "gold": "GC=F",            # Gold Futures
        "natural_gas": "NG=F"      # Natural Gas Futures
    },
    "volatility": {
        "vix": "^VIX"              # CBOE Volatility Index
    }
}

# Sector ETFs for sector-specific analysis
SECTOR_ETFS = {
    "technology": "XLK",
    "healthcare": "XLV",
    "financials": "XLF",
    "consumer_discretionary": "XLY",
    "consumer_staples": "XLP",
    "energy": "XLE",
    "utilities": "XLU",
    "real_estate": "XLRE",
    "materials": "XLB",
    "industrials": "XLI",
    "communication_services": "XLC"
}

# Sector to economic indicator sensitivity mapping
SECTOR_SENSITIVITIES = {
    "technology": ["interest_rates", "economic_growth"],
    "healthcare": ["inflation"],
    "financials": ["interest_rates", "economic_growth", "volatility"],
    "consumer_discretionary": ["economic_growth", "inflation"],
    "consumer_staples": ["inflation"],
    "energy": ["commodities", "economic_growth"],
    "utilities": ["interest_rates", "inflation"],
    "real_estate": ["interest_rates", "inflation"],
    "materials": ["commodities", "economic_growth"],
    "industrials": ["economic_growth", "commodities"],
    "communication_services": ["economic_growth"]
}

class EconomicFactorsAnalyzer:
    """Economic factors analysis class"""
    
    def __init__(self, cache_ttl_days: int = 1):
        """
        Initialize the economic factors analyzer
        
        Args:
            cache_ttl_days: Cache time-to-live in days
        """
        self.cache_ttl = timedelta(days=cache_ttl_days)
    
    def get_economic_indicators(self, period: str = "1y") -> Dict:
        """
        Get economic indicators data
        
        Args:
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            Dictionary with economic indicators data
        """
        try:
            # Create a cache key based on period
            cache_key = f"economic_indicators_{period}"
            cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
            
            # Check if cache exists and is recent
            if os.path.exists(cache_file):
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if datetime.now() - file_time < self.cache_ttl:
                    logger.info(f"Using cached economic indicators data")
                    return pd.read_pickle(cache_file)
            
            # Collect all indicator tickers
            all_tickers = []
            for category in ECONOMIC_INDICATORS.values():
                all_tickers.extend(category.values())
            
            # Get data for all indicators
            data = yf.download(all_tickers, period=period, progress=False)
            
            if data.empty:
                return {"error": "No economic indicators data found"}
            
            # Extract close prices
            if len(all_tickers) == 1:
                ticker = all_tickers[0]
                indicators_data = pd.DataFrame({ticker: data['Close']})
            else:
                indicators_data = pd.DataFrame()
                for ticker in all_tickers:
                    if (ticker, 'Close') in data.columns:
                        indicators_data[ticker] = data[(ticker, 'Close')]
            
            # Organize by category
            result = {}
            for category, indicators in ECONOMIC_INDICATORS.items():
                result[category] = {}
                for name, ticker in indicators.items():
                    if ticker in indicators_data.columns:
                        # Get current value and change
                        current = indicators_data[ticker].iloc[-1]
                        start = indicators_data[ticker].iloc[0]
                        change = current - start
                        change_pct = (change / start) * 100 if start != 0 else 0
                        
                        # Calculate trend (simple moving average comparison)
                        if len(indicators_data) > 20:
                            sma20 = indicators_data[ticker].iloc[-20:].mean()
                            sma50 = indicators_data[ticker].iloc[-50:].mean() if len(indicators_data) > 50 else start
                            trend = "up" if sma20 > sma50 else "down" if sma20 < sma50 else "neutral"
                        else:
                            trend = "neutral"
                        
                        result[category][name] = {
                            "ticker": ticker,
                            "current": current,
                            "change": change,
                            "change_pct": change_pct,
                            "trend": trend,
                            "data": indicators_data[ticker].to_dict()
                        }
            
            # Cache the result
            pd.to_pickle(result, cache_file)
            
            return result
        except Exception as e:
            logger.error(f"Error getting economic indicators: {e}")
            return {"error": f"Error getting economic indicators: {str(e)}"}
    
    def get_sector_performance(self, period: str = "1y") -> Dict:
        """
        Get sector performance data
        
        Args:
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            Dictionary with sector performance data
        """
        try:
            # Create a cache key based on period
            cache_key = f"sector_performance_{period}"
            cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
            
            # Check if cache exists and is recent
            if os.path.exists(cache_file):
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if datetime.now() - file_time < self.cache_ttl:
                    logger.info(f"Using cached sector performance data")
                    return pd.read_pickle(cache_file)
            
            # Get data for all sector ETFs
            data = yf.download(list(SECTOR_ETFS.values()), period=period, progress=False)
            
            if data.empty:
                return {"error": "No sector performance data found"}
            
            # Extract close prices
            if len(SECTOR_ETFS) == 1:
                ticker = list(SECTOR_ETFS.values())[0]
                sector_data = pd.DataFrame({ticker: data['Close']})
            else:
                sector_data = pd.DataFrame()
                for ticker in SECTOR_ETFS.values():
                    if (ticker, 'Close') in data.columns:
                        sector_data[ticker] = data[(ticker, 'Close')]
            
            # Calculate performance metrics
            result = {}
            for sector_name, ticker in SECTOR_ETFS.items():
                if ticker in sector_data.columns:
                    # Get current value and change
                    current = sector_data[ticker].iloc[-1]
                    start = sector_data[ticker].iloc[0]
                    change = current - start
                    change_pct = (change / start) * 100 if start != 0 else 0
                    
                    # Calculate volatility
                    returns = sector_data[ticker].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility in percentage
                    
                    # Calculate trend
                    if len(sector_data) > 20:
                        sma20 = sector_data[ticker].iloc[-20:].mean()
                        sma50 = sector_data[ticker].iloc[-50:].mean() if len(sector_data) > 50 else start
                        trend = "up" if sma20 > sma50 else "down" if sma20 < sma50 else "neutral"
                    else:
                        trend = "neutral"
                    
                    # Calculate relative strength vs S&P 500
                    try:
                        spy_data = yf.download("SPY", period=period, progress=False)['Close']
                        spy_change_pct = (spy_data.iloc[-1] / spy_data.iloc[0] - 1) * 100
                        relative_strength = change_pct - spy_change_pct
                    except:
                        relative_strength = 0
                    
                    result[sector_name] = {
                        "ticker": ticker,
                        "current": current,
                        "change": change,
                        "change_pct": change_pct,
                        "volatility": volatility,
                        "trend": trend,
                        "relative_strength": relative_strength,
                        "data": sector_data[ticker].to_dict()
                    }
            
            # Rank sectors by performance
            ranked_sectors = sorted(result.keys(), key=lambda x: result[x]["change_pct"], reverse=True)
            for i, sector in enumerate(ranked_sectors):
                result[sector]["rank"] = i + 1
            
            # Cache the result
            pd.to_pickle(result, cache_file)
            
            return result
        except Exception as e:
            logger.error(f"Error getting sector performance: {e}")
            return {"error": f"Error getting sector performance: {str(e)}"}
    
    def analyze_interest_rate_sensitivity(self, ticker: str, period: str = "1y") -> Dict:
        """
        Analyze a stock's sensitivity to interest rates
        
        Args:
            ticker: Stock symbol
            period: Time period (1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            Dictionary with interest rate sensitivity analysis
        """
        try:
            # Create a cache key
            cache_key = f"{ticker}_interest_rate_sensitivity_{period}"
            cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
            
            # Check if cache exists and is recent
            if os.path.exists(cache_file):
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if datetime.now() - file_time < self.cache_ttl:
                    logger.info(f"Using cached interest rate sensitivity data for {ticker}")
                    return pd.read_pickle(cache_file)
            
            # Get stock data
            stock_data = yf.download(ticker, period=period, progress=False)
            
            if stock_data.empty:
                return {"error": f"No data found for ticker: {ticker}"}
            
            # Get interest rate data
            interest_rate_tickers = [
                ECONOMIC_INDICATORS["interest_rates"]["fed_funds_rate"],
                ECONOMIC_INDICATORS["interest_rates"]["treasury_10y"]
            ]
            
            rate_data = yf.download(interest_rate_tickers, period=period, progress=False)
            
            if rate_data.empty:
                return {"error": "No interest rate data found"}
            
            # Extract close prices
            stock_prices = stock_data['Close']
            
            if len(interest_rate_tickers) == 1:
                rate_ticker = interest_rate_tickers[0]
                rates = pd.DataFrame({rate_ticker: rate_data['Close']})
            else:
                rates = pd.DataFrame()
                for rate_ticker in interest_rate_tickers:
                    if (rate_ticker, 'Close') in rate_data.columns:
                        rates[rate_ticker] = rate_data[(rate_ticker, 'Close')]
            
            # Calculate returns
            stock_returns = stock_prices.pct_change().dropna()
            rate_returns = rates.pct_change().dropna()
            
            # Align dates
            aligned_data = pd.concat([stock_returns, rate_returns], axis=1).dropna()
            
            if aligned_data.empty or len(aligned_data) < 20:
                return {"error": f"Insufficient data for analysis"}
            
            # Calculate correlations
            correlations = {}
            for rate_ticker in interest_rate_tickers:
                if rate_ticker in aligned_data.columns:
                    correlation = aligned_data[stock_returns.name].corr(aligned_data[rate_ticker])
                    correlations[rate_ticker] = correlation
            
            # Perform regression analysis
            regression_results = {}
            for rate_ticker in interest_rate_tickers:
                if rate_ticker in aligned_data.columns:
                    X = sm.add_constant(aligned_data[rate_ticker])
                    model = sm.OLS(aligned_data[stock_returns.name], X).fit()
                    
                    # Extract coefficient (beta) and p-value
                    beta = model.params[1]
                    p_value = model.pvalues[1]
                    r_squared = model.rsquared
                    
                    regression_results[rate_ticker] = {
                        "beta": beta,
                        "p_value": p_value,
                        "r_squared": r_squared,
                        "significant": p_value < 0.05
                    }
            
            # Determine overall sensitivity
            sensitivity_scores = []
            for rate_ticker in interest_rate_tickers:
                if rate_ticker in regression_results:
                    # Calculate sensitivity score based on beta and significance
                    beta = regression_results[rate_ticker]["beta"]
                    p_value = regression_results[rate_ticker]["p_value"]
                    
                    # Higher weight for significant relationships
                    weight = 1.0 if p_value < 0.05 else 0.3
                    
                    # Negative beta means stock goes down when rates go up
                    sensitivity_scores.append(beta * weight)
            
            # Average sensitivity score
            avg_sensitivity = np.mean(sensitivity_scores) if sensitivity_scores else 0
            
            # Determine sensitivity level
            if abs(avg_sensitivity) < 0.5:
                sensitivity_level = "low"
            elif abs(avg_sensitivity) < 1.5:
                sensitivity_level = "moderate"
            else:
                sensitivity_level = "high"
            
            # Determine direction
            direction = "negative" if avg_sensitivity < 0 else "positive" if avg_sensitivity > 0 else "neutral"
            
            # Prepare result
            result = {
                "ticker": ticker,
                "period": period,
                "correlations": correlations,
                "regression_results": regression_results,
                "sensitivity": {
                    "score": avg_sensitivity,
                    "level": sensitivity_level,
                    "direction": direction
                },
                "interpretation": self._interpret_interest_rate_sensitivity(avg_sensitivity, ticker)
            }
            
            # Cache the result
            pd.to_pickle(result, cache_file)
            
            return result
        except Exception as e:
            logger.error(f"Error analyzing interest rate sensitivity: {e}")
            return {"error": f"Error analyzing interest rate sensitivity: {str(e)}"}
    
    def _interpret_interest_rate_sensitivity(self, sensitivity_score: float, ticker: str) -> str:
        """
        Interpret interest rate sensitivity score
        
        Args:
            sensitivity_score: Sensitivity score
            ticker: Stock symbol
            
        Returns:
            Interpretation string
        """
        # Get company info to determine sector
        try:
            company_info = yf.Ticker(ticker).info
            sector = company_info.get("sector", "Unknown")
        except:
            sector = "Unknown"
        
        if sensitivity_score < -1.5:
            interpretation = f"{ticker} shows high negative sensitivity to interest rate changes. "
            interpretation += "This means the stock tends to decline significantly when interest rates rise. "
            
            if sector in ["Financials", "Financial Services"]:
                interpretation += "This is unusual for financial stocks, which typically benefit from higher rates. "
                interpretation += "This could indicate other factors affecting the company's profitability."
            elif sector in ["Utilities", "Real Estate"]:
                interpretation += "This is typical for utility and real estate companies, which are often viewed as 'bond proxies' "
                interpretation += "and are negatively affected by higher rates due to their high dividend yields and capital-intensive nature."
            elif sector in ["Technology", "Communication Services"]:
                interpretation += "Technology companies with high growth expectations and low current earnings are often "
                interpretation += "negatively affected by higher rates as their future cash flows are discounted more heavily."
        
        elif sensitivity_score < -0.5:
            interpretation = f"{ticker} shows moderate negative sensitivity to interest rate changes. "
            interpretation += "The stock tends to decline when interest rates rise, though not dramatically. "
            
            if sector in ["Consumer Discretionary"]:
                interpretation += "Consumer discretionary companies often see reduced consumer spending in high-rate environments "
                interpretation += "as borrowing costs increase for consumers."
        
        elif sensitivity_score < 0.5:
            interpretation = f"{ticker} shows low sensitivity to interest rate changes. "
            interpretation += "The stock price movements appear to be largely independent of interest rate fluctuations. "
            
            if sector in ["Consumer Staples", "Healthcare"]:
                interpretation += "This is typical for defensive sectors like consumer staples and healthcare, "
                interpretation += "which tend to be less affected by economic cycles and interest rate changes."
        
        elif sensitivity_score < 1.5:
            interpretation = f"{ticker} shows moderate positive sensitivity to interest rate changes. "
            interpretation += "The stock tends to rise when interest rates increase. "
            
            if sector in ["Financials", "Financial Services"]:
                interpretation += "This is typical for financial companies, which often benefit from higher interest rates "
                interpretation += "through increased net interest margins and higher returns on cash holdings."
        
        else:
            interpretation = f"{ticker} shows high positive sensitivity to interest rate changes. "
            interpretation += "The stock tends to rise significantly when interest rates increase. "
            
            if sector in ["Financials", "Financial Services"]:
                interpretation += "This strong positive relationship is common among financial institutions, particularly banks, "
                interpretation += "which can see substantial profit increases in rising rate environments."
        
        # Add forward-looking statement
        if sensitivity_score < -0.5:
            interpretation += "\n\nIn the current economic environment, "
            
            # Check recent Fed policy direction
            try:
                fed_funds = yf.download("^FVX", period="6mo")['Close']
                recent_trend = "rising" if fed_funds.iloc[-1] > fed_funds.iloc[-30] else "falling"
                
                if recent_trend == "rising":
                    interpretation += "with interest rates trending upward, this stock may face headwinds. "
                    interpretation += "Consider monitoring the Federal Reserve's policy statements closely."
                else:
                    interpretation += "with interest rates stabilizing or declining, this stock may benefit. "
                    interpretation += "Any signals of rate cuts could be positive catalysts."
            except:
                interpretation += "it's important to monitor Federal Reserve policy closely, as changes in "
                interpretation += "interest rate expectations could significantly impact this stock."
        
        elif sensitivity_score > 0.5:
            interpretation += "\n\nIn the current economic environment, "
            
            # Check recent Fed policy direction
            try:
                fed_funds = yf.download("^FVX", period="6mo")['Close']
                recent_trend = "rising" if fed_funds.iloc[-1] > fed_funds.iloc[-30] else "falling"
                
                if recent_trend == "rising":
                    interpretation += "with interest rates trending upward, this stock may benefit. "
                    interpretation += "Further rate hikes could be positive catalysts."
                else:
                    interpretation += "with interest rates stabilizing or declining, this stock may face headwinds. "
                    interpretation += "Any signals of rate cuts could negatively impact performance."
            except:
                interpretation += "it's important to monitor Federal Reserve policy closely, as changes in "
                interpretation += "interest rate expectations could significantly impact this stock."
        
        return interpretation
    
    def analyze_inflation_sensitivity(self, ticker: str, period: str = "1y") -> Dict:
        """
        Analyze a stock's sensitivity to inflation
        
        Args:
            ticker: Stock symbol
            period: Time period (1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            Dictionary with inflation sensitivity analysis
        """
        try:
            # Create a cache key
            cache_key = f"{ticker}_inflation_sensitivity_{period}"
            cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
            
            # Check if cache exists and is recent
            if os.path.exists(cache_file):
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if datetime.now() - file_time < self.cache_ttl:
                    logger.info(f"Using cached inflation sensitivity data for {ticker}")
                    return pd.read_pickle(cache_file)
            
            # Get stock data
            stock_data = yf.download(ticker, period=period, progress=False)
            
            if stock_data.empty:
                return {"error": f"No data found for ticker: {ticker}"}
            
            # Get inflation indicator data
            inflation_tickers = [
                ECONOMIC_INDICATORS["inflation"]["tips"],
                ECONOMIC_INDICATORS["inflation"]["inflation_expectation"]
            ]
            
            inflation_data = yf.download(inflation_tickers, period=period, progress=False)
            
            if inflation_data.empty:
                return {"error": "No inflation indicator data found"}
            
            # Extract close prices
            stock_prices = stock_data['Close']
            
            if len(inflation_tickers) == 1:
                inflation_ticker = inflation_tickers[0]
                inflation_prices = pd.DataFrame({inflation_ticker: inflation_data['Close']})
            else:
                inflation_prices = pd.DataFrame()
                for inflation_ticker in inflation_tickers:
                    if (inflation_ticker, 'Close') in inflation_data.columns:
                        inflation_prices[inflation_ticker] = inflation_data[(inflation_ticker, 'Close')]
            
            # Calculate returns
            stock_returns = stock_prices.pct_change().dropna()
            inflation_returns = inflation_prices.pct_change().dropna()
            
            # Align dates
            aligned_data = pd.concat([stock_returns, inflation_returns], axis=1).dropna()
            
            if aligned_data.empty or len(aligned_data) < 20:
                return {"error": f"Insufficient data for analysis"}
            
            # Calculate correlations
            correlations = {}
            for inflation_ticker in inflation_tickers:
                if inflation_ticker in aligned_data.columns:
                    correlation = aligned_data[stock_returns.name].corr(aligned_data[inflation_ticker])
                    correlations[inflation_ticker] = correlation
            
            # Perform regression analysis
            regression_results = {}
            for inflation_ticker in inflation_tickers:
                if inflation_ticker in aligned_data.columns:
                    X = sm.add_constant(aligned_data[inflation_ticker])
                    model = sm.OLS(aligned_data[stock_returns.name], X).fit()
                    
                    # Extract coefficient (beta) and p-value
                    beta = model.params[1]
                    p_value = model.pvalues[1]
                    r_squared = model.rsquared
                    
                    regression_results[inflation_ticker] = {
                        "beta": beta,
                        "p_value": p_value,
                        "r_squared": r_squared,
                        "significant": p_value < 0.05
                    }
            
            # Determine overall sensitivity
            sensitivity_scores = []
            for inflation_ticker in inflation_tickers:
                if inflation_ticker in regression_results:
                    # Calculate sensitivity score based on beta and significance
                    beta = regression_results[inflation_ticker]["beta"]
                    p_value = regression_results[inflation_ticker]["p_value"]
                    
                    # Higher weight for significant relationships
                    weight = 1.0 if p_value < 0.05 else 0.3
                    
                    sensitivity_scores.append(beta * weight)
            
            # Average sensitivity score
            avg_sensitivity = np.mean(sensitivity_scores) if sensitivity_scores else 0
            
            # Determine sensitivity level
            if abs(avg_sensitivity) < 0.5:
                sensitivity_level = "low"
            elif abs(avg_sensitivity) < 1.5:
                sensitivity_level = "moderate"
            else:
                sensitivity_level = "high"
            
            # Determine direction
            direction = "negative" if avg_sensitivity < 0 else "positive" if avg_sensitivity > 0 else "neutral"
            
            # Prepare result
            result = {
                "ticker": ticker,
                "period": period,
                "correlations": correlations,
                "regression_results": regression_results,
                "sensitivity": {
                    "score": avg_sensitivity,
                    "level": sensitivity_level,
                    "direction": direction
                },
                "interpretation": self._interpret_inflation_sensitivity(avg_sensitivity, ticker)
            }
            
            # Cache the result
            pd.to_pickle(result, cache_file)
            
            return result
        except Exception as e:
            logger.error(f"Error analyzing inflation sensitivity: {e}")
            return {"error": f"Error analyzing inflation sensitivity: {str(e)}"}
    
    def _interpret_inflation_sensitivity(self, sensitivity_score: float, ticker: str) -> str:
        """
        Interpret inflation sensitivity score
        
        Args:
            sensitivity_score: Sensitivity score
            ticker: Stock symbol
            
        Returns:
            Interpretation string
        """
        # Get company info to determine sector
        try:
            company_info = yf.Ticker(ticker).info
            sector = company_info.get("sector", "Unknown")
        except:
            sector = "Unknown"
        
        if sensitivity_score < -1.5:
            interpretation = f"{ticker} shows high negative sensitivity to inflation indicators. "
            interpretation += "This means the stock tends to decline significantly when inflation rises. "
            
            if sector in ["Technology", "Communication Services", "Consumer Discretionary"]:
                interpretation += "This is common for growth-oriented companies with earnings expected far in the future. "
                interpretation += "Higher inflation erodes the present value of these future earnings."
            elif sector in ["Utilities"]:
                interpretation += "Utility companies often struggle in high inflation environments due to regulated pricing "
                interpretation += "that may not keep pace with rising costs."
        
        elif sensitivity_score < -0.5:
            interpretation = f"{ticker} shows moderate negative sensitivity to inflation indicators. "
            interpretation += "The stock tends to decline when inflation rises, though not dramatically. "
        
        elif sensitivity_score < 0.5:
            interpretation = f"{ticker} shows low sensitivity to inflation indicators. "
            interpretation += "The stock price movements appear to be largely independent of inflation fluctuations. "
        
        elif sensitivity_score < 1.5:
            interpretation = f"{ticker} shows moderate positive sensitivity to inflation indicators. "
            interpretation += "The stock tends to rise when inflation increases. "
            
            if sector in ["Energy", "Materials"]:
                interpretation += "This is typical for companies in the energy and materials sectors, which often "
                interpretation += "benefit from rising commodity prices that accompany inflation."
            elif sector in ["Financials"]:
                interpretation += "Financial companies can sometimes benefit from inflation through higher interest rates, "
                interpretation += "though this depends on the steepness of the yield curve."
        
        else:
            interpretation = f"{ticker} shows high positive sensitivity to inflation indicators. "
            interpretation += "The stock tends to rise significantly when inflation increases. "
            
            if sector in ["Energy", "Materials"]:
                interpretation += "Companies that produce commodities or own hard assets often serve as inflation hedges, "
                interpretation += "as they can pass through higher prices to customers."
            elif sector in ["Real Estate"]:
                interpretation += "Real estate assets often appreciate during inflationary periods, and REITs can "
                interpretation += "increase rents to keep pace with inflation."
        
        # Add forward-looking statement
        if sensitivity_score < -0.5:
            interpretation += "\n\nIn the current economic environment, "
            
            # Check recent inflation trend
            try:
                inflation_etf = yf.download("RINF", period="6mo")['Close']
                recent_trend = "rising" if inflation_etf.iloc[-1] > inflation_etf.iloc[-30] else "falling"
                
                if recent_trend == "rising":
                    interpretation += "with inflation trending upward, this stock may face headwinds. "
                    interpretation += "Consider monitoring inflation reports and Fed policy closely."
                else:
                    interpretation += "with inflation stabilizing or declining, this stock may benefit. "
                    interpretation += "Any signals of inflation peaking could be positive catalysts."
            except:
                interpretation += "it's important to monitor inflation indicators closely, as changes in "
                interpretation += "inflation expectations could significantly impact this stock."
        
        elif sensitivity_score > 0.5:
            interpretation += "\n\nIn the current economic environment, "
            
            # Check recent inflation trend
            try:
                inflation_etf = yf.download("RINF", period="6mo")['Close']
                recent_trend = "rising" if inflation_etf.iloc[-1] > inflation_etf.iloc[-30] else "falling"
                
                if recent_trend == "rising":
                    interpretation += "with inflation trending upward, this stock may benefit. "
                    interpretation += "Further inflation could be a positive catalyst."
                else:
                    interpretation += "with inflation stabilizing or declining, this stock may face headwinds. "
                    interpretation += "Disinflation could negatively impact performance."
            except:
                interpretation += "it's important to monitor inflation indicators closely, as changes in "
                interpretation += "inflation expectations could significantly impact this stock."
        
        return interpretation
    
    def get_macroeconomic_context(self, ticker: str, period: str = "1y") -> Dict:
        """
        Get comprehensive macroeconomic context for a stock
        
        Args:
            ticker: Stock symbol
            period: Time period (1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            Dictionary with macroeconomic context
        """
        try:
            # Create a cache key
            cache_key = f"{ticker}_macro_context_{period}"
            cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
            
            # Check if cache exists and is recent
            if os.path.exists(cache_file):
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if datetime.now() - file_time < self.cache_ttl:
                    logger.info(f"Using cached macroeconomic context for {ticker}")
                    return pd.read_pickle(cache_file)
            
            # Get company info to determine sector
            try:
                company_info = yf.Ticker(ticker).info
                sector = company_info.get("sector", "Unknown")
                industry = company_info.get("industry", "Unknown")
            except:
                sector = "Unknown"
                industry = "Unknown"
            
            # Get economic indicators
            indicators = self.get_economic_indicators(period)
            
            # Get sector performance
            sector_performance = self.get_sector_performance(period)
            
            # Get interest rate sensitivity
            interest_rate_sensitivity = self.analyze_interest_rate_sensitivity(ticker, period)
            
            # Get inflation sensitivity
            inflation_sensitivity = self.analyze_inflation_sensitivity(ticker, period)
            
            # Identify relevant economic indicators based on sector
            relevant_indicators = {}
            if sector != "Unknown":
                # Map sector to standard sector name
                sector_lower = sector.lower()
                for standard_sector in SECTOR_SENSITIVITIES:
                    if standard_sector in sector_lower:
                        sector = standard_sector
                        break
                
                # Get relevant indicator categories for this sector
                indicator_categories = SECTOR_SENSITIVITIES.get(sector, [])
                
                # Extract relevant indicators
                for category in indicator_categories:
                    if category in indicators and isinstance(indicators, dict) and "error" not in indicators:
                        relevant_indicators[category] = indicators[category]
            
            # Get sector ETF if available
            sector_etf = None
            sector_data = None
            for sector_name, etf in SECTOR_ETFS.items():
                if sector_name in sector.lower():
                    sector_etf = etf
                    if isinstance(sector_performance, dict) and "error" not in sector_performance and sector_name in sector_performance:
                        sector_data = sector_performance[sector_name]
                    break
            
            # Prepare result
            result = {
                "ticker": ticker,
                "sector": sector,
                "industry": industry,
                "period": period,
                "economic_indicators": {
                    "all": indicators,
                    "relevant": relevant_indicators
                },
                "sector_performance": sector_data,
                "interest_rate_sensitivity": interest_rate_sensitivity,
                "inflation_sensitivity": inflation_sensitivity,
                "summary": self._generate_macro_context_summary(
                    ticker, sector, indicators, sector_data, 
                    interest_rate_sensitivity, inflation_sensitivity
                )
            }
            
            # Cache the result
            pd.to_pickle(result, cache_file)
            
            return result
        except Exception as e:
            logger.error(f"Error getting macroeconomic context: {e}")
            return {"error": f"Error getting macroeconomic context: {str(e)}"}
    
    def _generate_macro_context_summary(self, ticker: str, sector: str, 
                                      indicators: Dict, sector_data: Dict,
                                      interest_rate_sensitivity: Dict, 
                                      inflation_sensitivity: Dict) -> str:
        """
        Generate a summary of macroeconomic context
        
        Args:
            ticker: Stock symbol
            sector: Stock sector
            indicators: Economic indicators data
            sector_data: Sector performance data
            interest_rate_sensitivity: Interest rate sensitivity data
            inflation_sensitivity: Inflation sensitivity data
            
        Returns:
            Summary string
        """
        summary = f"Macroeconomic Context for {ticker} ({sector})\n\n"
        
        # Add interest rate context
        if "error" not in interest_rate_sensitivity:
            sensitivity = interest_rate_sensitivity["sensitivity"]
            summary += f"Interest Rate Sensitivity: {sensitivity['level'].title()} {sensitivity['direction']}\n"
            
            # Add current interest rate environment
            if "interest_rates" in indicators:
                rates = indicators["interest_rates"]
                if "treasury_10y" in rates:
                    current_10y = rates["treasury_10y"]["current"]
                    trend_10y = rates["treasury_10y"]["trend"]
                    summary += f"Current 10-Year Treasury: {current_10y:.2f}% (Trend: {trend_10y})\n"
        
        # Add inflation context
        if "error" not in inflation_sensitivity:
            sensitivity = inflation_sensitivity["sensitivity"]
            summary += f"Inflation Sensitivity: {sensitivity['level'].title()} {sensitivity['direction']}\n"
            
            # Add current inflation environment
            if "inflation" in indicators:
                inflation = indicators["inflation"]
                if "inflation_expectation" in inflation:
                    trend = inflation["inflation_expectation"]["trend"]
                    summary += f"Current Inflation Trend: {trend}\n"
        
        # Add sector performance context
        if sector_data and "error" not in sector_data:
            sector_change = sector_data["change_pct"]
            sector_trend = sector_data["trend"]
            sector_rank = sector_data.get("rank", "N/A")
            
            summary += f"\nSector Performance: {sector_change:.2f}% (Trend: {sector_trend}, Rank: {sector_rank}/11)\n"
            
            # Add relative strength
            if "relative_strength" in sector_data:
                rel_strength = sector_data["relative_strength"]
                rel_str_text = "outperforming" if rel_strength > 0 else "underperforming"
                summary += f"Sector is {rel_str_text} the S&P 500 by {abs(rel_strength):.2f}%\n"
        
        # Add economic growth context
        if "economic_growth" in indicators:
            growth = indicators["economic_growth"]
            if "gdp_proxy" in growth:
                gdp_trend = growth["gdp_proxy"]["trend"]
                summary += f"\nEconomic Growth Trend: {gdp_trend}\n"
        
        # Add volatility context
        if "volatility" in indicators:
            vol = indicators["volatility"]
            if "vix" in vol:
                vix_value = vol["vix"]["current"]
                vix_change = vol["vix"]["change_pct"]
                summary += f"Market Volatility (VIX): {vix_value:.2f} ({vix_change:+.2f}%)\n"
        
        # Add commodities context if relevant
        if sector in ["energy", "materials", "industrials"]:
            if "commodities" in indicators:
                commodities = indicators["commodities"]
                summary += "\nRelevant Commodities:\n"
                
                if "oil" in commodities and sector in ["energy"]:
                    oil_price = commodities["oil"]["current"]
                    oil_change = commodities["oil"]["change_pct"]
                    oil_trend = commodities["oil"]["trend"]
                    summary += f"Crude Oil: ${oil_price:.2f} ({oil_change:+.2f}%, Trend: {oil_trend})\n"
                
                if "gold" in commodities and sector in ["materials"]:
                    gold_price = commodities["gold"]["current"]
                    gold_change = commodities["gold"]["change_pct"]
                    gold_trend = commodities["gold"]["trend"]
                    summary += f"Gold: ${gold_price:.2f} ({gold_change:+.2f}%, Trend: {gold_trend})\n"
        
        # Add forward-looking statement
        summary += "\nForward-Looking Implications:\n"
        
        # Interest rate implications
        if "error" not in interest_rate_sensitivity:
            ir_sensitivity = interest_rate_sensitivity["sensitivity"]
            
            # Check recent Fed policy direction
            try:
                fed_funds = yf.download("^FVX", period="6mo")['Close']
                recent_trend = "rising" if fed_funds.iloc[-1] > fed_funds.iloc[-30] else "falling"
                
                if ir_sensitivity["direction"] == "negative" and ir_sensitivity["level"] != "low":
                    if recent_trend == "rising":
                        summary += "- Interest rates trending upward may create headwinds\n"
                    else:
                        summary += "- Interest rates stabilizing/declining may be beneficial\n"
                elif ir_sensitivity["direction"] == "positive" and ir_sensitivity["level"] != "low":
                    if recent_trend == "rising":
                        summary += "- Interest rates trending upward may be beneficial\n"
                    else:
                        summary += "- Interest rates stabilizing/declining may create headwinds\n"
            except:
                pass
        
        # Inflation implications
        if "error" not in inflation_sensitivity:
            inf_sensitivity = inflation_sensitivity["sensitivity"]
            
            # Check recent inflation trend
            try:
                inflation_etf = yf.download("RINF", period="6mo")['Close']
                recent_trend = "rising" if inflation_etf.iloc[-1] > inflation_etf.iloc[-30] else "falling"
                
                if inf_sensitivity["direction"] == "negative" and inf_sensitivity["level"] != "low":
                    if recent_trend == "rising":
                        summary += "- Inflation trending upward may create headwinds\n"
                    else:
                        summary += "- Inflation stabilizing/declining may be beneficial\n"
                elif inf_sensitivity["direction"] == "positive" and inf_sensitivity["level"] != "low":
                    if recent_trend == "rising":
                        summary += "- Inflation trending upward may be beneficial\n"
                    else:
                        summary += "- Inflation stabilizing/declining may create headwinds\n"
            except:
                pass
        
        # Sector trend implications
        if sector_data and "error" not in sector_data:
            sector_trend = sector_data["trend"]
            if sector_trend == "up":
                summary += "- Sector momentum is positive, which may provide tailwinds\n"
            elif sector_trend == "down":
                summary += "- Sector momentum is negative, which may create headwinds\n"
        
        return summary

# Helper functions for economic analysis
def get_economic_indicators(period: str = "1y") -> Dict:
    """
    Get economic indicators data
    
    Args:
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
    Returns:
        Dictionary with economic indicators data
    """
    try:
        analyzer = EconomicFactorsAnalyzer()
        result = analyzer.get_economic_indicators(period)
        return result
    except Exception as e:
        logger.error(f"Error getting economic indicators: {e}")
        return {"error": f"Error getting economic indicators: {str(e)}"}

def get_sector_performance(period: str = "1y") -> Dict:
    """
    Get sector performance data
    
    Args:
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
    Returns:
        Dictionary with sector performance data
    """
    try:
        analyzer = EconomicFactorsAnalyzer()
        result = analyzer.get_sector_performance(period)
        return result
    except Exception as e:
        logger.error(f"Error getting sector performance: {e}")
        return {"error": f"Error getting sector performance: {str(e)}"}

def analyze_interest_rate_sensitivity(ticker: str, period: str = "1y") -> Dict:
    """
    Analyze a stock's sensitivity to interest rates
    
    Args:
        ticker: Stock symbol
        period: Time period (1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
    Returns:
        Dictionary with interest rate sensitivity analysis
    """
    try:
        analyzer = EconomicFactorsAnalyzer()
        result = analyzer.analyze_interest_rate_sensitivity(ticker, period)
        return result
    except Exception as e:
        logger.error(f"Error analyzing interest rate sensitivity: {e}")
        return {"error": f"Error analyzing interest rate sensitivity: {str(e)}"}

def analyze_inflation_sensitivity(ticker: str, period: str = "1y") -> Dict:
    """
    Analyze a stock's sensitivity to inflation
    
    Args:
        ticker: Stock symbol
        period: Time period (1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
    Returns:
        Dictionary with inflation sensitivity analysis
    """
    try:
        analyzer = EconomicFactorsAnalyzer()
        result = analyzer.analyze_inflation_sensitivity(ticker, period)
        return result
    except Exception as e:
        logger.error(f"Error analyzing inflation sensitivity: {e}")
        return {"error": f"Error analyzing inflation sensitivity: {str(e)}"}

def get_macroeconomic_context(ticker: str, period: str = "1y") -> Dict:
    """
    Get comprehensive macroeconomic context for a stock
    
    Args:
        ticker: Stock symbol
        period: Time period (1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
    Returns:
        Dictionary with macroeconomic context
    """
    try:
        analyzer = EconomicFactorsAnalyzer()
        result = analyzer.get_macroeconomic_context(ticker, period)
        return result
    except Exception as e:
        logger.error(f"Error getting macroeconomic context: {e}")
        return {"error": f"Error getting macroeconomic context: {str(e)}"}
