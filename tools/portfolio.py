# tools/portfolio.py

import os
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Union, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import json
import scipy.optimize as sco
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache directory for portfolio data
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../cache")
os.makedirs(CACHE_DIR, exist_ok=True)

class PortfolioManager:
    """Portfolio management and optimization class"""
    
    def __init__(self, risk_free_rate: float = 0.04):
        """
        Initialize the portfolio manager
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 4%)
        """
        self.risk_free_rate = risk_free_rate
        self.daily_risk_free_rate = (1 + risk_free_rate) ** (1/252) - 1
    
    def get_historical_data(self, tickers: List[str], period: str = "1y") -> pd.DataFrame:
        """
        Get historical price data for a list of tickers
        
        Args:
            tickers: List of stock symbols
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            DataFrame with historical prices
        """
        try:
            # Create a cache key based on tickers and period
            cache_key = f"portfolio_{'_'.join(tickers)}_{period}"
            cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
            
            # Check if cache exists and is recent (less than 24 hours old)
            if os.path.exists(cache_file):
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if datetime.now() - file_time < timedelta(hours=24):
                    logger.info(f"Using cached data for portfolio {tickers}")
                    return pd.read_pickle(cache_file)
            
            # Get data for all tickers
            data = yf.download(tickers, period=period, group_by='ticker')
            
            # Check if data is empty
            if data.empty:
                logger.error(f"No data found for tickers: {tickers}")
                return pd.DataFrame()
            
            # If only one ticker, restructure the data
            if len(tickers) == 1:
                ticker = tickers[0]
                if 'Close' in data.columns:
                    data = pd.DataFrame({ticker: data['Close']})
                else:
                    logger.error(f"No 'Close' column found for ticker: {ticker}")
                    return pd.DataFrame()
            else:
                # Extract close prices
                close_prices = pd.DataFrame()
                for ticker in tickers:
                    if (ticker, 'Close') in data.columns:
                        close_prices[ticker] = data[(ticker, 'Close')]
                    else:
                        logger.warning(f"No data found for ticker: {ticker}")
                
                # Check if we have data for at least one ticker
                if close_prices.empty:
                    logger.error(f"No close price data found for any tickers: {tickers}")
                    return pd.DataFrame()
                
                data = close_prices
            
            # Fill missing values with forward fill then backward fill
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Check for remaining NaN values
            if data.isna().any().any():
                logger.warning(f"Some NaN values remain after filling: {data.isna().sum().sum()} NaNs")
                # Drop rows with any remaining NaN values
                data = data.dropna()
                
                if data.empty:
                    logger.error("All data was NaN after cleaning")
                    return pd.DataFrame()
            
            # Cache the data
            data.to_pickle(cache_file)
            
            return data
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()
    
    def calculate_returns(self, prices: pd.DataFrame, log_returns: bool = False) -> pd.DataFrame:
        """
        Calculate returns from price data
        
        Args:
            prices: DataFrame with price data
            log_returns: Whether to calculate log returns (default: False)
            
        Returns:
            DataFrame with returns
        """
        if log_returns:
            return np.log(prices / prices.shift(1)).dropna()
        else:
            return (prices / prices.shift(1) - 1).dropna()
    
    def calculate_portfolio_performance(self, returns: pd.DataFrame, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio performance metrics
        
        Args:
            returns: DataFrame with returns
            weights: Array of portfolio weights
            
        Returns:
            Tuple of (expected return, volatility, Sharpe ratio)
        """
        # Convert daily returns to annual returns
        expected_returns = np.sum(returns.mean() * weights) * 252
        
        # Calculate portfolio volatility
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        
        # Calculate Sharpe ratio
        sharpe_ratio = (expected_returns - self.risk_free_rate) / portfolio_volatility
        
        return expected_returns, portfolio_volatility, sharpe_ratio
    
    def negative_sharpe_ratio(self, weights: np.ndarray, returns: pd.DataFrame, 
                             min_assets: int = 2, diversification_penalty: float = 1.0) -> float:
        """
        Calculate negative Sharpe ratio for optimization with diversification penalty
        
        Args:
            weights: Array of portfolio weights
            returns: DataFrame with returns
            min_assets: Minimum number of assets with non-zero weights
            diversification_penalty: Penalty factor for lack of diversification
            
        Returns:
            Negative Sharpe ratio with diversification penalty
        """
        expected_returns, portfolio_volatility, sharpe_ratio = self.calculate_portfolio_performance(returns, weights)
        
        # Count assets with significant weights (>1%)
        significant_assets = np.sum(weights > 0.01)
        
        # Apply penalty if fewer than min_assets have significant weights
        if significant_assets < min_assets:
            diversification_penalty_value = (min_assets - significant_assets) * diversification_penalty
            return -sharpe_ratio + diversification_penalty_value
        
        return -sharpe_ratio
    
    def optimize_portfolio(self, tickers: List[str], period: str = "1y", 
                          constraint_set: str = "long_only", 
                          max_weight: float = 0.4,
                          min_assets: int = 2) -> Dict:
        """
        Optimize portfolio weights to maximize Sharpe ratio
        
        Args:
            tickers: List of stock symbols
            period: Time period for historical data
            constraint_set: Type of constraints ('long_only', 'long_short', 'market_neutral')
            max_weight: Maximum weight for any single asset (default: 0.4 or 40%)
            min_assets: Minimum number of assets with non-zero weights (default: 2)
            
        Returns:
            Dictionary with optimization results
        """
        try:
            # Get historical data
            prices = self.get_historical_data(tickers, period)
            if prices.empty:
                return {"error": f"No data found for tickers: {tickers}"}
            
            # Calculate returns
            returns = self.calculate_returns(prices)
            
            # Set up optimization constraints
            num_assets = len(tickers)
            
            # Ensure we have enough assets for diversification
            if num_assets < min_assets:
                return {"error": f"Need at least {min_assets} assets for proper diversification"}
            
            # Adjust max_weight to ensure we can satisfy min_assets constraint
            max_weight = min(max_weight, 1.0 - (min_assets - 1) * 0.01)
            
            if constraint_set == "long_only":
                # Long only: weights between 0 and max_weight, sum to 1
                bounds = tuple((0, max_weight) for _ in range(num_assets))
                constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
                
                # Add constraint to ensure minimum number of assets
                # This is a soft constraint using a penalty function in the objective
                initial_weights = np.array([1/num_assets] * num_assets)
            
            elif constraint_set == "long_short":
                # Long-short: weights between -0.3 and max_weight, sum to 1
                bounds = tuple((-0.3, max_weight) for _ in range(num_assets))
                constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
                initial_weights = np.array([1/num_assets] * num_assets)
            
            elif constraint_set == "market_neutral":
                # Market neutral: weights sum to 0, between -max_weight and max_weight
                bounds = tuple((-max_weight, max_weight) for _ in range(num_assets))
                constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x)}]
                initial_weights = np.array([0] * num_assets)
            
            else:
                # Default to long only with diversification
                bounds = tuple((0, max_weight) for _ in range(num_assets))
                constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
                initial_weights = np.array([1/num_assets] * num_assets)
            
            # Run optimization
            optimization_result = sco.minimize(
                self.negative_sharpe_ratio,
                initial_weights,
                args=(returns,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            # Get optimal weights
            optimal_weights = optimization_result['x']
            
            # Calculate performance metrics
            expected_return, volatility, sharpe_ratio = self.calculate_portfolio_performance(returns, optimal_weights)
            
            # Calculate individual asset metrics
            asset_expected_returns = returns.mean() * 252
            asset_volatilities = np.sqrt(np.diag(returns.cov() * 252))
            asset_sharpe_ratios = (asset_expected_returns - self.risk_free_rate) / asset_volatilities
            
            # Calculate correlation matrix
            correlation_matrix = returns.corr().values.tolist()
            
            # Calculate portfolio beta (using S&P 500 as market proxy)
            try:
                sp500 = self.get_historical_data(["^GSPC"], period)
                
                if not sp500.empty and not returns.empty:
                    sp500_returns = self.calculate_returns(sp500)
                    
                    # Align dates
                    common_dates = returns.index.intersection(sp500_returns.index)
                    
                    if len(common_dates) > 0:
                        aligned_returns = returns.loc[common_dates]
                        aligned_sp500 = sp500_returns.loc[common_dates]
                        
                        # Calculate portfolio returns
                        portfolio_returns = aligned_returns.dot(optimal_weights)
                        
                        # Check if we have enough data
                        if len(portfolio_returns) > 0 and not aligned_sp500.empty:
                            # Get the S&P 500 returns as a Series
                            sp500_return_series = aligned_sp500.iloc[:, 0] if aligned_sp500.shape[1] > 0 else None
                            
                            if sp500_return_series is not None:
                                # Calculate beta
                                covariance = np.cov(portfolio_returns, sp500_return_series)[0, 1]
                                market_variance = np.var(sp500_return_series)
                                portfolio_beta = covariance / market_variance if market_variance > 0 else None
                            else:
                                portfolio_beta = None
                        else:
                            portfolio_beta = None
                    else:
                        portfolio_beta = None
                else:
                    portfolio_beta = None
            except Exception as e:
                logger.error(f"Error calculating beta: {e}")
                portfolio_beta = None
            
            # Calculate Value at Risk (VaR)
            portfolio_returns = returns.dot(optimal_weights)
            var_95 = np.percentile(portfolio_returns, 5) * 100  # 95% VaR (daily)
            var_99 = np.percentile(portfolio_returns, 1) * 100  # 99% VaR (daily)
            
            # Calculate Conditional VaR (CVaR) / Expected Shortfall
            cvar_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * 100
            
            # Calculate maximum drawdown
            portfolio_cumulative = (1 + portfolio_returns).cumprod()
            running_max = portfolio_cumulative.cummax()
            drawdown = (portfolio_cumulative / running_max - 1) * 100
            max_drawdown = drawdown.min()
            
            # Prepare results
            result = {
                "tickers": tickers,
                "weights": {ticker: weight for ticker, weight in zip(tickers, optimal_weights)},
                "expected_annual_return": expected_return * 100,  # Convert to percentage
                "annual_volatility": volatility * 100,  # Convert to percentage
                "sharpe_ratio": sharpe_ratio,
                "portfolio_beta": portfolio_beta,
                "risk_metrics": {
                    "value_at_risk_95": -var_95,  # Daily VaR at 95% confidence (positive number)
                    "value_at_risk_99": -var_99,  # Daily VaR at 99% confidence (positive number)
                    "conditional_var_95": -cvar_95,  # Daily CVaR at 95% confidence (positive number)
                    "max_drawdown": -max_drawdown  # Maximum historical drawdown (positive number)
                },
                "asset_metrics": {
                    ticker: {
                        "weight": weight,
                        "expected_return": ret * 100,  # Convert to percentage
                        "volatility": vol * 100,  # Convert to percentage
                        "sharpe_ratio": sharpe
                    } for ticker, weight, ret, vol, sharpe in zip(
                        tickers, optimal_weights, asset_expected_returns, asset_volatilities, asset_sharpe_ratios
                    )
                },
                "correlation_matrix": correlation_matrix,
                "optimization_success": optimization_result['success'],
                "constraint_set": constraint_set
            }
            
            return result
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            return {"error": f"Error optimizing portfolio: {str(e)}"}
    
    def calculate_efficient_frontier(self, tickers: List[str], period: str = "1y", 
                                    points: int = 20) -> Dict:
        """
        Calculate the efficient frontier for a set of assets
        
        Args:
            tickers: List of stock symbols
            period: Time period for historical data
            points: Number of points on the efficient frontier
            
        Returns:
            Dictionary with efficient frontier data
        """
        try:
            # Get historical data
            prices = self.get_historical_data(tickers, period)
            if prices.empty:
                return {"error": f"No data found for tickers: {tickers}"}
            
            # Calculate returns
            returns = self.calculate_returns(prices)
            
            # Set up optimization
            num_assets = len(tickers)
            bounds = tuple((0, 1) for _ in range(num_assets))
            
            # Function to minimize portfolio volatility for a given target return
            def portfolio_volatility(weights, returns):
                return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            
            # Constraint: weights sum to 1
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            )
            
            # Calculate asset expected returns
            asset_expected_returns = returns.mean() * 252
            
            # Calculate the range of target returns
            min_return = min(asset_expected_returns)
            max_return = max(asset_expected_returns)
            target_returns = np.linspace(min_return, max_return, points)
            
            # Calculate efficient frontier
            efficient_frontier = []
            for target_return in target_returns:
                # Add constraint for target return
                target_constraint = {
                    'type': 'eq',
                    'fun': lambda x: np.sum(x * asset_expected_returns) - target_return
                }
                
                # Initial weights (equal allocation)
                initial_weights = np.array([1/num_assets] * num_assets)
                
                # Run optimization for this target return
                result = sco.minimize(
                    portfolio_volatility,
                    initial_weights,
                    args=(returns,),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[constraints, target_constraint]
                )
                
                if result['success']:
                    optimal_weights = result['x']
                    volatility = portfolio_volatility(optimal_weights, returns)
                    sharpe = (target_return - self.risk_free_rate) / volatility
                    
                    efficient_frontier.append({
                        "return": target_return * 100,  # Convert to percentage
                        "volatility": volatility * 100,  # Convert to percentage
                        "sharpe_ratio": sharpe,
                        "weights": {ticker: weight for ticker, weight in zip(tickers, optimal_weights)}
                    })
            
            # Calculate the global minimum variance portfolio
            min_vol_result = sco.minimize(
                portfolio_volatility,
                np.array([1/num_assets] * num_assets),
                args=(returns,),
                method='SLSQP',
                bounds=bounds,
                constraints=[constraints]
            )
            
            if min_vol_result['success']:
                min_vol_weights = min_vol_result['x']
                min_vol_volatility = portfolio_volatility(min_vol_weights, returns)
                min_vol_return = np.sum(min_vol_weights * asset_expected_returns)
                min_vol_sharpe = (min_vol_return - self.risk_free_rate) / min_vol_volatility
                
                min_variance_portfolio = {
                    "return": min_vol_return * 100,  # Convert to percentage
                    "volatility": min_vol_volatility * 100,  # Convert to percentage
                    "sharpe_ratio": min_vol_sharpe,
                    "weights": {ticker: weight for ticker, weight in zip(tickers, min_vol_weights)}
                }
            else:
                min_variance_portfolio = None
            
            # Calculate the tangency portfolio (maximum Sharpe ratio)
            max_sharpe_result = self.optimize_portfolio(tickers, period)
            
            # Prepare results
            result = {
                "tickers": tickers,
                "efficient_frontier": efficient_frontier,
                "min_variance_portfolio": min_variance_portfolio,
                "max_sharpe_portfolio": {
                    "return": max_sharpe_result["expected_annual_return"],
                    "volatility": max_sharpe_result["annual_volatility"],
                    "sharpe_ratio": max_sharpe_result["sharpe_ratio"],
                    "weights": max_sharpe_result["weights"]
                },
                "risk_free_rate": self.risk_free_rate * 100  # Convert to percentage
            }
            
            return result
        except Exception as e:
            logger.error(f"Error calculating efficient frontier: {e}")
            return {"error": f"Error calculating efficient frontier: {str(e)}"}
    
    def analyze_portfolio(self, tickers: List[str], weights: List[float], 
                         period: str = "1y") -> Dict:
        """
        Analyze an existing portfolio
        
        Args:
            tickers: List of stock symbols
            weights: List of portfolio weights
            period: Time period for historical data
            
        Returns:
            Dictionary with portfolio analysis
        """
        try:
            # Validate inputs
            if len(tickers) != len(weights):
                return {"error": "Number of tickers must match number of weights"}
            
            if abs(sum(weights) - 1.0) > 0.0001:
                return {"error": "Weights must sum to 1"}
            
            # Get historical data
            prices = self.get_historical_data(tickers, period)
            if prices.empty:
                return {"error": f"No data found for tickers: {tickers}"}
            
            # Calculate returns
            returns = self.calculate_returns(prices)
            
            # Convert weights to numpy array
            weights_array = np.array(weights)
            
            # Calculate performance metrics
            expected_return, volatility, sharpe_ratio = self.calculate_portfolio_performance(returns, weights_array)
            
            # Calculate individual asset metrics
            asset_expected_returns = returns.mean() * 252
            asset_volatilities = np.sqrt(np.diag(returns.cov() * 252))
            asset_sharpe_ratios = (asset_expected_returns - self.risk_free_rate) / asset_volatilities
            
            # Calculate correlation matrix
            correlation_matrix = returns.corr().values.tolist()
            
            # Calculate portfolio beta (using S&P 500 as market proxy)
            try:
                sp500 = self.get_historical_data(["^GSPC"], period)
                sp500_returns = self.calculate_returns(sp500)
                
                # Align dates
                common_dates = returns.index.intersection(sp500_returns.index)
                aligned_returns = returns.loc[common_dates]
                aligned_sp500 = sp500_returns.loc[common_dates]
                
                # Calculate portfolio returns
                portfolio_returns = aligned_returns.dot(weights_array)
                
                # Calculate beta
                covariance = np.cov(portfolio_returns, aligned_sp500.iloc[:, 0])[0, 1]
                market_variance = np.var(aligned_sp500.iloc[:, 0])
                portfolio_beta = covariance / market_variance
            except Exception as e:
                logger.error(f"Error calculating beta: {e}")
                portfolio_beta = None
            
            # Calculate Value at Risk (VaR)
            portfolio_returns = returns.dot(weights_array)
            var_95 = np.percentile(portfolio_returns, 5) * 100  # 95% VaR (daily)
            var_99 = np.percentile(portfolio_returns, 1) * 100  # 99% VaR (daily)
            
            # Calculate Conditional VaR (CVaR) / Expected Shortfall
            cvar_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * 100
            
            # Calculate maximum drawdown
            portfolio_cumulative = (1 + portfolio_returns).cumprod()
            running_max = portfolio_cumulative.cummax()
            drawdown = (portfolio_cumulative / running_max - 1) * 100
            max_drawdown = drawdown.min()
            
            # Calculate portfolio performance
            portfolio_cumulative_return = (portfolio_cumulative.iloc[-1] - 1) * 100
            
            # Calculate performance attribution
            attribution = {}
            for i, ticker in enumerate(tickers):
                # Calculate contribution to return
                asset_contribution = weights[i] * asset_expected_returns[i] / expected_return
                attribution[ticker] = {
                    "weight": weights[i],
                    "expected_return": asset_expected_returns[i] * 100,
                    "contribution_to_return": asset_contribution * expected_return * 100,
                    "contribution_to_risk": weights[i] * asset_volatilities[i] / volatility * volatility * 100
                }
            
            # Calculate optimal portfolio for comparison
            optimal_portfolio = self.optimize_portfolio(tickers, period)
            
            # Calculate improvement potential
            if "error" not in optimal_portfolio:
                improvement = {
                    "current_sharpe": sharpe_ratio,
                    "optimal_sharpe": optimal_portfolio["sharpe_ratio"],
                    "potential_improvement": (optimal_portfolio["sharpe_ratio"] - sharpe_ratio) / sharpe_ratio * 100 if sharpe_ratio > 0 else float('inf'),
                    "suggested_weights": optimal_portfolio["weights"]
                }
            else:
                improvement = None
            
            # Prepare results
            result = {
                "tickers": tickers,
                "weights": {ticker: weight for ticker, weight in zip(tickers, weights)},
                "expected_annual_return": expected_return * 100,  # Convert to percentage
                "annual_volatility": volatility * 100,  # Convert to percentage
                "sharpe_ratio": sharpe_ratio,
                "portfolio_beta": portfolio_beta,
                "cumulative_return": portfolio_cumulative_return,
                "risk_metrics": {
                    "value_at_risk_95": -var_95,  # Daily VaR at 95% confidence (positive number)
                    "value_at_risk_99": -var_99,  # Daily VaR at 99% confidence (positive number)
                    "conditional_var_95": -cvar_95,  # Daily CVaR at 95% confidence (positive number)
                    "max_drawdown": -max_drawdown  # Maximum historical drawdown (positive number)
                },
                "asset_metrics": {
                    ticker: {
                        "weight": weight,
                        "expected_return": ret * 100,  # Convert to percentage
                        "volatility": vol * 100,  # Convert to percentage
                        "sharpe_ratio": sharpe
                    } for ticker, weight, ret, vol, sharpe in zip(
                        tickers, weights, asset_expected_returns, asset_volatilities, asset_sharpe_ratios
                    )
                },
                "attribution": attribution,
                "correlation_matrix": correlation_matrix,
                "improvement_potential": improvement
            }
            
            return result
        except Exception as e:
            logger.error(f"Error analyzing portfolio: {e}")
            return {"error": f"Error analyzing portfolio: {str(e)}"}
    
    def rebalance_portfolio(self, current_tickers: List[str], current_weights: List[float],
                           target_tickers: List[str] = None, target_weights: List[float] = None,
                           optimize: bool = False, period: str = "1y") -> Dict:
        """
        Generate portfolio rebalancing recommendations
        
        Args:
            current_tickers: List of current stock symbols
            current_weights: List of current portfolio weights
            target_tickers: List of target stock symbols (if different from current)
            target_weights: List of target portfolio weights (if not optimizing)
            optimize: Whether to optimize the target weights
            period: Time period for historical data
            
        Returns:
            Dictionary with rebalancing recommendations
        """
        try:
            # Validate current portfolio
            if len(current_tickers) != len(current_weights):
                return {"error": "Number of current tickers must match number of current weights"}
            
            if abs(sum(current_weights) - 1.0) > 0.0001:
                return {"error": "Current weights must sum to 1"}
            
            # Set target tickers and weights
            if target_tickers is None:
                target_tickers = current_tickers.copy()
            
            if optimize:
                # Optimize target portfolio
                optimal_portfolio = self.optimize_portfolio(target_tickers, period)
                if "error" in optimal_portfolio:
                    return optimal_portfolio
                
                target_weights_dict = optimal_portfolio["weights"]
                target_weights = [target_weights_dict.get(ticker, 0) for ticker in target_tickers]
            elif target_weights is not None:
                # Validate target weights
                if len(target_tickers) != len(target_weights):
                    return {"error": "Number of target tickers must match number of target weights"}
                
                if abs(sum(target_weights) - 1.0) > 0.0001:
                    return {"error": "Target weights must sum to 1"}
            else:
                return {"error": "Either target_weights must be provided or optimize must be True"}
            
            # Create dictionaries for easier comparison
            current_portfolio = {ticker: weight for ticker, weight in zip(current_tickers, current_weights)}
            target_portfolio = {ticker: weight for ticker, weight in zip(target_tickers, target_weights)}
            
            # Calculate trades needed
            all_tickers = list(set(current_tickers + target_tickers))
            trades = []
            
            for ticker in all_tickers:
                current_weight = current_portfolio.get(ticker, 0)
                target_weight = target_portfolio.get(ticker, 0)
                weight_difference = target_weight - current_weight
                
                if abs(weight_difference) > 0.0001:  # Ignore very small differences
                    action = "BUY" if weight_difference > 0 else "SELL"
                    trades.append({
                        "ticker": ticker,
                        "action": action,
                        "current_weight": current_weight,
                        "target_weight": target_weight,
                        "weight_difference": abs(weight_difference),
                        "percentage_change": abs(weight_difference) / (current_weight if current_weight > 0 else 0.0001) * 100 if current_weight > 0 else float('inf')
                    })
            
            # Sort trades by absolute weight difference (largest first)
            trades.sort(key=lambda x: x["weight_difference"], reverse=True)
            
            # Calculate portfolio turnover
            turnover = sum(abs(current_portfolio.get(ticker, 0) - target_portfolio.get(ticker, 0)) for ticker in all_tickers) / 2
            
            # Analyze current and target portfolios
            current_analysis = self.analyze_portfolio(current_tickers, current_weights, period)
            
            if optimize:
                target_analysis = optimal_portfolio
            else:
                target_analysis = self.analyze_portfolio(target_tickers, target_weights, period)
            
            # Calculate improvement metrics
            if "error" not in current_analysis and "error" not in target_analysis:
                sharpe_improvement = target_analysis["sharpe_ratio"] - current_analysis["sharpe_ratio"]
                return_improvement = target_analysis["expected_annual_return"] - current_analysis["expected_annual_return"]
                risk_change = target_analysis["annual_volatility"] - current_analysis["annual_volatility"]
                
                improvement_metrics = {
                    "sharpe_ratio_change": sharpe_improvement,
                    "expected_return_change": return_improvement,
                    "volatility_change": risk_change,
                    "risk_adjusted_improvement": sharpe_improvement / current_analysis["sharpe_ratio"] * 100 if current_analysis["sharpe_ratio"] > 0 else float('inf')
                }
            else:
                improvement_metrics = None
            
            # Prepare results
            result = {
                "current_portfolio": {
                    "tickers": current_tickers,
                    "weights": current_portfolio
                },
                "target_portfolio": {
                    "tickers": target_tickers,
                    "weights": target_portfolio,
                    "is_optimized": optimize
                },
                "trades": trades,
                "portfolio_turnover": turnover * 100,  # Convert to percentage
                "improvement_metrics": improvement_metrics,
                "current_portfolio_metrics": {
                    "expected_return": current_analysis.get("expected_annual_return"),
                    "volatility": current_analysis.get("annual_volatility"),
                    "sharpe_ratio": current_analysis.get("sharpe_ratio")
                } if "error" not in current_analysis else {"error": current_analysis["error"]},
                "target_portfolio_metrics": {
                    "expected_return": target_analysis.get("expected_annual_return"),
                    "volatility": target_analysis.get("annual_volatility"),
                    "sharpe_ratio": target_analysis.get("sharpe_ratio")
                } if "error" not in target_analysis else {"error": target_analysis["error"]}
            }
            
            return result
        except Exception as e:
            logger.error(f"Error generating rebalancing recommendations: {e}")
            return {"error": f"Error generating rebalancing recommendations: {str(e)}"}
    
    def stress_test_portfolio(self, tickers: List[str], weights: List[float], 
                             scenarios: List[str] = None) -> Dict:
        """
        Perform stress testing on a portfolio
        
        Args:
            tickers: List of stock symbols
            weights: List of portfolio weights
            scenarios: List of historical stress scenarios to test
            
        Returns:
            Dictionary with stress test results
        """
        try:
            # Validate inputs
            if len(tickers) != len(weights):
                return {"error": "Number of tickers must match number of weights"}
            
            if abs(sum(weights) - 1.0) > 0.0001:
                return {"error": "Weights must sum to 1"}
            
            # Define historical stress scenarios
            default_scenarios = [
                "financial_crisis_2008",  # 2008 Financial Crisis
                "covid_crash_2020",       # COVID-19 Crash
                "tech_bubble_2000",       # Dot-com Bubble Burst
                "rate_hike_2022",         # 2022 Rate Hikes
                "inflation_shock"         # Inflation Shock Scenario
            ]
            
            if scenarios is None:
                scenarios = default_scenarios
            
            # Define scenario date ranges and descriptions
            scenario_definitions = {
                "financial_crisis_2008": {
                    "start_date": "2008-09-01",
                    "end_date": "2009-03-01",
                    "description": "2008 Financial Crisis (Lehman Brothers collapse and aftermath)"
                },
                "covid_crash_2020": {
                    "start_date": "2020-02-15",
                    "end_date": "2020-03-23",
                    "description": "COVID-19 Market Crash (Initial pandemic shock)"
                },
                "tech_bubble_2000": {
                    "start_date": "2000-03-01",
                    "end_date": "2002-10-01",
                    "description": "Dot-com Bubble Burst (Tech market crash)"
                },
                "rate_hike_2022": {
                    "start_date": "2022-01-01",
                    "end_date": "2022-06-30",
                    "description": "2022 Rate Hikes (Fed tightening cycle)"
                },
                "inflation_shock": {
                    "start_date": "2021-10-01",
                    "end_date": "2022-06-30",
                    "description": "Inflation Shock (High inflation period)"
                }
            }
            
            # Filter scenarios to those that are defined
            scenarios = [s for s in scenarios if s in scenario_definitions]
            
            if not scenarios:
                return {"error": "No valid scenarios specified"}
            
            # Get historical data for each scenario
            scenario_results = {}
            
            for scenario in scenarios:
                scenario_def = scenario_definitions[scenario]
                start_date = scenario_def["start_date"]
                end_date = scenario_def["end_date"]
                
                try:
                    # Get data for the scenario period
                    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
                    
                    # Extract close prices
                    if len(tickers) == 1:
                        ticker = tickers[0]
                        prices = pd.DataFrame({ticker: data['Close']})
                    else:
                        prices = pd.DataFrame()
                        for ticker in tickers:
                            if (ticker, 'Close') in data.columns:
                                prices[ticker] = data[(ticker, 'Close')]
                    
                    if prices.empty:
                        scenario_results[scenario] = {
                            "error": f"No data available for scenario period {start_date} to {end_date}"
                        }
                        continue
                    
                    # Calculate returns
                    returns = self.calculate_returns(prices)
                    
                    if returns.empty:
                        scenario_results[scenario] = {
                            "error": f"Insufficient data for returns calculation in scenario period"
                        }
                        continue
                    
                    # Calculate portfolio returns
                    weights_array = np.array(weights)
                    portfolio_returns = returns.dot(weights_array)
                    
                    # Calculate cumulative return
                    cumulative_return = (1 + portfolio_returns).cumprod().iloc[-1] - 1
                    
                    # Calculate maximum drawdown
                    portfolio_cumulative = (1 + portfolio_returns).cumprod()
                    running_max = portfolio_cumulative.cummax()
                    drawdown = (portfolio_cumulative / running_max - 1)
                    max_drawdown = drawdown.min()
                    
                    # Calculate volatility
                    volatility = portfolio_returns.std() * np.sqrt(252)
                    
                    # Calculate worst day
                    worst_day = portfolio_returns.min()
                    worst_day_date = portfolio_returns.idxmin()
                    
                    # Calculate recovery time (if applicable)
                    if cumulative_return < 0:
                        recovery_time = "N/A - Did not recover by end of scenario"
                    else:
                        # Find the lowest point
                        lowest_point_idx = portfolio_cumulative.idxmin()
                        if lowest_point_idx == portfolio_cumulative.index[-1]:
                            recovery_time = "N/A - Still declining at end of scenario"
                        else:
                            # Calculate days from lowest point to recovery
                            recovery_days = (portfolio_cumulative.index[-1] - lowest_point_idx).days
                            recovery_time = f"{recovery_days} days"
                    
                    # Store results
                    scenario_results[scenario] = {
                        "description": scenario_def["description"],
                        "period": f"{start_date} to {end_date}",
                        "cumulative_return": cumulative_return * 100,  # Convert to percentage
                        "max_drawdown": max_drawdown * 100,  # Convert to percentage
                        "annualized_volatility": volatility * 100,  # Convert to percentage
                        "worst_day_return": worst_day * 100,  # Convert to percentage
                        "worst_day_date": worst_day_date.strftime('%Y-%m-%d') if isinstance(worst_day_date, pd.Timestamp) else str(worst_day_date),
                        "recovery_time": recovery_time
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing scenario {scenario}: {e}")
                    scenario_results[scenario] = {
                        "error": f"Error processing scenario: {str(e)}"
                    }
            
            # Calculate overall stress test summary
            valid_scenarios = [s for s in scenario_results if "error" not in scenario_results[s]]
            
            if not valid_scenarios:
                return {
                    "error": "No valid scenarios could be processed",
                    "scenario_results": scenario_results
                }
            
            # Calculate average impact across scenarios
            avg_return = np.mean([scenario_results[s]["cumulative_return"] for s in valid_scenarios])
            avg_drawdown = np.mean([scenario_results[s]["max_drawdown"] for s in valid_scenarios])
            worst_scenario = min(valid_scenarios, key=lambda s: scenario_results[s]["cumulative_return"])
            best_scenario = max(valid_scenarios, key=lambda s: scenario_results[s]["cumulative_return"])
            
            # Prepare results
            result = {
                "tickers": tickers,
                "weights": {ticker: weight for ticker, weight in zip(tickers, weights)},
                "scenarios_tested": len(valid_scenarios),
                "summary": {
                    "average_return": avg_return,
                    "average_max_drawdown": avg_drawdown,
                    "worst_scenario": {
                        "name": worst_scenario,
                        "description": scenario_results[worst_scenario]["description"],
                        "return": scenario_results[worst_scenario]["cumulative_return"]
                    },
                    "best_scenario": {
                        "name": best_scenario,
                        "description": scenario_results[best_scenario]["description"],
                        "return": scenario_results[best_scenario]["cumulative_return"]
                    }
                },
                "scenario_results": scenario_results
            }
            
            return result
        except Exception as e:
            logger.error(f"Error performing stress test: {e}")
            return {"error": f"Error performing stress test: {str(e)}"}

# Helper functions for portfolio management
def get_optimal_portfolio(tickers: List[str], period: str = "1y") -> Dict:
    """
    Get the optimal portfolio for a list of tickers
    
    Args:
        tickers: List of stock symbols
        period: Time period for historical data
        
    Returns:
        Dictionary with optimal portfolio
    """
    try:
        portfolio_manager = PortfolioManager()
        result = portfolio_manager.optimize_portfolio(tickers, period)
        return result
    except Exception as e:
        logger.error(f"Error getting optimal portfolio: {e}")
        return {"error": f"Error getting optimal portfolio: {str(e)}"}

def analyze_existing_portfolio(tickers: List[str], weights: List[float], period: str = "1y") -> Dict:
    """
    Analyze an existing portfolio
    
    Args:
        tickers: List of stock symbols
        weights: List of portfolio weights
        period: Time period for historical data
        
    Returns:
        Dictionary with portfolio analysis
    """
    try:
        portfolio_manager = PortfolioManager()
        result = portfolio_manager.analyze_portfolio(tickers, weights, period)
        return result
    except Exception as e:
        logger.error(f"Error analyzing portfolio: {e}")
        return {"error": f"Error analyzing portfolio: {str(e)}"}

def get_efficient_frontier(tickers: List[str], period: str = "1y", points: int = 20) -> Dict:
    """
    Get the efficient frontier for a list of tickers
    
    Args:
        tickers: List of stock symbols
        period: Time period for historical data
        points: Number of points on the efficient frontier
        
    Returns:
        Dictionary with efficient frontier data
    """
    try:
        portfolio_manager = PortfolioManager()
        result = portfolio_manager.calculate_efficient_frontier(tickers, period, points)
        return result
    except Exception as e:
        logger.error(f"Error calculating efficient frontier: {e}")
        return {"error": f"Error calculating efficient frontier: {str(e)}"}

def get_portfolio_rebalance(current_tickers: List[str], current_weights: List[float],
                           target_tickers: List[str] = None, target_weights: List[float] = None,
                           optimize: bool = False, period: str = "1y") -> Dict:
    """
    Get portfolio rebalancing recommendations
    
    Args:
        current_tickers: List of current stock symbols
        current_weights: List of current portfolio weights
        target_tickers: List of target stock symbols (if different from current)
        target_weights: List of target portfolio weights (if not optimizing)
        optimize: Whether to optimize the target weights
        period: Time period for historical data
        
    Returns:
        Dictionary with rebalancing recommendations
    """
    try:
        portfolio_manager = PortfolioManager()
        result = portfolio_manager.rebalance_portfolio(
            current_tickers, current_weights, target_tickers, target_weights, optimize, period
        )
        return result
    except Exception as e:
        logger.error(f"Error generating rebalancing recommendations: {e}")
        return {"error": f"Error generating rebalancing recommendations: {str(e)}"}

def stress_test_portfolio(tickers: List[str], weights: List[float], scenarios: List[str] = None) -> Dict:
    """
    Perform stress testing on a portfolio
    
    Args:
        tickers: List of stock symbols
        weights: List of portfolio weights
        scenarios: List of historical stress scenarios to test
        
    Returns:
        Dictionary with stress test results
    """
    try:
        portfolio_manager = PortfolioManager()
        result = portfolio_manager.stress_test_portfolio(tickers, weights, scenarios)
        return result
    except Exception as e:
        logger.error(f"Error performing stress test: {e}")
        return {"error": f"Error performing stress test: {str(e)}"}
