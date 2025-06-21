# tools/visualization.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Union, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import base64
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory to store visualization images
VIZ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)

# Set the style for all visualizations
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def generate_stock_price_chart(ticker: str, period: str = "1y", interval: str = "1d", 
                              chart_type: str = "line", moving_averages: List[int] = None,
                              volume: bool = True, save_path: Optional[str] = None) -> str:
    """
    Generate a stock price chart with optional moving averages and volume
    
    Args:
        ticker: Stock symbol
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        chart_type: Type of chart (line, candle)
        moving_averages: List of moving average periods to include (e.g., [20, 50, 200])
        volume: Whether to include volume
        save_path: Path to save the chart image (if None, will save to default location)
        
    Returns:
        Path to the saved chart image
    """
    try:
        # Validate ticker
        if not ticker or not isinstance(ticker, str):
            logger.error("Invalid ticker provided")
            return ""
        
        # Clean and standardize ticker
        ticker = ticker.strip().upper()
        if not ticker:
            logger.error("Empty ticker name after cleaning")
            return ""
        
        # Get stock data
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        
        if data.empty:
            logger.error(f"No data found for {ticker}")
            return ""
        
        # Create figure with subplots
        if volume:
            fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        else:
            fig, ax1 = plt.subplots(1, 1)
        
        # Plot price data
        if chart_type.lower() == "line":
            ax1.plot(data.index, data['Close'], label='Close Price', linewidth=2)
        elif chart_type.lower() == "candle":
            # Simple candlestick implementation
            for i in range(len(data)):
                date = data.index[i]
                open_price = data['Open'].iloc[i]
                close_price = data['Close'].iloc[i]
                high_price = data['High'].iloc[i]
                low_price = data['Low'].iloc[i]
                
                # Determine color based on price movement
                color = 'green' if close_price >= open_price else 'red'
                
                # Plot candlestick body
                ax1.plot([date, date], [open_price, close_price], color=color, linewidth=4)
                
                # Plot high/low wicks
                ax1.plot([date, date], [low_price, high_price], color=color, linewidth=1)
        
        # Add moving averages if requested
        if moving_averages:
            for ma in moving_averages:
                if len(data) >= ma:
                    ma_data = data['Close'].rolling(window=ma).mean()
                    ax1.plot(data.index, ma_data, label=f'{ma}-day MA', linewidth=1.5)
        
        # Add volume if requested
        if volume:
            ax2.bar(data.index, data['Volume'], color='gray', alpha=0.5)
            ax2.set_ylabel('Volume')
        
        # Set labels and title
        ax1.set_title(f"{ticker} Stock Price ({period})")
        ax1.set_ylabel('Price')
        if not volume:
            ax1.set_xlabel('Date')
        
        # Add legend
        ax1.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the chart
        if save_path is None:
            save_path = os.path.join(VIZ_DIR, f"{ticker}_price_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    except Exception as e:
        logger.error(f"Error generating stock price chart: {e}")
        return ""

def generate_comparison_chart(tickers: List[str], period: str = "1y", metric: str = "price",
                             chart_type: str = "line", normalize: bool = True,
                             save_path: Optional[str] = None) -> str:
    """
    Generate a comparison chart for multiple stocks
    
    Args:
        tickers: List of stock symbols
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        metric: Metric to compare (price, returns, volume)
        chart_type: Type of chart (line, bar)
        normalize: Whether to normalize data to the same starting point (for price/returns)
        save_path: Path to save the chart image (if None, will save to default location)
        
    Returns:
        Path to the saved chart image
    """
    try:
        # Get data for all tickers
        data_dict = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            if not data.empty:
                data_dict[ticker] = data
        
        if not data_dict:
            logger.error(f"No data found for any of the tickers: {tickers}")
            return ""
        
        # Create figure
        plt.figure()
        
        # Plot data based on metric and chart type
        if metric.lower() == "price":
            for ticker, data in data_dict.items():
                price_data = data['Close']
                
                # Normalize if requested
                if normalize:
                    price_data = price_data / price_data.iloc[0] * 100
                
                if chart_type.lower() == "line":
                    plt.plot(data.index, price_data, label=ticker, linewidth=2)
                elif chart_type.lower() == "bar":
                    # For bar charts, just show the current value compared to initial
                    if normalize:
                        initial_value = 100
                    else:
                        initial_value = data['Close'].iloc[0]
                    current_value = data['Close'].iloc[-1]
                    
                    # We'll collect these and plot them all at once after the loop
                    bar_data = {ticker: current_value}
            
            if chart_type.lower() == "line":
                plt.title(f"{'Normalized ' if normalize else ''}Price Comparison ({period})")
                plt.ylabel(f"{'Normalized Price (%)' if normalize else 'Price'}")
                plt.xlabel("Date")
            elif chart_type.lower() == "bar":
                # Plot bar chart with all tickers
                tickers_list = list(data_dict.keys())
                values = [data_dict[t]['Close'].iloc[-1] / data_dict[t]['Close'].iloc[0] * 100 if normalize 
                         else data_dict[t]['Close'].iloc[-1] for t in tickers_list]
                
                plt.bar(tickers_list, values)
                plt.title(f"{'Normalized ' if normalize else ''}Current Price Comparison")
                plt.ylabel(f"{'Normalized Price (%)' if normalize else 'Price'}")
                plt.xticks(rotation=45)
        
        elif metric.lower() == "returns":
            for ticker, data in data_dict.items():
                # Calculate returns
                returns = data['Close'].pct_change().cumsum() * 100
                
                if chart_type.lower() == "line":
                    plt.plot(data.index[1:], returns[1:], label=ticker, linewidth=2)
                elif chart_type.lower() == "bar":
                    # For bar charts, just show the total return
                    total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                    bar_data = {ticker: total_return}
            
            if chart_type.lower() == "line":
                plt.title(f"Cumulative Returns Comparison ({period})")
                plt.ylabel("Cumulative Returns (%)")
                plt.xlabel("Date")
            elif chart_type.lower() == "bar":
                # Plot bar chart with all tickers
                tickers_list = list(data_dict.keys())
                values = [(data_dict[t]['Close'].iloc[-1] / data_dict[t]['Close'].iloc[0] - 1) * 100 
                         for t in tickers_list]
                
                plt.bar(tickers_list, values)
                plt.title(f"Total Returns Comparison ({period})")
                plt.ylabel("Total Return (%)")
                plt.xticks(rotation=45)
        
        elif metric.lower() == "volume":
            for ticker, data in data_dict.items():
                if chart_type.lower() == "line":
                    plt.plot(data.index, data['Volume'], label=ticker, linewidth=2)
                elif chart_type.lower() == "bar":
                    # For bar charts, show average daily volume
                    avg_volume = data['Volume'].mean()
                    bar_data = {ticker: avg_volume}
            
            if chart_type.lower() == "line":
                plt.title(f"Volume Comparison ({period})")
                plt.ylabel("Volume")
                plt.xlabel("Date")
            elif chart_type.lower() == "bar":
                # Plot bar chart with all tickers
                tickers_list = list(data_dict.keys())
                values = [data_dict[t]['Volume'].mean() for t in tickers_list]
                
                plt.bar(tickers_list, values)
                plt.title(f"Average Daily Volume Comparison ({period})")
                plt.ylabel("Average Volume")
                plt.xticks(rotation=45)
        
        # Add legend for line charts
        if chart_type.lower() == "line":
            plt.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the chart
        if save_path is None:
            tickers_str = "_".join(tickers)
            save_path = os.path.join(VIZ_DIR, f"{tickers_str}_{metric}_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    except Exception as e:
        logger.error(f"Error generating comparison chart: {e}")
        return ""

def generate_financial_metrics_chart(ticker: str, metrics: List[str] = None, 
                                    chart_type: str = "bar", save_path: Optional[str] = None) -> str:
    """
    Generate a chart of financial metrics for a stock
    
    Args:
        ticker: Stock symbol
        metrics: List of metrics to include (if None, will use default set)
        chart_type: Type of chart (bar, radar)
        save_path: Path to save the chart image (if None, will save to default location)
        
    Returns:
        Path to the saved chart image
    """
    try:
        # Get stock data
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Default metrics if none provided
        if metrics is None:
            metrics = ['trailingPE', 'forwardPE', 'priceToBook', 'profitMargins', 
                      'returnOnEquity', 'returnOnAssets', 'debtToEquity']
        
        # Collect available metrics
        metric_data = {}
        for metric in metrics:
            if metric in info and info[metric] is not None:
                metric_data[metric] = info[metric]
        
        if not metric_data:
            logger.error(f"No metrics found for {ticker}")
            return ""
        
        # Create figure
        plt.figure()
        
        # Plot based on chart type
        if chart_type.lower() == "bar":
            # Create bar chart
            plt.bar(metric_data.keys(), metric_data.values())
            plt.title(f"{ticker} Financial Metrics")
            plt.ylabel("Value")
            plt.xticks(rotation=45)
        
        elif chart_type.lower() == "radar":
            # Create radar chart
            # Convert to numpy arrays for radar chart
            categories = list(metric_data.keys())
            values = list(metric_data.values())
            
            # Number of variables
            N = len(categories)
            
            # Create angles for each metric
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Add the first value at the end to close the loop
            values += values[:1]
            
            # Create the plot
            ax = plt.subplot(111, polar=True)
            
            # Draw the shape
            ax.plot(angles, values, linewidth=2, linestyle='solid')
            ax.fill(angles, values, alpha=0.25)
            
            # Add labels
            plt.xticks(angles[:-1], categories, size=10)
            
            # Add title
            plt.title(f"{ticker} Financial Metrics")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the chart
        if save_path is None:
            save_path = os.path.join(VIZ_DIR, f"{ticker}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    except Exception as e:
        logger.error(f"Error generating financial metrics chart: {e}")
        return ""

def generate_industry_comparison_chart(ticker: str, metrics: List[str] = None,
                                      chart_type: str = "bar", save_path: Optional[str] = None) -> str:
    """
    Generate a chart comparing a stock to its industry peers
    
    Args:
        ticker: Stock symbol
        metrics: List of metrics to include (if None, will use default set)
        chart_type: Type of chart (bar, radar)
        save_path: Path to save the chart image (if None, will save to default location)
        
    Returns:
        Path to the saved chart image
    """
    try:
        # Import the industry comparison function
        from tools.financial_analysis import get_industry_comparison
        
        # Get industry comparison data
        comparison = get_industry_comparison(ticker)
        
        if 'error' in comparison:
            logger.error(f"Error getting industry comparison: {comparison['error']}")
            return ""
        
        # Default metrics if none provided
        if metrics is None:
            metrics = ['pe_ratio', 'price_to_book', 'profit_margin', 'roe', 'revenue_growth']
        
        # Extract company and industry metrics
        company_metrics = comparison.get('company_metrics', {})
        industry_averages = comparison.get('industry_averages', {})
        
        # Collect available metrics
        metric_data = {}
        for metric in metrics:
            if metric in company_metrics and metric in industry_averages:
                if company_metrics[metric] is not None and industry_averages[metric] is not None:
                    metric_data[metric] = {
                        'company': float(company_metrics[metric]),
                        'industry': float(industry_averages[metric])
                    }
        
        if not metric_data:
            logger.error(f"No comparable metrics found for {ticker}")
            return ""
        
        # Create figure
        plt.figure()
        
        # Plot based on chart type
        if chart_type.lower() == "bar":
            # Set up bar positions
            metrics_list = list(metric_data.keys())
            x = np.arange(len(metrics_list))
            width = 0.35
            
            # Create bars
            company_values = [metric_data[m]['company'] for m in metrics_list]
            industry_values = [metric_data[m]['industry'] for m in metrics_list]
            
            plt.bar(x - width/2, company_values, width, label=ticker)
            plt.bar(x + width/2, industry_values, width, label='Industry Average')
            
            # Add labels and title
            plt.title(f"{ticker} vs Industry Comparison")
            plt.ylabel("Value")
            plt.xticks(x, metrics_list, rotation=45)
            plt.legend()
        
        elif chart_type.lower() == "radar":
            # Create radar chart
            # Convert to numpy arrays for radar chart
            categories = list(metric_data.keys())
            company_values = [metric_data[m]['company'] for m in categories]
            industry_values = [metric_data[m]['industry'] for m in categories]
            
            # Number of variables
            N = len(categories)
            
            # Create angles for each metric
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Add the first value at the end to close the loop
            company_values += company_values[:1]
            industry_values += industry_values[:1]
            
            # Create the plot
            ax = plt.subplot(111, polar=True)
            
            # Draw the shapes
            ax.plot(angles, company_values, linewidth=2, linestyle='solid', label=ticker)
            ax.fill(angles, company_values, alpha=0.25)
            
            ax.plot(angles, industry_values, linewidth=2, linestyle='solid', label='Industry Average')
            ax.fill(angles, industry_values, alpha=0.1)
            
            # Add labels
            plt.xticks(angles[:-1], categories, size=10)
            
            # Add legend and title
            plt.legend(loc='upper right')
            plt.title(f"{ticker} vs Industry Comparison")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the chart
        if save_path is None:
            save_path = os.path.join(VIZ_DIR, f"{ticker}_industry_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    except Exception as e:
        logger.error(f"Error generating industry comparison chart: {e}")
        return ""

def generate_correlation_matrix(tickers: List[str], period: str = "1y", 
                               save_path: Optional[str] = None) -> str:
    """
    Generate a correlation matrix for multiple stocks
    
    Args:
        tickers: List of stock symbols
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        save_path: Path to save the chart image (if None, will save to default location)
        
    Returns:
        Path to the saved chart image
    """
    try:
        # Get data for all tickers
        price_data = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            if not data.empty:
                price_data[ticker] = data['Close']
        
        if not price_data:
            logger.error(f"No data found for any of the tickers: {tickers}")
            return ""
        
        # Create DataFrame with all price data
        df = pd.DataFrame(price_data)
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                   square=True, linewidths=.5, cbar_kws={"shrink": .8})
        
        # Add title
        plt.title(f"Correlation Matrix ({period})")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the chart
        if save_path is None:
            tickers_str = "_".join(tickers)
            save_path = os.path.join(VIZ_DIR, f"{tickers_str}_correlation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    except Exception as e:
        logger.error(f"Error generating correlation matrix: {e}")
        return ""

def generate_distribution_chart(ticker: str, period: str = "1y", metric: str = "returns",
                               chart_type: str = "histogram", save_path: Optional[str] = None) -> str:
    """
    Generate a distribution chart for a stock
    
    Args:
        ticker: Stock symbol
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        metric: Metric to analyze (returns, volume)
        chart_type: Type of chart (histogram, kde, box)
        save_path: Path to save the chart image (if None, will save to default location)
        
    Returns:
        Path to the saved chart image
    """
    try:
        # Get stock data
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        
        if data.empty:
            logger.error(f"No data found for {ticker}")
            return ""
        
        # Create figure
        plt.figure()
        
        # Prepare data based on metric
        if metric.lower() == "returns":
            # Calculate daily returns
            returns = data['Close'].pct_change().dropna() * 100
            plot_data = returns
            title = f"{ticker} Daily Returns Distribution ({period})"
            xlabel = "Daily Returns (%)"
        elif metric.lower() == "volume":
            plot_data = data['Volume']
            title = f"{ticker} Volume Distribution ({period})"
            xlabel = "Volume"
        else:
            logger.error(f"Unsupported metric: {metric}")
            return ""
        
        # Plot based on chart type
        if chart_type.lower() == "histogram":
            sns.histplot(plot_data, kde=True)
        elif chart_type.lower() == "kde":
            sns.kdeplot(plot_data, fill=True)
        elif chart_type.lower() == "box":
            sns.boxplot(y=plot_data)
            xlabel = ""  # No x-label for boxplot
        else:
            logger.error(f"Unsupported chart type: {chart_type}")
            return ""
        
        # Add labels and title
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Frequency" if chart_type.lower() != "box" else metric.capitalize())
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the chart
        if save_path is None:
            save_path = os.path.join(VIZ_DIR, f"{ticker}_{metric}_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    except Exception as e:
        logger.error(f"Error generating distribution chart: {e}")
        return ""

def generate_forecast_chart(ticker: str, forecast_data: Dict = None, days: int = 20, 
                           save_path: Optional[str] = None) -> str:
    """
    Generate a forecast chart for a stock with confidence intervals
    
    Args:
        ticker: Stock symbol
        forecast_data: Dictionary containing forecast data (if None, will fetch it)
        days: Number of days to forecast
        save_path: Path to save the chart image (if None, will save to default location)
        
    Returns:
        Path to the saved chart image
    """
    try:
        # Validate ticker
        if not ticker or not isinstance(ticker, str):
            logger.error("Invalid ticker provided for forecast chart")
            return ""
        
        # Clean and standardize ticker
        ticker = ticker.strip().upper()
        if not ticker:
            logger.error("Empty ticker name after cleaning for forecast chart")
            return ""
        
        # If forecast_data is not provided, fetch it
        if forecast_data is None:
            # Import the forecast function
            from tools.forecast import get_forecast
            
            # Get forecast data
            forecast_data = get_forecast(ticker, forecast_days=days)
            
            if 'error' in forecast_data:
                logger.error(f"Error getting forecast data: {forecast_data['error']}")
                return ""
        
        # Get stock data for historical prices
        stock = yf.Ticker(ticker)
        historical_data = stock.history(period="6mo")
        
        if historical_data.empty:
            logger.error(f"No historical data found for {ticker}")
            return ""
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot historical prices
        plt.plot(historical_data.index[-30:], historical_data['Close'][-30:], 
                label='Historical Price', color='blue', linewidth=2)
        
        # Get current price and forecast
        current_price = forecast_data.get('current_price', historical_data['Close'].iloc[-1])
        predicted_return = forecast_data.get('predicted_return', 0)
        predicted_price = forecast_data.get('predicted_price', current_price * (1 + predicted_return/100))
        
        # Create forecast dates
        last_date = historical_data.index[-1]
        forecast_dates = [last_date + timedelta(days=i) for i in range(days+1)]
        
        # Create forecast prices (linear interpolation)
        forecast_prices = [current_price]
        for i in range(1, days+1):
            forecast_prices.append(current_price + (predicted_price - current_price) * i / days)
        
        # Plot forecast line
        plt.plot(forecast_dates, forecast_prices, label='Price Forecast', 
                color='green', linewidth=2, linestyle='--')
        
        # Add confidence intervals if available
        if 'confidence_intervals' in forecast_data:
            intervals = forecast_data['confidence_intervals']
            
            if '90%' in intervals:
                interval_90 = intervals['90%']
                lower_price = interval_90.get('lower_price', predicted_price * 0.9)
                upper_price = interval_90.get('upper_price', predicted_price * 1.1)
                
                # Create confidence interval bands
                lower_band = [current_price]
                upper_band = [current_price]
                
                for i in range(1, days+1):
                    lower_band.append(current_price + (lower_price - current_price) * i / days)
                    upper_band.append(current_price + (upper_price - current_price) * i / days)
                
                # Plot confidence intervals
                plt.fill_between(forecast_dates, lower_band, upper_band, 
                                color='green', alpha=0.2, label='90% Confidence Interval')
        
        # Add labels and title
        plt.title(f"{ticker} Price Forecast ({days} days)")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add annotations
        plt.annotate(f"Current: ${current_price:.2f}", 
                    xy=(forecast_dates[0], current_price),
                    xytext=(forecast_dates[0], current_price * 1.05),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=10)
        
        plt.annotate(f"Forecast: ${predicted_price:.2f} ({'+' if predicted_return > 0 else ''}{predicted_return:.2f}%)", 
                    xy=(forecast_dates[-1], predicted_price),
                    xytext=(forecast_dates[-1], predicted_price * 1.05),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the chart
        if save_path is None:
            save_path = os.path.join(VIZ_DIR, f"{ticker}_forecast_{days}d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    except Exception as e:
        logger.error(f"Error generating forecast chart: {e}")
        return ""

def generate_visualization(viz_type: str, **kwargs) -> Dict:
    """
    Generate a visualization based on the specified type and parameters
    
    Args:
        viz_type: Type of visualization to generate
        **kwargs: Additional parameters for the specific visualization
        
    Returns:
        Dictionary with visualization information
    """
    try:
        # Determine which visualization function to call based on viz_type
        if viz_type.lower() == "forecast_chart":
            ticker = kwargs.get("ticker", "")
            
            # If forecast_data is provided as a string, convert it to dict
            forecast_data = kwargs.get("forecast_data", None)
            if isinstance(forecast_data, str):
                try:
                    forecast_data = eval(forecast_data)
                except:
                    forecast_data = None
            
            # If ticker is not provided but forecast_data is, try to extract ticker
            if not ticker and forecast_data and isinstance(forecast_data, dict):
                ticker = forecast_data.get("ticker", "")
            
            days = kwargs.get("days", 20)
            
            image_path = generate_forecast_chart(
                ticker=ticker,
                forecast_data=forecast_data,
                days=days
            )
            
            description = f"{ticker} price forecast chart ({days} days)"
        
        elif viz_type.lower() == "price_chart":
            ticker = kwargs.get("ticker", "")
            period = kwargs.get("period", "1y")
            interval = kwargs.get("interval", "1d")
            chart_type = kwargs.get("chart_type", "line")
            moving_averages = kwargs.get("moving_averages", None)
            volume = kwargs.get("volume", True)
            
            image_path = generate_stock_price_chart(
                ticker=ticker,
                period=period,
                interval=interval,
                chart_type=chart_type,
                moving_averages=moving_averages,
                volume=volume
            )
            
            description = f"{ticker} stock price chart ({period})"
        
        elif viz_type.lower() == "comparison_chart":
            tickers = kwargs.get("tickers", [])
            period = kwargs.get("period", "1y")
            metric = kwargs.get("metric", "price")
            chart_type = kwargs.get("chart_type", "line")
            normalize = kwargs.get("normalize", True)
            
            image_path = generate_comparison_chart(
                tickers=tickers,
                period=period,
                metric=metric,
                chart_type=chart_type,
                normalize=normalize
            )
            
            description = f"Comparison chart of {', '.join(tickers)} ({metric}, {period})"
        
        elif viz_type.lower() == "metrics_chart":
            ticker = kwargs.get("ticker", "")
            metrics = kwargs.get("metrics", None)
            chart_type = kwargs.get("chart_type", "bar")
            
            image_path = generate_financial_metrics_chart(
                ticker=ticker,
                metrics=metrics,
                chart_type=chart_type
            )
            
            description = f"{ticker} financial metrics chart"
        
        elif viz_type.lower() == "industry_chart":
            ticker = kwargs.get("ticker", "")
            metrics = kwargs.get("metrics", None)
            chart_type = kwargs.get("chart_type", "bar")
            
            image_path = generate_industry_comparison_chart(
                ticker=ticker,
                metrics=metrics,
                chart_type=chart_type
            )
            
            description = f"{ticker} industry comparison chart"
        
        elif viz_type.lower() == "correlation_matrix":
            tickers = kwargs.get("tickers", [])
            period = kwargs.get("period", "1y")
            
            image_path = generate_correlation_matrix(
                tickers=tickers,
                period=period
            )
            
            description = f"Correlation matrix of {', '.join(tickers)} ({period})"
        
        elif viz_type.lower() == "distribution_chart":
            ticker = kwargs.get("ticker", "")
            period = kwargs.get("period", "1y")
            metric = kwargs.get("metric", "returns")
            chart_type = kwargs.get("chart_type", "histogram")
            
            image_path = generate_distribution_chart(
                ticker=ticker,
                period=period,
                metric=metric,
                chart_type=chart_type
            )
            
            description = f"{ticker} {metric} distribution chart ({period})"
        
        else:
            logger.error(f"Unsupported visualization type: {viz_type}")
            return {
                "error": f"Unsupported visualization type: {viz_type}",
                "supported_types": ["price_chart", "comparison_chart", "metrics_chart", 
                                   "industry_chart", "correlation_matrix", "distribution_chart"]
            }
        
        # Check if image was generated successfully
        if not image_path:
            return {
                "error": f"Failed to generate {viz_type} visualization",
                "viz_type": viz_type,
                "parameters": kwargs
            }
        
        # Return visualization information
        return {
            "viz_type": viz_type,
            "description": description,
            "image_path": image_path,
            "parameters": kwargs,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        logger.error(f"Error generating visualization: {e}")
        return {
            "error": str(e),
            "viz_type": viz_type,
            "parameters": kwargs
        }
