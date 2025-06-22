# tools/financial_analysis.py

import os
import re
import json
import time
import pickle
import numpy as np
import pandas as pd
import asyncio
import aiohttp
from typing import Dict, List, Union, Optional, Tuple, Any
from datetime import datetime, timedelta
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import logging
import concurrent.futures
from functools import partial

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory to store cached data
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Directory to store vector embeddings
VECTOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../vectors")
os.makedirs(VECTOR_DIR, exist_ok=True)

# Cache TTL in days for different data types
CACHE_TTL = {
    'default': 7,  # Default: 7 days (1 week)
    'price_data': 1,  # Price data: 1 day
    'fundamentals': 3,  # Fundamental data: 3 days
    'filings': 14,  # SEC filings: 14 days (2 weeks)
    'transcripts': 30,  # Earnings call transcripts: 30 days (1 month)
    'metrics': 2,  # Key metrics: 2 days
}

class FinancialAnalyzer:
    """Class to handle comprehensive financial analysis including statements and earnings"""
    
    def __init__(self, ticker: str):
        """
        Initialize the financial analyzer for a specific ticker
        
        Args:
            ticker: Stock symbol
        """
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self.cache_file = os.path.join(CACHE_DIR, f"{self.ticker}_financial_data.pkl")
        self.vector_file = os.path.join(VECTOR_DIR, f"{self.ticker}_vectors.pkl")
        self.data = self._load_cached_data()
        
        # Get sector and industry information for peer comparison
        try:
            info = self.stock.info
            self.sector = info.get('sector', '')
            self.industry = info.get('industry', '')
        except:
            self.sector = ''
            self.industry = ''
        
    def _load_cached_data(self) -> Dict:
        """Load data from cache if it exists and is not expired"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Check if cache is expired (using default TTL)
                last_updated = cached_data.get('last_updated')
                if last_updated and (datetime.now() - last_updated).days < CACHE_TTL['default']:
                    logger.info(f"Using cached data for {self.ticker}")
                    
                    # Check if any specific data types need refreshing
                    needs_refresh = False
                    
                    # Check financial statements (fundamentals)
                    if cached_data.get('financial_statements') and (
                        datetime.now() - last_updated).days >= CACHE_TTL['fundamentals']:
                        cached_data['financial_statements'] = {}
                        needs_refresh = True
                        logger.info(f"Financial statements cache expired for {self.ticker}, will refresh")
                    
                    # Check price-sensitive data
                    if cached_data.get('key_metrics') and (
                        datetime.now() - last_updated).days >= CACHE_TTL['price_data']:
                        cached_data['key_metrics'] = {}
                        needs_refresh = True
                        logger.info(f"Price data cache expired for {self.ticker}, will refresh")
                    
                    # Check filings
                    if cached_data.get('sec_filings') and (
                        datetime.now() - last_updated).days >= CACHE_TTL['filings']:
                        cached_data['sec_filings'] = {}
                        needs_refresh = True
                        logger.info(f"Filings cache expired for {self.ticker}, will refresh")
                    
                    # Check transcripts
                    if cached_data.get('earnings_call_transcripts') and (
                        datetime.now() - last_updated).days >= CACHE_TTL['transcripts']:
                        cached_data['earnings_call_transcripts'] = {}
                        needs_refresh = True
                        logger.info(f"Transcripts cache expired for {self.ticker}, will refresh")
                    
                    # If any data needs refreshing, update the last_updated timestamp
                    if needs_refresh:
                        cached_data['last_updated'] = datetime.now()
                    
                    return cached_data
                
                logger.info(f"Cached data for {self.ticker} is expired, fetching fresh data")
            except Exception as e:
                logger.error(f"Error loading cached data: {e}")
        
        # Initialize empty data structure
        return {
            'financial_statements': {},
            'earnings_info': {},
            'sec_filings': {},
            'earnings_call_transcripts': {},
            'key_metrics': {},
            'last_updated': None
        }
    
    def _save_cached_data(self):
        """Save data to cache"""
        try:
            self.data['last_updated'] = datetime.now()
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.data, f)
            logger.info(f"Saved data to cache for {self.ticker}")
        except Exception as e:
            logger.error(f"Error saving data to cache: {e}")
    
    def get_financial_statements(self) -> Dict:
        """Get enhanced income statement, balance sheet, and cash flow statement with additional data"""
        try:
            # Check if we already have this data cached
            if self.data['financial_statements'] and self.data['last_updated'] and \
               (datetime.now() - self.data['last_updated']).days < CACHE_TTL['fundamentals']:
                return self.data['financial_statements']
            
            # Get quarterly and annual statements
            income_stmt = {
                'quarterly': self.stock.quarterly_financials,
                'annual': self.stock.financials
            }
            
            balance_sheet = {
                'quarterly': self.stock.quarterly_balance_sheet,
                'annual': self.stock.balance_sheet
            }
            
            cash_flow = {
                'quarterly': self.stock.quarterly_cashflow,
                'annual': self.stock.cashflow
            }
            
            # Get additional data
            try:
                institutional_holders = self.stock.institutional_holders
            except Exception as e:
                logger.warning(f"Error getting institutional holders: {e}")
                institutional_holders = pd.DataFrame()
                
            try:
                major_holders = self.stock.major_holders
            except Exception as e:
                logger.warning(f"Error getting major holders: {e}")
                major_holders = pd.DataFrame()
                
            try:
                sustainability = self.stock.sustainability
            except Exception as e:
                logger.warning(f"Error getting sustainability data: {e}")
                sustainability = pd.DataFrame()
                
            try:
                recommendations = self.stock.recommendations
            except Exception as e:
                logger.warning(f"Error getting recommendations: {e}")
                recommendations = pd.DataFrame()
                
            try:
                # The earnings_trend attribute is not consistently available in yfinance
                # Let's create a custom earnings trend from the earnings history
                earnings_history = self.stock.earnings_history
                if isinstance(earnings_history, pd.DataFrame) and not earnings_history.empty:
                    # Create a custom earnings trend dataframe
                    earnings_trend = pd.DataFrame({
                        'Quarter': earnings_history.index,
                        'Actual EPS': earnings_history['epsActual'],
                        'Estimated EPS': earnings_history['epsEstimate'],
                        'Surprise': earnings_history['surprisePercent']
                    })
                else:
                    earnings_trend = pd.DataFrame()
            except Exception as e:
                logger.warning(f"Error creating earnings trend: {e}")
                earnings_trend = pd.DataFrame()
                
            try:
                shares_outstanding = self.stock.info.get('sharesOutstanding', 0)
                beta = self.stock.info.get('beta', 0)
                fifty_two_week_high = self.stock.info.get('fiftyTwoWeekHigh', 0)
                fifty_two_week_low = self.stock.info.get('fiftyTwoWeekLow', 0)
                
                additional_info = {
                    'shares_outstanding': shares_outstanding,
                    'beta': beta,
                    'fifty_two_week_high': fifty_two_week_high,
                    'fifty_two_week_low': fifty_two_week_low
                }
            except Exception as e:
                logger.warning(f"Error getting additional info: {e}")
                additional_info = {}
            
            # Store in data dictionary
            self.data['financial_statements'] = {
                'income_statement': income_stmt,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow,
                'institutional_holders': institutional_holders,
                'major_holders': major_holders,
                'sustainability': sustainability,
                'recommendations': recommendations,
                'earnings_trend': earnings_trend,
                'additional_info': additional_info
            }
            
            # Save to cache
            self._save_cached_data()
            
            return self.data['financial_statements']
        except Exception as e:
            logger.error(f"Error getting financial statements: {e}")
            return {}
    
    def get_earnings_info(self) -> Dict:
        """Get earnings dates, EPS estimates vs. actual"""
        try:
            # Check if we already have this data cached
            if self.data['earnings_info'] and self.data['last_updated'] and \
               (datetime.now() - self.data['last_updated']).days < CACHE_TTL['fundamentals']:
                return self.data['earnings_info']
            
            # Get earnings calendar and history
            earnings_calendar = self.stock.calendar
            earnings_history = self.stock.earnings_history
            
            # Get earnings dates
            earnings_dates = self.stock.earnings_dates
            
            # Store in data dictionary
            self.data['earnings_info'] = {
                'calendar': earnings_calendar,
                'history': earnings_history,
                'dates': earnings_dates
            }
            
            # Save to cache
            self._save_cached_data()
            
            return self.data['earnings_info']
        except Exception as e:
            logger.error(f"Error getting earnings info: {e}")
            return {}
    
    async def get_company_filings_async(self, filing_type: str = "10-Q", limit: int = 5) -> List[Dict]:
        """
        Asynchronous version: Get recent company filings using Alpha Vantage instead of SEC
        
        Args:
            filing_type: Type of filing (10-K, 10-Q, 8-K, etc.)
            limit: Maximum number of filings to retrieve
            
        Returns:
            List of dictionaries with filing information
        """
        try:
            # Check if we already have this data cached
            cache_key = f"{filing_type}_{limit}"
            if cache_key in self.data['sec_filings'] and self.data['last_updated'] and \
               (datetime.now() - self.data['last_updated']).days < CACHE_TTL['filings']:
                return self.data['sec_filings'][cache_key]
            
            # Get company overview from Alpha Vantage
            api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
            base_url = "https://www.alphavantage.co/query"
            
            # Create async session
            async with aiohttp.ClientSession() as session:
                # Get company overview
                overview_params = {
                    "function": "OVERVIEW",
                    "symbol": self.ticker,
                    "apikey": api_key
                }
                
                async with session.get(base_url, params=overview_params) as response:
                    overview_data = await response.json()
                
                # Create a synthetic filing structure similar to what we had with SEC
                filings = []
                if overview_data and not 'Error Message' in overview_data:
                    # Add a synthetic 10-K "filing" with company overview data
                    filings.append({
                        'form': '10-K',
                        'filing_date': overview_data.get('LatestQuarter', 'Unknown'),
                        'description': 'Annual Report',
                        'highlights': {
                            'Industry': overview_data.get('Industry', 'Unknown'),
                            'Sector': overview_data.get('Sector', 'Unknown'),
                            'Market Cap': overview_data.get('MarketCapitalization', 'Unknown'),
                            'PE Ratio': overview_data.get('PERatio', 'Unknown'),
                            'Dividend Yield': overview_data.get('DividendYield', 'Unknown'),
                            'EPS': overview_data.get('EPS', 'Unknown'),
                            'Revenue TTM': overview_data.get('RevenueTTM', 'Unknown'),
                            'Profit Margin': overview_data.get('ProfitMargin', 'Unknown')
                        }
                    })
                    
                    # Respect Alpha Vantage rate limits with asyncio.sleep instead of time.sleep
                    await asyncio.sleep(12)
                    
                    # Add quarterly earnings as "10-Q" filings
                    earnings_params = {
                        "function": "EARNINGS",
                        "symbol": self.ticker,
                        "apikey": api_key
                    }
                    
                    async with session.get(base_url, params=earnings_params) as response:
                        earnings_data = await response.json()
                    
                    if 'quarterlyEarnings' in earnings_data:
                        for i, quarter in enumerate(earnings_data['quarterlyEarnings'][:limit-1]):
                            filings.append({
                                'form': '10-Q',
                                'filing_date': quarter.get('fiscalDateEnding', 'Unknown'),
                                'description': 'Quarterly Report',
                                'highlights': {
                                    'Reported EPS': quarter.get('reportedEPS', 'Unknown'),
                                    'Estimated EPS': quarter.get('estimatedEPS', 'Unknown'),
                                    'Surprise': quarter.get('surprise', 'Unknown'),
                                    'Surprise Percentage': quarter.get('surprisePercentage', 'Unknown')
                                }
                            })
            
            # Store in data dictionary
            if 'sec_filings' not in self.data:
                self.data['sec_filings'] = {}
                
            self.data['sec_filings'][cache_key] = filings
            
            # Save to cache
            self._save_cached_data()
            
            return filings
        except Exception as e:
            logger.error(f"Error getting company filings asynchronously: {e}")
            return []
    
    def get_company_filings(self, filing_type: str = "10-Q", limit: int = 5) -> List[Dict]:
        """
        Get recent company filings using Alpha Vantage instead of SEC
        
        Args:
            filing_type: Type of filing (10-K, 10-Q, 8-K, etc.)
            limit: Maximum number of filings to retrieve
            
        Returns:
            List of dictionaries with filing information
        """
        try:
            # Check if we already have this data cached
            cache_key = f"{filing_type}_{limit}"
            if cache_key in self.data['sec_filings'] and self.data['last_updated'] and \
               (datetime.now() - self.data['last_updated']).days < CACHE_TTL['filings']:
                return self.data['sec_filings'][cache_key]
            
            # Try to use the async version with asyncio
            try:
                # Create an event loop or use the existing one
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Run the async function
                return loop.run_until_complete(self.get_company_filings_async(filing_type, limit))
            except Exception as async_error:
                logger.warning(f"Async call failed, falling back to synchronous: {async_error}")
                
                # Fallback to synchronous version
                # Get company overview from Alpha Vantage
                api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
                base_url = "https://www.alphavantage.co/query"
                
                # Get company overview
                overview_params = {
                    "function": "OVERVIEW",
                    "symbol": self.ticker,
                    "apikey": api_key
                }
                overview_response = requests.get(base_url, params=overview_params)
                overview_data = overview_response.json()
                
                # Create a synthetic filing structure similar to what we had with SEC
                filings = []
                if overview_data and not 'Error Message' in overview_data:
                    # Add a synthetic 10-K "filing" with company overview data
                    filings.append({
                        'form': '10-K',
                        'filing_date': overview_data.get('LatestQuarter', 'Unknown'),
                        'description': 'Annual Report',
                        'highlights': {
                            'Industry': overview_data.get('Industry', 'Unknown'),
                            'Sector': overview_data.get('Sector', 'Unknown'),
                            'Market Cap': overview_data.get('MarketCapitalization', 'Unknown'),
                            'PE Ratio': overview_data.get('PERatio', 'Unknown'),
                            'Dividend Yield': overview_data.get('DividendYield', 'Unknown'),
                            'EPS': overview_data.get('EPS', 'Unknown'),
                            'Revenue TTM': overview_data.get('RevenueTTM', 'Unknown'),
                            'Profit Margin': overview_data.get('ProfitMargin', 'Unknown')
                        }
                    })
                    
                    # Add quarterly earnings as "10-Q" filings
                    time.sleep(12)  # Respect Alpha Vantage rate limits
                    earnings_params = {
                        "function": "EARNINGS",
                        "symbol": self.ticker,
                        "apikey": api_key
                    }
                    earnings_response = requests.get(base_url, params=earnings_params)
                    earnings_data = earnings_response.json()
                    
                    if 'quarterlyEarnings' in earnings_data:
                        for i, quarter in enumerate(earnings_data['quarterlyEarnings'][:limit-1]):
                            filings.append({
                                'form': '10-Q',
                                'filing_date': quarter.get('fiscalDateEnding', 'Unknown'),
                                'description': 'Quarterly Report',
                                'highlights': {
                                    'Reported EPS': quarter.get('reportedEPS', 'Unknown'),
                                    'Estimated EPS': quarter.get('estimatedEPS', 'Unknown'),
                                    'Surprise': quarter.get('surprise', 'Unknown'),
                                    'Surprise Percentage': quarter.get('surprisePercentage', 'Unknown')
                                }
                            })
                
                # Store in data dictionary
                if 'sec_filings' not in self.data:
                    self.data['sec_filings'] = {}
                    
                self.data['sec_filings'][cache_key] = filings
                
                # Save to cache
                self._save_cached_data()
                
                return filings
        except Exception as e:
            logger.error(f"Error getting company filings: {e}")
            return []
    
    # Alias for backward compatibility
    get_sec_filings = get_company_filings
    
    def get_earnings_call_transcript(self, quarter: str = None) -> str:
        """
        Get earnings call transcript from Seeking Alpha (free method)
        
        Args:
            quarter: Quarter to get transcript for (e.g., "Q1 2023")
                    If None, gets the most recent transcript
                    
        Returns:
            Transcript text or empty string if not found
        """
        try:
            # Check if we already have this data cached
            cache_key = quarter if quarter else "latest"
            if cache_key in self.data['earnings_call_transcripts'] and self.data['last_updated'] and \
               (datetime.now() - self.data['last_updated']).days < CACHE_TTL['transcripts']:
                return self.data['earnings_call_transcripts'][cache_key]
            
            # Try to get transcript from Seeking Alpha
            try:
                # Use a more browser-like User-Agent to avoid being blocked
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Referer': 'https://seekingalpha.com/',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Cache-Control': 'max-age=0'
                }
                
                # Search for the transcript on Seeking Alpha
                search_url = f"https://seekingalpha.com/symbol/{self.ticker}/earnings/transcripts"
                
                response = requests.get(search_url, headers=headers)
                if response.status_code != 200:
                    logger.warning(f"Error searching for transcript: {response.status_code}, trying fallback method")
                    raise Exception(f"HTTP error: {response.status_code}")
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find transcript links
                transcript_links = []
                for a in soup.find_all('a', href=True):
                    if '/article/' in a['href'] and 'earnings-call-transcript' in a['href']:
                        transcript_links.append(a['href'])
                
                if not transcript_links:
                    logger.warning(f"No transcript links found for {self.ticker}, trying fallback method")
                    raise Exception("No transcript links found")
                
                # Get the most recent transcript or the one matching the specified quarter
                transcript_url = f"https://seekingalpha.com{transcript_links[0]}"
                
                # Get the transcript
                time.sleep(1.5)  # Longer delay to respect rate limits
                response = requests.get(transcript_url, headers=headers)
                if response.status_code != 200:
                    logger.warning(f"Error getting transcript: {response.status_code}, trying fallback method")
                    raise Exception(f"HTTP error: {response.status_code}")
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract the transcript text
                transcript_text = ""
                article_body = soup.find('div', {'data-test-id': 'content-container'})
                
                if article_body:
                    paragraphs = article_body.find_all('p')
                    for p in paragraphs:
                        transcript_text += p.get_text() + "\n\n"
                
                if not transcript_text:
                    logger.warning(f"No transcript text found for {self.ticker}, trying fallback method")
                    raise Exception("No transcript text found")
                
            except Exception as e:
                logger.warning(f"Seeking Alpha transcript retrieval failed: {e}, using fallback method")
                
                # Fallback method: Generate a synthetic transcript from earnings data
                transcript_text = self._generate_synthetic_transcript()
            
            # Store in data dictionary
            if 'earnings_call_transcripts' not in self.data:
                self.data['earnings_call_transcripts'] = {}
                
            self.data['earnings_call_transcripts'][cache_key] = transcript_text
            
            # Save to cache
            self._save_cached_data()
            
            return transcript_text
        except Exception as e:
            logger.error(f"Error getting earnings call transcript: {e}")
            return ""
    
    def _generate_synthetic_transcript(self) -> str:
        """Generate a synthetic transcript from earnings data when real transcript is unavailable"""
        try:
            # Get earnings info and financial statements
            earnings_info = self.get_earnings_info()
            financial_statements = self.get_financial_statements()
            key_metrics = self.calculate_key_metrics()
            
            # Create a synthetic transcript
            transcript = f"Earnings Call Transcript for {self.ticker} (Synthetic)\n\n"
            transcript += f"Note: This is an AI-generated summary based on available financial data, not an actual transcript.\n\n"
            
            # Add company information
            transcript += f"Company: {key_metrics.get('company_name', self.ticker)}\n"
            transcript += f"Sector: {key_metrics.get('sector', 'Unknown')}\n"
            transcript += f"Industry: {key_metrics.get('industry', 'Unknown')}\n\n"
            
            # Add earnings information
            transcript += "EARNINGS SUMMARY:\n\n"
            
            # Get the most recent earnings
            earnings_history = earnings_info.get('history', {})
            if earnings_history:
                latest_date = list(earnings_history.keys())[0] if earnings_history else "Unknown"
                latest_earnings = earnings_history.get(latest_date, {})
                
                transcript += f"Date: {latest_date}\n"
                transcript += f"EPS Actual: ${latest_earnings.get('epsActual', 'N/A')}\n"
                transcript += f"EPS Estimate: ${latest_earnings.get('epsEstimate', 'N/A')}\n"
                
                # Calculate surprise percentage
                if 'epsActual' in latest_earnings and 'epsEstimate' in latest_earnings:
                    actual = float(latest_earnings['epsActual'])
                    estimate = float(latest_earnings['epsEstimate'])
                    if estimate != 0:
                        surprise_pct = ((actual - estimate) / abs(estimate)) * 100
                        transcript += f"Surprise: {surprise_pct:.2f}%\n"
            
            # Add financial highlights
            transcript += "\nFINANCIAL HIGHLIGHTS:\n\n"
            
            # Get annual and quarterly data
            annual_income = financial_statements.get('income_statement', {}).get('annual', pd.DataFrame())
            quarterly_income = financial_statements.get('income_statement', {}).get('quarterly', pd.DataFrame())
            
            if isinstance(annual_income, pd.DataFrame) and not annual_income.empty:
                try:
                    latest_annual = annual_income.iloc[:, 0]  # Most recent column
                    transcript += "Annual Results:\n"
                    
                    # Safely get values with fallbacks
                    try:
                        revenue_val = latest_annual.get('Total Revenue')
                        if revenue_val is not None and pd.notna(revenue_val):
                            revenue = float(revenue_val)
                            transcript += f"Revenue: ${revenue:,.2f}\n"
                        else:
                            transcript += "Revenue: Not available\n"
                    except:
                        transcript += "Revenue: Not available\n"
                        
                    try:
                        net_income_val = latest_annual.get('Net Income')
                        if net_income_val is not None and pd.notna(net_income_val):
                            net_income = float(net_income_val)
                            transcript += f"Net Income: ${net_income:,.2f}\n"
                        else:
                            transcript += "Net Income: Not available\n"
                    except:
                        transcript += "Net Income: Not available\n"
                        
                    try:
                        eps_val = latest_annual.get('Basic EPS')
                        if eps_val is not None and pd.notna(eps_val):
                            eps = float(eps_val)
                            transcript += f"EPS: ${eps:,.2f}\n\n"
                        else:
                            transcript += "EPS: Not available\n\n"
                    except:
                        transcript += "EPS: Not available\n\n"
                except Exception as e:
                    transcript += f"Annual Results: Error processing data - {str(e)}\n\n"
            
            if isinstance(quarterly_income, pd.DataFrame) and not quarterly_income.empty:
                try:
                    latest_quarterly = quarterly_income.iloc[:, 0]  # Most recent column
                    transcript += "Quarterly Results:\n"
                    
                    # Safely get values with fallbacks
                    try:
                        revenue = float(latest_quarterly.get('Total Revenue', 0))
                        transcript += f"Revenue: ${revenue:,.2f}\n"
                    except:
                        transcript += "Revenue: Not available\n"
                        
                    try:
                        net_income = float(latest_quarterly.get('Net Income', 0))
                        transcript += f"Net Income: ${net_income:,.2f}\n"
                    except:
                        transcript += "Net Income: Not available\n"
                        
                    try:
                        eps = float(latest_quarterly.get('Basic EPS', 0))
                        transcript += f"EPS: ${eps:,.2f}\n\n"
                    except:
                        transcript += "EPS: Not available\n\n"
                except Exception as e:
                    transcript += f"Quarterly Results: Error processing data - {str(e)}\n\n"
            
            # Add key metrics
            transcript += "KEY METRICS:\n\n"
            for key, value in key_metrics.items():
                if key not in ['company_name', 'sector', 'industry'] and value:
                    # Format the key name for better readability
                    formatted_key = key.replace('_', ' ').title()
                    
                    # Format the value based on its type
                    if isinstance(value, float):
                        if 'ratio' in key or 'margin' in key or 'growth' in key or 'yield' in key:
                            formatted_value = f"{value:.2%}" if value < 1 else f"{value:.2f}%"
                        elif 'price' in key or 'eps' in key or 'dividend' in key:
                            formatted_value = f"${value:.2f}"
                        else:
                            formatted_value = f"{value:,.2f}"
                    else:
                        formatted_value = str(value)
                    
                    transcript += f"{formatted_key}: {formatted_value}\n"
            
            # Add recommendations if available
            recommendations = financial_statements.get('recommendations', pd.DataFrame())
            if not recommendations.empty:
                transcript += "\nANALYST RECOMMENDATIONS:\n\n"
                recent_recs = recommendations.iloc[:5]  # Get 5 most recent
                for i, (date, grade, firm) in enumerate(zip(recent_recs.index, recent_recs['To Grade'], recent_recs['Firm'])):
                    transcript += f"{date.strftime('%Y-%m-%d')}: {firm} - {grade}\n"
            
            return transcript
        except Exception as e:
            logger.error(f"Error generating synthetic transcript: {e}")
            return f"Unable to generate transcript for {self.ticker}. Error: {str(e)}"
    
    def calculate_key_metrics(self) -> Dict:
        """Calculate key financial metrics and ratios"""
        try:
            # Check if we already have this data cached
            if self.data['key_metrics'] and self.data['last_updated'] and \
               (datetime.now() - self.data['last_updated']).days < CACHE_TTL['metrics']:
                return self.data['key_metrics']
            
            # Get financial statements
            self.get_financial_statements()
            
            # Get stock info
            info = self.stock.info
            
            # Initialize metrics dictionary
            metrics = {}
            
            # Add basic info
            metrics['company_name'] = info.get('longName', self.ticker)
            metrics['sector'] = info.get('sector', 'Unknown')
            metrics['industry'] = info.get('industry', 'Unknown')
            metrics['market_cap'] = info.get('marketCap', 0)
            
            # Add valuation metrics
            metrics['pe_ratio'] = info.get('trailingPE', 0)
            metrics['forward_pe'] = info.get('forwardPE', 0)
            metrics['peg_ratio'] = info.get('pegRatio', 0)
            metrics['price_to_book'] = info.get('priceToBook', 0)
            metrics['price_to_sales'] = info.get('priceToSalesTrailing12Months', 0)
            
            # Add profitability metrics
            metrics['profit_margin'] = info.get('profitMargins', 0)
            metrics['operating_margin'] = info.get('operatingMargins', 0)
            metrics['roa'] = info.get('returnOnAssets', 0)
            metrics['roe'] = info.get('returnOnEquity', 0)
            
            # Add growth metrics
            metrics['revenue_growth'] = info.get('revenueGrowth', 0)
            metrics['earnings_growth'] = info.get('earningsGrowth', 0)
            
            # Add dividend metrics
            metrics['dividend_yield'] = info.get('dividendYield', 0)
            metrics['dividend_rate'] = info.get('dividendRate', 0)
            metrics['payout_ratio'] = info.get('payoutRatio', 0)
            
            # Add debt metrics
            metrics['debt_to_equity'] = info.get('debtToEquity', 0)
            metrics['current_ratio'] = info.get('currentRatio', 0)
            
            # Store in data dictionary
            self.data['key_metrics'] = metrics
            
            # Save to cache
            self._save_cached_data()
            
            return metrics
        except Exception as e:
            logger.error(f"Error calculating key metrics: {e}")
            return {}
    
    def get_comprehensive_analysis(self) -> Dict:
        """Get comprehensive financial analysis with enhanced data"""
        # Get all data
        financial_statements = self.get_financial_statements()
        earnings_info = self.get_earnings_info()
        company_filings = self.get_company_filings()
        key_metrics = self.calculate_key_metrics()
        
        # Try to get earnings call transcript
        transcript = self.get_earnings_call_transcript()
        
        # Combine all data
        analysis = {
            'ticker': self.ticker,
            'company_name': key_metrics.get('company_name', self.ticker),
            'sector': key_metrics.get('sector', 'Unknown'),
            'industry': key_metrics.get('industry', 'Unknown'),
            'key_metrics': key_metrics,
            'recent_earnings': earnings_info.get('history', {}),
            'next_earnings_date': earnings_info.get('calendar', {}).get('Earnings Date', 'Unknown'),
            'recent_filings': company_filings,
            'has_transcript': bool(transcript),
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        # Add financial statement summaries
        if financial_statements:
            # Get the most recent annual income statement
            annual_income = financial_statements.get('income_statement', {}).get('annual', pd.DataFrame())
            if not annual_income.empty:
                latest_annual = annual_income.iloc[:, 0]  # Most recent column
                analysis['annual_revenue'] = float(latest_annual.get('Total Revenue', 0))
                analysis['annual_net_income'] = float(latest_annual.get('Net Income', 0))
                analysis['annual_eps'] = float(latest_annual.get('Basic EPS', 0))
            
            # Get the most recent quarterly income statement
            quarterly_income = financial_statements.get('income_statement', {}).get('quarterly', pd.DataFrame())
            if not quarterly_income.empty:
                latest_quarterly = quarterly_income.iloc[:, 0]  # Most recent column
                analysis['quarterly_revenue'] = float(latest_quarterly.get('Total Revenue', 0))
                analysis['quarterly_net_income'] = float(latest_quarterly.get('Net Income', 0))
                analysis['quarterly_eps'] = float(latest_quarterly.get('Basic EPS', 0))
            
            # Add additional data from enhanced financial statements
            analysis['institutional_holders'] = financial_statements.get('institutional_holders', pd.DataFrame()).to_dict() if isinstance(financial_statements.get('institutional_holders'), pd.DataFrame) else {}
            analysis['major_holders'] = financial_statements.get('major_holders', pd.DataFrame()).to_dict() if isinstance(financial_statements.get('major_holders'), pd.DataFrame) else {}
            analysis['sustainability'] = financial_statements.get('sustainability', pd.DataFrame()).to_dict() if isinstance(financial_statements.get('sustainability'), pd.DataFrame) else {}
            analysis['recommendations'] = financial_statements.get('recommendations', pd.DataFrame()).to_dict() if isinstance(financial_statements.get('recommendations'), pd.DataFrame) else {}
            analysis['earnings_trend'] = financial_statements.get('earnings_trend', pd.DataFrame()).to_dict() if isinstance(financial_statements.get('earnings_trend'), pd.DataFrame) else {}
            analysis['additional_info'] = financial_statements.get('additional_info', {})
        
        # Add Alpha Vantage data if available
        if company_filings and len(company_filings) > 0:
            # Extract highlights from the first filing (usually the 10-K with company overview)
            if 'highlights' in company_filings[0]:
                analysis['alpha_vantage_highlights'] = company_filings[0].get('highlights', {})
        
        return analysis


# Function to get comprehensive financial analysis
def get_financial_analysis(ticker: str) -> Dict:
    """
    Get comprehensive financial analysis for a stock
    
    Args:
        ticker: Stock symbol
        
    Returns:
        Dictionary with comprehensive financial analysis
    """
    analyzer = FinancialAnalyzer(ticker)
    return analyzer.get_comprehensive_analysis()


# Simple vector database implementation using numpy and pickle
class SimpleVectorDB:
    """Simple vector database implementation using numpy and pickle"""
    
    def __init__(self, name: str):
        """
        Initialize the vector database
        
        Args:
            name: Name of the database
        """
        self.name = name
        self.db_file = os.path.join(VECTOR_DIR, f"{name}_vector_db.pkl")
        self.vectors = []
        self.texts = []
        self.metadata = []
        self._load_db()
    
    def _load_db(self):
        """Load the database from disk"""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'rb') as f:
                    data = pickle.load(f)
                    self.vectors = data.get('vectors', [])
                    self.texts = data.get('texts', [])
                    self.metadata = data.get('metadata', [])
                logger.info(f"Loaded vector database {self.name} with {len(self.vectors)} entries")
            except Exception as e:
                logger.error(f"Error loading vector database: {e}")
    
    def _save_db(self):
        """Save the database to disk"""
        try:
            with open(self.db_file, 'wb') as f:
                pickle.dump({
                    'vectors': self.vectors,
                    'texts': self.texts,
                    'metadata': self.metadata
                }, f)
            logger.info(f"Saved vector database {self.name} with {len(self.vectors)} entries")
        except Exception as e:
            logger.error(f"Error saving vector database: {e}")
    
    def add_text(self, text: str, vector: List[float], metadata: Dict = None):
        """
        Add text and its vector to the database
        
        Args:
            text: Text to add
            vector: Vector representation of the text
            metadata: Additional metadata
        """
        self.texts.append(text)
        self.vectors.append(vector)
        self.metadata.append(metadata or {})
        self._save_db()
    
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Vector to search for
            top_k: Number of results to return
            
        Returns:
            List of dictionaries with text, similarity score, and metadata
        """
        if not self.vectors:
            return []
        
        # Convert to numpy arrays
        query_vector = np.array(query_vector)
        vectors = np.array(self.vectors)
        
        # Calculate cosine similarity
        similarities = np.dot(vectors, query_vector) / (
            np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector)
        )
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return results
        results = []
        for i in top_indices:
            results.append({
                'text': self.texts[i],
                'similarity': float(similarities[i]),
                'metadata': self.metadata[i]
            })
        
        return results


# Function to generate embeddings using sentence-transformers (free alternative to OpenAI)
def generate_embeddings(text: str) -> List[float]:
    """
    Generate embeddings for text using sentence-transformers
    
    Args:
        text: Text to generate embeddings for
        
    Returns:
        List of floats representing the embedding
    """
    try:
        # Try to import sentence-transformers
        from sentence_transformers import SentenceTransformer
        
        # Load model (will download if not already present)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate embedding
        embedding = model.encode(text)
        
        return embedding.tolist()
    except ImportError:
        logger.error("sentence-transformers not installed. Please install with: pip install sentence-transformers")
        # Return a random embedding as fallback (not recommended for production)
        return np.random.rand(384).tolist()  # 384 is the dimension of all-MiniLM-L6-v2


# Function to store financial analysis in vector database
def store_financial_analysis_in_vector_db(ticker: str) -> bool:
    """
    Store financial analysis in vector database
    
    Args:
        ticker: Stock symbol
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get financial analysis
        analysis = get_financial_analysis(ticker)
        
        # Create vector database
        db = SimpleVectorDB(f"{ticker}_financial")
        
        # Store key metrics
        metrics_text = f"Key metrics for {ticker} ({analysis.get('company_name', ticker)}):\n"
        for key, value in analysis.get('key_metrics', {}).items():
            metrics_text += f"{key}: {value}\n"
        
        metrics_vector = generate_embeddings(metrics_text)
        db.add_text(metrics_text, metrics_vector, {
            'ticker': ticker,
            'type': 'key_metrics',
            'date': analysis.get('analysis_date')
        })
        
        # Store earnings info
        if 'recent_earnings' in analysis:
            earnings_text = f"Recent earnings for {ticker} ({analysis.get('company_name', ticker)}):\n"
            earnings_text += f"Next earnings date: {analysis.get('next_earnings_date', 'Unknown')}\n\n"
            
            # Add recent earnings history
            earnings_text += "Earnings history:\n"
            for date, data in analysis.get('recent_earnings', {}).items():
                earnings_text += f"{date}: EPS {data.get('epsActual', 'N/A')} vs. estimate {data.get('epsEstimate', 'N/A')}\n"
            
            earnings_vector = generate_embeddings(earnings_text)
            db.add_text(earnings_text, earnings_vector, {
                'ticker': ticker,
                'type': 'earnings_info',
                'date': analysis.get('analysis_date')
            })
        
        # Store financial statement summaries
        statement_text = f"Financial statement summary for {ticker} ({analysis.get('company_name', ticker)}):\n\n"
        
        # Annual data
        statement_text += "Annual data:\n"
        statement_text += f"Revenue: {analysis.get('annual_revenue', 'N/A')}\n"
        statement_text += f"Net Income: {analysis.get('annual_net_income', 'N/A')}\n"
        statement_text += f"EPS: {analysis.get('annual_eps', 'N/A')}\n\n"
        
        # Quarterly data
        statement_text += "Quarterly data:\n"
        statement_text += f"Revenue: {analysis.get('quarterly_revenue', 'N/A')}\n"
        statement_text += f"Net Income: {analysis.get('quarterly_net_income', 'N/A')}\n"
        statement_text += f"EPS: {analysis.get('quarterly_eps', 'N/A')}\n"
        
        statement_vector = generate_embeddings(statement_text)
        db.add_text(statement_text, statement_vector, {
            'ticker': ticker,
            'type': 'financial_statements',
            'date': analysis.get('analysis_date')
        })
        
        # Store SEC filings
        if 'recent_filings' in analysis:
            filings_text = f"Recent SEC filings for {ticker} ({analysis.get('company_name', ticker)}):\n\n"
            
            for filing in analysis.get('recent_filings', []):
                filings_text += f"Form {filing.get('form', 'N/A')} filed on {filing.get('filing_date', 'N/A')}\n"
                filings_text += f"URL: {filing.get('url', 'N/A')}\n\n"
            
            filings_vector = generate_embeddings(filings_text)
            db.add_text(filings_text, filings_vector, {
                'ticker': ticker,
                'type': 'sec_filings',
                'date': analysis.get('analysis_date')
            })
        
        # Store earnings call transcript if available
        if analysis.get('has_transcript'):
            analyzer = FinancialAnalyzer(ticker)
            transcript = analyzer.get_earnings_call_transcript()
            
            # Split transcript into chunks to avoid token limits
            chunk_size = 1000  # characters
            chunks = [transcript[i:i+chunk_size] for i in range(0, len(transcript), chunk_size)]
            
            for i, chunk in enumerate(chunks):
                chunk_vector = generate_embeddings(chunk)
                db.add_text(chunk, chunk_vector, {
                    'ticker': ticker,
                    'type': 'earnings_transcript',
                    'chunk': i,
                    'total_chunks': len(chunks),
                    'date': analysis.get('analysis_date')
                })
        
        return True
    except Exception as e:
        logger.error(f"Error storing financial analysis in vector DB: {e}")
        return False


# Function to search financial analysis in vector database
def search_financial_analysis(ticker: str, query: str, top_k: int = 5) -> List[Dict]:
    """
    Search financial analysis in vector database
    
    Args:
        ticker: Stock symbol
        query: Query text
        top_k: Number of results to return
        
    Returns:
        List of dictionaries with text, similarity score, and metadata
    """
    try:
        # Generate query embedding
        query_vector = generate_embeddings(query)
        
        # Load vector database
        db = SimpleVectorDB(f"{ticker}_financial")
        
        # Search
        results = db.search(query_vector, top_k)
        
        return results
    except Exception as e:
        logger.error(f"Error searching financial analysis: {e}")
        return []


# Function to get fallback data when APIs fail
def get_fallback_financial_data(ticker: str) -> Dict:
    """
    Get fallback financial data when APIs fail
    
    Args:
        ticker: Stock symbol
        
    Returns:
        Dictionary with basic financial data
    """
    try:
        # Try to get minimal data from yfinance as a last resort
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Create a minimal analysis with whatever data we can get
        analysis = {
            'ticker': ticker,
            'company_name': info.get('longName', ticker),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'eps': info.get('trailingEps', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'beta': info.get('beta', 0),
            'fallback_data': True,
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        # Try to get price history
        try:
            hist = stock.history(period="1mo")
            if not hist.empty:
                analysis['price_history'] = {
                    'current': hist['Close'].iloc[-1] if not hist.empty else 0,
                    'change_1d': ((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100 if len(hist) > 1 else 0,
                    'change_1w': ((hist['Close'].iloc[-1] / hist['Close'].iloc[-5]) - 1) * 100 if len(hist) >= 5 else 0,
                    'change_1m': ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100 if len(hist) > 0 else 0,
                    'volume': hist['Volume'].iloc[-1] if not hist.empty else 0
                }
        except Exception as e:
            logger.warning(f"Error getting price history for fallback: {e}")
            analysis['price_history'] = {}
        
        return analysis
    except Exception as e:
        logger.error(f"Error getting fallback data: {e}")
        return {
            'ticker': ticker,
            'error': f"Unable to retrieve any financial data: {str(e)}",
            'fallback_data': True
        }

# Function to get comprehensive financial insights
def get_industry_comparison(ticker: str) -> Dict:
    """
    Get industry comparison metrics for a stock
    
    Args:
        ticker: Stock symbol
        
    Returns:
        Dictionary with industry comparison metrics
    """
    try:
        # Get the analyzer for the ticker
        analyzer = FinancialAnalyzer(ticker)
        
        # Get the sector and industry
        sector = analyzer.sector
        industry = analyzer.industry
        
        if not sector or not industry:
            return {
                "error": "Unable to determine sector or industry",
                "ticker": ticker
            }
        
        # Get the company's key metrics
        company_metrics = analyzer.calculate_key_metrics()
        
        # Find peer companies in the same industry
        # We'll use a simple approach: search for companies in the same sector/industry
        # using predefined lists of major companies by sector
        
        # Define major companies by sector
        sector_companies = {
            "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "INTC", "CSCO", "ORCL", "IBM", "ADBE", "CRM"],
            "Financial Services": ["JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "V", "MA", "BLK", "SCHW"],
            "Healthcare": ["JNJ", "PFE", "MRK", "ABBV", "ABT", "UNH", "CVS", "AMGN", "GILD", "BMY", "LLY"],
            "Consumer Cyclical": ["AMZN", "HD", "NKE", "SBUX", "MCD", "LOW", "TGT", "BKNG", "MAR", "F", "GM"],
            "Consumer Defensive": ["WMT", "PG", "KO", "PEP", "COST", "PM", "MO", "EL", "CL", "GIS", "K"],
            "Industrials": ["BA", "HON", "UNP", "UPS", "CAT", "DE", "LMT", "RTX", "GE", "MMM", "EMR"],
            "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "OXY", "MPC", "KMI"],
            "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED", "WEC"],
            "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "PSA", "O", "AVB", "EQR", "DLR", "SPG"],
            "Basic Materials": ["LIN", "APD", "ECL", "SHW", "NEM", "FCX", "NUE", "DOW", "DD", "VMC"],
            "Communication Services": ["GOOGL", "META", "VZ", "T", "CMCSA", "NFLX", "DIS", "TMUS", "EA", "ATVI"]
        }
        
        # Get peer companies
        peer_companies = []
        if sector in sector_companies:
            peer_companies = [company for company in sector_companies[sector] if company != ticker]
        
        # Limit to 5 peers for efficiency
        peer_companies = peer_companies[:5]
        
        if not peer_companies:
            return {
                "error": f"No peer companies found for {ticker} in sector {sector}",
                "ticker": ticker,
                "sector": sector,
                "industry": industry
            }
        
        # Get metrics for peer companies
        peer_metrics = {}
        for peer in peer_companies:
            try:
                peer_analyzer = FinancialAnalyzer(peer)
                peer_metrics[peer] = peer_analyzer.calculate_key_metrics()
            except Exception as e:
                logger.warning(f"Error getting metrics for peer {peer}: {e}")
        
        # Calculate industry averages
        industry_averages = {}
        for metric in ['pe_ratio', 'forward_pe', 'price_to_book', 'price_to_sales', 
                      'profit_margin', 'operating_margin', 'roa', 'roe', 
                      'revenue_growth', 'earnings_growth', 'dividend_yield', 
                      'debt_to_equity', 'current_ratio']:
            # Get values for this metric from all peers
            values = [metrics.get(metric, 0) for peer, metrics in peer_metrics.items() if metrics.get(metric, 0) > 0]
            
            # Calculate average if we have values
            if values:
                industry_averages[metric] = sum(values) / len(values)
            else:
                industry_averages[metric] = 0
        
        # Calculate relative performance (company vs industry)
        relative_performance = {}
        for metric in industry_averages:
            company_value = company_metrics.get(metric)
            industry_value = industry_averages.get(metric)
            
            # Skip if either value is None or zero for division
            if company_value is None or industry_value is None or industry_value == 0:
                continue
                
            # Ensure both values are numeric
            try:
                company_value = float(company_value)
                industry_value = float(industry_value)
                
                if industry_value > 0 and company_value > 0:
                    # For metrics where higher is better
                    if metric in ['roa', 'roe', 'revenue_growth', 'earnings_growth', 'current_ratio']:
                        relative_performance[metric] = (company_value / industry_value) - 1
                    
                    # For metrics where lower is better
                    elif metric in ['pe_ratio', 'forward_pe', 'price_to_book', 'price_to_sales', 'debt_to_equity']:
                        relative_performance[metric] = 1 - (company_value / industry_value)
                    
                    # For other metrics, just show the difference
                    else:
                        relative_performance[metric] = company_value - industry_value
            except (TypeError, ValueError):
                # Skip metrics that can't be converted to float
                continue
        
        # Get sector ETF performance if available
        sector_etfs = {
            "Technology": "XLK",
            "Financial Services": "XLF",
            "Healthcare": "XLV",
            "Consumer Cyclical": "XLY",
            "Consumer Defensive": "XLP",
            "Industrials": "XLI",
            "Energy": "XLE",
            "Utilities": "XLU",
            "Real Estate": "XLRE",
            "Basic Materials": "XLB",
            "Communication Services": "XLC"
        }
        
        sector_etf_performance = {}
        if sector in sector_etfs:
            try:
                etf_ticker = sector_etfs[sector]
                etf = yf.Ticker(etf_ticker)
                hist = etf.history(period="1y")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    price_1m_ago = hist['Close'].iloc[-22] if len(hist) >= 22 else hist['Close'].iloc[0]
                    price_3m_ago = hist['Close'].iloc[-66] if len(hist) >= 66 else hist['Close'].iloc[0]
                    price_6m_ago = hist['Close'].iloc[-126] if len(hist) >= 126 else hist['Close'].iloc[0]
                    price_1y_ago = hist['Close'].iloc[0]
                    
                    sector_etf_performance = {
                        'etf_ticker': etf_ticker,
                        'current_price': current_price,
                        'change_1m': ((current_price / price_1m_ago) - 1) * 100,
                        'change_3m': ((current_price / price_3m_ago) - 1) * 100,
                        'change_6m': ((current_price / price_6m_ago) - 1) * 100,
                        'change_1y': ((current_price / price_1y_ago) - 1) * 100
                    }
            except Exception as e:
                logger.warning(f"Error getting sector ETF performance: {e}")
        
        # Compile the industry comparison data
        industry_comparison = {
            'ticker': ticker,
            'sector': sector,
            'industry': industry,
            'peer_companies': list(peer_metrics.keys()),
            'company_metrics': company_metrics,
            'industry_averages': industry_averages,
            'relative_performance': relative_performance,
            'sector_etf_performance': sector_etf_performance,
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        return industry_comparison
    except Exception as e:
        logger.error(f"Error getting industry comparison: {e}")
        return {
            "error": str(e),
            "ticker": ticker
        }

def get_financial_insights(ticker: str, query: str = None) -> Dict:
    """
    Get comprehensive financial insights for a stock
    
    Args:
        ticker: Stock symbol
        query: Optional query to search for specific information
        
    Returns:
        Dictionary with comprehensive financial insights
    """
    try:
        # Check if vector database exists for this ticker
        db = SimpleVectorDB(f"{ticker}_financial")
        vector_db_exists = len(db.vectors) > 0
        
        # Check if this is an earnings call related query
        earnings_keywords = ['earnings call', 'earnings', 'call', 'transcript', 'ceo', 'management', 'guidance', 'outlook', 'recent earning']
        is_earnings_query = query and any(keyword in query.lower() for keyword in earnings_keywords)
        
        # If no vector database exists, create one
        if not vector_db_exists:
            logger.info(f"No vector database found for {ticker}, creating one...")
            try:
                # Get financial analysis first
                analysis = get_financial_analysis(ticker)
                
                # Store in vector database
                if not analysis.get('fallback_data', False):
                    store_financial_analysis_in_vector_db(ticker)
                    # Reload the database after storing
                    db = SimpleVectorDB(f"{ticker}_financial")
                    vector_db_exists = len(db.vectors) > 0
                    logger.info(f"Created vector database for {ticker} with {len(db.vectors)} entries")
            except Exception as creation_error:
                logger.error(f"Failed to create vector database for {ticker}: {creation_error}")
                # Fall back to regular analysis
                try:
                    analysis = get_financial_analysis(ticker)
                except Exception as analysis_error:
                    logger.error(f"Primary analysis failed: {analysis_error}, trying fallback")
                    analysis = get_fallback_financial_data(ticker)
        else:
            # Get basic analysis
            try:
                analysis = get_financial_analysis(ticker)
            except Exception as analysis_error:
                logger.error(f"Primary analysis failed: {analysis_error}, trying fallback")
                analysis = get_fallback_financial_data(ticker)
        
        # If this is an earnings query and we have a vector database, check if we have transcript data
        if is_earnings_query and vector_db_exists:
            # Check if we have any earnings transcript data in the vector database
            transcript_entries = [entry for entry in db.metadata if entry.get('type') == 'earnings_transcript']
            
            if not transcript_entries:
                logger.info(f"No earnings transcript found in vector database for {ticker}, attempting to fetch and store...")
                try:
                    # Try to get earnings call transcript directly
                    analyzer = FinancialAnalyzer(ticker)
                    transcript = analyzer.get_earnings_call_transcript()
                    
                    if transcript and len(transcript.strip()) > 100:  # Make sure we got substantial content
                        logger.info(f"Successfully retrieved earnings transcript for {ticker}, storing in vector database...")
                        
                        # Split transcript into chunks and store in vector database
                        chunk_size = 1000  # characters
                        chunks = [transcript[i:i+chunk_size] for i in range(0, len(transcript), chunk_size)]
                        
                        for i, chunk in enumerate(chunks):
                            chunk_vector = generate_embeddings(chunk)
                            db.add_text(chunk, chunk_vector, {
                                'ticker': ticker,
                                'type': 'earnings_transcript',
                                'chunk': i,
                                'total_chunks': len(chunks),
                                'date': datetime.now().strftime('%Y-%m-%d'),
                                'source': 'auto_retrieved'
                            })
                        
                        logger.info(f"Stored {len(chunks)} transcript chunks for {ticker} in vector database")
                        
                        # Reload the database to include new transcript data
                        db = SimpleVectorDB(f"{ticker}_financial")
                    else:
                        logger.warning(f"No substantial earnings transcript content found for {ticker}")
                        
                except Exception as transcript_error:
                    logger.error(f"Failed to retrieve and store earnings transcript for {ticker}: {transcript_error}")
        
        # If query is provided, search for specific information in vector database
        if query and vector_db_exists:
            try:
                # Check if this is an earnings call related query
                earnings_keywords = ['earnings call', 'earnings', 'call', 'transcript', 'ceo', 'management', 'guidance', 'outlook', 'recent earning']
                is_earnings_query = any(keyword in query.lower() for keyword in earnings_keywords)
                
                # Search the vector database
                search_results = search_financial_analysis(ticker, query, top_k=10)
                
                # Filter and prioritize earnings call transcript results if it's an earnings query
                if is_earnings_query and search_results:
                    # Prioritize earnings transcript results
                    transcript_results = [r for r in search_results if r.get('metadata', {}).get('type') == 'earnings_transcript']
                    other_results = [r for r in search_results if r.get('metadata', {}).get('type') != 'earnings_transcript']
                    
                    # Combine with transcript results first
                    prioritized_results = transcript_results + other_results
                    search_results = prioritized_results[:5]  # Keep top 5
                
                # Add search results to analysis
                analysis['search_results'] = search_results
                analysis['query_processed'] = query
                analysis['is_earnings_query'] = is_earnings_query
                
                # If we found earnings transcript content, extract and format it
                if is_earnings_query and search_results:
                    transcript_content = []
                    for result in search_results:
                        if result.get('metadata', {}).get('type') == 'earnings_transcript':
                            transcript_content.append({
                                'text': result['text'],
                                'similarity': result['similarity'],
                                'chunk': result.get('metadata', {}).get('chunk', 0)
                            })
                    
                    if transcript_content:
                        # Sort by chunk order to maintain narrative flow
                        transcript_content.sort(key=lambda x: x.get('chunk', 0))
                        
                        # Combine transcript chunks
                        combined_transcript = "\n\n".join([chunk['text'] for chunk in transcript_content])
                        
                        analysis['earnings_call_content'] = {
                            'transcript_found': True,
                            'content': combined_transcript,
                            'chunks_found': len(transcript_content),
                            'avg_similarity': sum([chunk['similarity'] for chunk in transcript_content]) / len(transcript_content)
                        }
                    else:
                        # No transcript content found, but we have other relevant info
                        analysis['earnings_call_content'] = {
                            'transcript_found': False,
                            'message': f"No earnings call transcript found in vector database for {ticker}. Showing other relevant financial information.",
                            'alternative_content': search_results[:3]  # Show top 3 alternative results
                        }
                else:
                    # Not an earnings query, just show search results
                    analysis['search_summary'] = {
                        'results_found': len(search_results),
                        'top_result': search_results[0] if search_results else None
                    }
                    
            except Exception as search_error:
                logger.error(f"Search failed: {search_error}")
                analysis['search_error'] = str(search_error)
        elif query and not vector_db_exists:
            # No vector database available, provide a helpful message
            analysis['search_message'] = f"Vector database not available for {ticker}. Please wait while we create one, then try your query again."
            
            # Try to provide a direct answer for common queries
            if 'revenue' in query.lower() and 'growth' in query.lower():
                if 'annual_revenue' in analysis and 'quarterly_revenue' in analysis:
                    analysis['direct_answer'] = {
                        'query': query,
                        'answer': f"The most recent annual revenue was ${analysis.get('annual_revenue', 0):,.2f} and quarterly revenue was ${analysis.get('quarterly_revenue', 0):,.2f}."
                    }
            elif any(keyword in query.lower() for keyword in ['earnings call', 'earnings', 'call', 'transcript']):
                # Try to get earnings call transcript directly
                try:
                    analyzer = FinancialAnalyzer(ticker)
                    transcript = analyzer.get_earnings_call_transcript()
                    if transcript:
                        analysis['earnings_call_content'] = {
                            'transcript_found': True,
                            'content': transcript[:2000] + "..." if len(transcript) > 2000 else transcript,  # Truncate if too long
                            'source': 'direct_retrieval',
                            'note': 'Retrieved directly as vector database was not available'
                        }
                    else:
                        analysis['earnings_call_content'] = {
                            'transcript_found': False,
                            'message': f"No earnings call transcript available for {ticker}."
                        }
                except Exception as transcript_error:
                    logger.error(f"Direct transcript retrieval failed: {transcript_error}")
                    analysis['earnings_call_content'] = {
                        'transcript_found': False,
                        'error': f"Unable to retrieve earnings call transcript: {str(transcript_error)}"
                    }
        
        # Add metadata about vector database status
        analysis['vector_db_status'] = {
            'exists': vector_db_exists,
            'entries': len(db.vectors) if vector_db_exists else 0,
            'ticker': ticker
        }
        
        return analysis
    except Exception as e:
        logger.error(f"Error getting financial insights: {e}")
        # Return minimal data as absolute fallback
        return {
            "error": str(e), 
            "ticker": ticker,
            "fallback_data": True,
            "analysis_date": datetime.now().strftime('%Y-%m-%d')
        }
