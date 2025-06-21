#!/usr/bin/env python3
# test_fallback.py - Test script for fallback mechanisms

import sys
import os
import json

# Add the parent directory to the path to import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the modules we want to test
from tools.fundamentals import get_fundamentals
from tools.technicals import get_technicals

def test_stock(ticker):
    """Test both fundamentals and technicals for a given ticker"""
    print(f"\n{'='*50}")
    print(f"Testing ticker: {ticker}")
    print(f"{'='*50}")
    
    # Test fundamentals
    print("\nFundamentals:")
    fundamentals = get_fundamentals(ticker)
    print(json.dumps(fundamentals, indent=2))
    
    # Test technicals
    print("\nTechnicals:")
    technicals = get_technicals(ticker)
    print(json.dumps(technicals, indent=2))

if __name__ == "__main__":
    # Test a few different tickers
    test_stock("AAPL")  # Common stock that should work with yfinance
    test_stock("MCHP")  # Stock that might need fallback
    test_stock("AI")    # Another stock that might need fallback
    test_stock("NONEXISTENT")  # A non-existent ticker to test error handling
