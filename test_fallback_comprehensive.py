#!/usr/bin/env python3
# test_fallback_comprehensive.py - Comprehensive test script for fallback mechanisms

import sys
import os
import json
import time
from pprint import pprint

# Add the parent directory to the path to import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the modules we want to test
from tools.fundamentals import get_fundamentals, get_fallback_fundamentals
from tools.technicals import get_technicals, get_alphavantage_price

def test_direct_fallback():
    """Test the fallback functions directly"""
    print("\n" + "="*80)
    print("TESTING DIRECT FALLBACK FUNCTIONS")
    print("="*80)
    
    # Test Alpha Vantage price function
    tickers = ["AAPL", "MSFT", "GOOGL", "MCHP", "AI", "NONEXISTENT"]
    
    print("\nTesting Alpha Vantage Price Function:")
    for ticker in tickers:
        price = get_alphavantage_price(ticker)
        print(f"{ticker}: {price}")
    
    # Test fallback fundamentals function
    print("\nTesting Fallback Fundamentals Function:")
    for ticker in tickers:
        fundamentals = get_fallback_fundamentals(ticker)
        print(f"{ticker}: {len(fundamentals)} fields retrieved")
        if fundamentals:
            print(f"  Sample fields: {list(fundamentals.keys())[:3]}")

def test_caching():
    """Test the caching mechanism"""
    print("\n" + "="*80)
    print("TESTING CACHING MECHANISM")
    print("="*80)
    
    # Test caching for technicals
    ticker = "AAPL"
    
    print(f"\nTesting caching for {ticker} technicals:")
    
    # First call - should hit the API
    start_time = time.time()
    technicals1 = get_technicals(ticker)
    first_call_time = time.time() - start_time
    
    print(f"First call time: {first_call_time:.4f} seconds")
    
    # Second call - should use cache
    start_time = time.time()
    technicals2 = get_technicals(ticker)
    second_call_time = time.time() - start_time
    
    print(f"Second call time: {second_call_time:.4f} seconds")
    print(f"Cache speedup: {first_call_time / second_call_time:.2f}x faster")
    
    # Test caching for fundamentals
    print(f"\nTesting caching for {ticker} fundamentals:")
    
    # First call - should hit the API
    start_time = time.time()
    fundamentals1 = get_fundamentals(ticker)
    first_call_time = time.time() - start_time
    
    print(f"First call time: {first_call_time:.4f} seconds")
    
    # Second call - should use cache
    start_time = time.time()
    fundamentals2 = get_fundamentals(ticker)
    second_call_time = time.time() - start_time
    
    print(f"Second call time: {second_call_time:.4f} seconds")
    print(f"Cache speedup: {first_call_time / second_call_time:.2f}x faster")

def test_error_handling():
    """Test error handling for non-existent tickers"""
    print("\n" + "="*80)
    print("TESTING ERROR HANDLING")
    print("="*80)
    
    ticker = "NONEXISTENT"
    
    print(f"\nTesting error handling for non-existent ticker {ticker}:")
    
    # Test fundamentals error handling
    fundamentals = get_fundamentals(ticker)
    print("\nFundamentals result:")
    pprint(fundamentals)
    
    # Test technicals error handling
    technicals = get_technicals(ticker)
    print("\nTechnicals result:")
    pprint(technicals)

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("COMPREHENSIVE FALLBACK MECHANISM TEST")
    print("="*80)
    
    # Test direct fallback functions
    test_direct_fallback()
    
    # Test caching
    test_caching()
    
    # Test error handling
    test_error_handling()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()
