# test_economic_factors.py

import os
import json
from datetime import datetime
from tools.economic_factors import (
    get_economic_indicators,
    get_sector_performance,
    analyze_interest_rate_sensitivity,
    analyze_inflation_sensitivity,
    get_macroeconomic_context
)

def print_section(title):
    """Print a section title"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def print_json(data):
    """Print JSON data in a readable format"""
    print(json.dumps(data, indent=2))

def test_economic_indicators():
    """Test economic indicators functionality"""
    print_section("Economic Indicators")
    
    # Get economic indicators
    print("Getting economic indicators...")
    indicators = get_economic_indicators(period="6mo")
    
    if "error" not in indicators:
        # Print interest rates
        if "interest_rates" in indicators:
            print("\nInterest Rates:")
            for name, data in indicators["interest_rates"].items():
                print(f"  {name.replace('_', ' ').title()}: {data['current']:.2f}% ({data['change_pct']:+.2f}%, Trend: {data['trend']})")
        
        # Print inflation indicators
        if "inflation" in indicators:
            print("\nInflation Indicators:")
            for name, data in indicators["inflation"].items():
                print(f"  {name.replace('_', ' ').title()}: {data['current']:.2f} ({data['change_pct']:+.2f}%, Trend: {data['trend']})")
        
        # Print economic growth indicators
        if "economic_growth" in indicators:
            print("\nEconomic Growth Indicators:")
            for name, data in indicators["economic_growth"].items():
                print(f"  {name.replace('_', ' ').title()}: {data['current']:.2f} ({data['change_pct']:+.2f}%, Trend: {data['trend']})")
        
        # Print commodities
        if "commodities" in indicators:
            print("\nCommodities:")
            for name, data in indicators["commodities"].items():
                print(f"  {name.replace('_', ' ').title()}: ${data['current']:.2f} ({data['change_pct']:+.2f}%, Trend: {data['trend']})")
        
        # Print volatility
        if "volatility" in indicators:
            print("\nVolatility:")
            for name, data in indicators["volatility"].items():
                print(f"  {name.replace('_', ' ').title()}: {data['current']:.2f} ({data['change_pct']:+.2f}%, Trend: {data['trend']})")
    else:
        print("Error:", indicators.get("error", "Unknown error"))

def test_sector_performance():
    """Test sector performance functionality"""
    print_section("Sector Performance")
    
    # Get sector performance
    print("Getting sector performance...")
    sectors = get_sector_performance(period="6mo")
    
    if "error" not in sectors:
        # Sort sectors by performance
        ranked_sectors = sorted(sectors.keys(), key=lambda x: sectors[x]["change_pct"], reverse=True)
        
        print("\nSector Performance Ranking:")
        for i, sector in enumerate(ranked_sectors, 1):
            data = sectors[sector]
            print(f"  {i}. {sector.replace('_', ' ').title()}: {data['change_pct']:+.2f}% (Volatility: {data['volatility']:.2f}%, Trend: {data['trend']})")
            
            # Print relative strength vs S&P 500
            if "relative_strength" in data:
                rel_str = data["relative_strength"]
                rel_str_text = "outperforming" if rel_str > 0 else "underperforming"
                print(f"     {rel_str_text} S&P 500 by {abs(rel_str):.2f}%")
    else:
        print("Error:", sectors.get("error", "Unknown error"))

def test_interest_rate_sensitivity():
    """Test interest rate sensitivity analysis"""
    print_section("Interest Rate Sensitivity Analysis")
    
    # Define tickers to analyze
    tickers = ["JPM", "AAPL", "XLU", "XLF", "MSFT"]
    
    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")
        sensitivity = analyze_interest_rate_sensitivity(ticker, period="1y")
        
        if "error" not in sensitivity:
            # Print sensitivity metrics
            sens = sensitivity["sensitivity"]
            print(f"  Sensitivity Level: {sens['level'].title()}")
            print(f"  Direction: {sens['direction'].title()}")
            print(f"  Score: {sens['score']:.2f}")
            
            # Print correlations
            print("\n  Correlations:")
            for rate, corr in sensitivity["correlations"].items():
                print(f"    {rate}: {corr:.2f}")
            
            # Print regression results
            print("\n  Regression Results:")
            for rate, results in sensitivity["regression_results"].items():
                sig = "Significant" if results["significant"] else "Not significant"
                print(f"    {rate}: Beta = {results['beta']:.2f}, p-value = {results['p_value']:.4f} ({sig})")
            
            # Print interpretation summary
            print("\n  Interpretation:")
            interpretation = sensitivity["interpretation"].split("\n\n")
            for para in interpretation:
                print(f"    {para}")
        else:
            print("  Error:", sensitivity.get("error", "Unknown error"))

def test_inflation_sensitivity():
    """Test inflation sensitivity analysis"""
    print_section("Inflation Sensitivity Analysis")
    
    # Define tickers to analyze
    tickers = ["XLE", "AAPL", "XLP", "GLD", "TGT"]
    
    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")
        sensitivity = analyze_inflation_sensitivity(ticker, period="1y")
        
        if "error" not in sensitivity:
            # Print sensitivity metrics
            sens = sensitivity["sensitivity"]
            print(f"  Sensitivity Level: {sens['level'].title()}")
            print(f"  Direction: {sens['direction'].title()}")
            print(f"  Score: {sens['score']:.2f}")
            
            # Print correlations
            print("\n  Correlations:")
            for inflation_indicator, corr in sensitivity["correlations"].items():
                print(f"    {inflation_indicator}: {corr:.2f}")
            
            # Print interpretation summary (first paragraph only)
            print("\n  Interpretation:")
            interpretation = sensitivity["interpretation"].split("\n\n")[0]
            print(f"    {interpretation}")
        else:
            print("  Error:", sensitivity.get("error", "Unknown error"))

def test_macroeconomic_context():
    """Test comprehensive macroeconomic context"""
    print_section("Comprehensive Macroeconomic Context")
    
    # Define tickers to analyze
    tickers = ["JPM", "AAPL", "XLE"]
    
    for ticker in tickers:
        print(f"\nGetting macroeconomic context for {ticker}...")
        context = get_macroeconomic_context(ticker, period="1y")
        
        if "error" not in context:
            # Print sector and industry
            print(f"  Sector: {context['sector']}")
            print(f"  Industry: {context['industry']}")
            
            # Print summary
            print("\n  Macroeconomic Context Summary:")
            summary_lines = context["summary"].split("\n")
            for line in summary_lines:
                print(f"    {line}")
        else:
            print("  Error:", context.get("error", "Unknown error"))

def main():
    """Main function"""
    print_section("Economic Factors Analysis Test")
    
    # Test economic indicators
    test_economic_indicators()
    
    # Test sector performance
    test_sector_performance()
    
    # Test interest rate sensitivity
    test_interest_rate_sensitivity()
    
    # Test inflation sensitivity
    test_inflation_sensitivity()
    
    # Test comprehensive macroeconomic context
    test_macroeconomic_context()

if __name__ == "__main__":
    main()
