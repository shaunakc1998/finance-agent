# test_industry_comparison.py

from tools.financial_analysis import get_industry_comparison
import json

def test_industry_comparison():
    print("Testing industry comparison for AAPL...")
    result = get_industry_comparison('AAPL')
    
    # Check if we got a valid result
    if 'error' in result:
        print(f"Error: {result['error']}")
        return False
    
    # Check if we have the expected keys
    expected_keys = ['ticker', 'sector', 'industry', 'peer_companies', 'company_metrics', 
                     'industry_averages', 'relative_performance', 'sector_etf_performance']
    
    for key in expected_keys:
        if key not in result:
            print(f"Missing key: {key}")
            return False
    
    # Check if relative_performance has values
    if not result['relative_performance']:
        print("No relative performance metrics found")
        return False
    
    # Print some key metrics for verification
    print("\nIndustry Comparison Results:")
    print(f"Ticker: {result['ticker']}")
    print(f"Sector: {result['sector']}")
    print(f"Industry: {result['industry']}")
    print(f"Peer Companies: {', '.join(result['peer_companies'])}")
    
    print("\nRelative Performance (Company vs Industry):")
    for metric, value in result['relative_performance'].items():
        # Format the value as a percentage
        formatted_value = f"{value * 100:.2f}%"
        # Determine if this is good or bad
        if metric in ['roa', 'roe', 'revenue_growth', 'earnings_growth', 'current_ratio']:
            # For these metrics, higher is better
            performance = "better" if value > 0 else "worse"
        elif metric in ['pe_ratio', 'forward_pe', 'price_to_book', 'price_to_sales', 'debt_to_equity']:
            # For these metrics, lower is better
            performance = "better" if value > 0 else "worse"
        else:
            performance = "different"
        
        print(f"  {metric}: {formatted_value} ({performance} than industry average)")
    
    print("\nSector ETF Performance:")
    if result['sector_etf_performance']:
        etf = result['sector_etf_performance']
        print(f"  ETF: {etf.get('etf_ticker', 'N/A')}")
        print(f"  1-Month Change: {etf.get('change_1m', 0):.2f}%")
        print(f"  3-Month Change: {etf.get('change_3m', 0):.2f}%")
        print(f"  1-Year Change: {etf.get('change_1y', 0):.2f}%")
    else:
        print("  No sector ETF data available")
    
    return True

if __name__ == "__main__":
    success = test_industry_comparison()
    print(f"\nTest {'passed' if success else 'failed'}")
