# test_financial_analysis.py

import os
import sys
import json
from datetime import datetime
from tools.financial_analysis import (
    FinancialAnalyzer, 
    get_financial_analysis,
    generate_embeddings,
    store_financial_analysis_in_vector_db,
    search_financial_analysis,
    get_financial_insights
)

def test_financial_analyzer(ticker="AAPL"):
    """Test the FinancialAnalyzer class"""
    print(f"\n=== Testing FinancialAnalyzer for {ticker} ===")
    
    # Initialize analyzer
    analyzer = FinancialAnalyzer(ticker)
    
    # Test getting financial statements
    print("\nTesting get_financial_statements()...")
    statements = analyzer.get_financial_statements()
    if statements:
        print("✅ Successfully retrieved financial statements")
        # Print some sample data
        income_stmt = statements.get('income_statement', {}).get('annual', None)
        if income_stmt is not None and not income_stmt.empty:
            print(f"Sample annual income statement data (most recent):")
            print(income_stmt.iloc[:5, 0])  # First 5 rows of most recent column
    else:
        print("❌ Failed to retrieve financial statements")
    
    # Test getting earnings info
    print("\nTesting get_earnings_info()...")
    earnings = analyzer.get_earnings_info()
    if earnings:
        print("✅ Successfully retrieved earnings info")
        # Print earnings calendar if available
        if 'calendar' in earnings and earnings['calendar'] is not None:
            print("Upcoming earnings date:", earnings['calendar'].get('Earnings Date', 'N/A'))
    else:
        print("❌ Failed to retrieve earnings info")
    
    # Test getting SEC filings
    print("\nTesting get_sec_filings()...")
    filings = analyzer.get_sec_filings(filing_type="10-Q", limit=3)
    if filings:
        print(f"✅ Successfully retrieved {len(filings)} SEC filings")
        # Print first filing
        if filings:
            print("Most recent filing:")
            print(f"  Form: {filings[0].get('form', 'N/A')}")
            print(f"  Date: {filings[0].get('filing_date', 'N/A')}")
            print(f"  URL: {filings[0].get('url', 'N/A')}")
    else:
        print("❌ Failed to retrieve SEC filings")
    
    # Test getting earnings call transcript
    print("\nTesting get_earnings_call_transcript()...")
    transcript = analyzer.get_earnings_call_transcript()
    if transcript:
        print("✅ Successfully retrieved earnings call transcript")
        print(f"Transcript length: {len(transcript)} characters")
        print("First 200 characters:")
        print(transcript[:200] + "...")
    else:
        print("❌ Failed to retrieve earnings call transcript")
    
    # Test calculating key metrics
    print("\nTesting calculate_key_metrics()...")
    metrics = analyzer.calculate_key_metrics()
    if metrics:
        print("✅ Successfully calculated key metrics")
        print("Sample metrics:")
        for key in ['company_name', 'sector', 'pe_ratio', 'profit_margin', 'dividend_yield']:
            print(f"  {key}: {metrics.get(key, 'N/A')}")
    else:
        print("❌ Failed to calculate key metrics")
    
    # Test getting comprehensive analysis
    print("\nTesting get_comprehensive_analysis()...")
    analysis = analyzer.get_comprehensive_analysis()
    if analysis:
        print("✅ Successfully retrieved comprehensive analysis")
        print("Analysis includes:")
        for key in analysis.keys():
            print(f"  - {key}")
    else:
        print("❌ Failed to retrieve comprehensive analysis")
    
    return analysis

def test_vector_embeddings(text="Apple Inc. is a technology company that makes iPhones, iPads, and Mac computers."):
    """Test generating vector embeddings"""
    print("\n=== Testing Vector Embeddings ===")
    
    print(f"Generating embeddings for text: '{text}'")
    embeddings = generate_embeddings(text)
    
    if embeddings:
        print(f"✅ Successfully generated embeddings")
        print(f"Embedding dimension: {len(embeddings)}")
        print(f"First 5 values: {embeddings[:5]}")
    else:
        print("❌ Failed to generate embeddings")
    
    return embeddings

def test_vector_storage(ticker="AAPL"):
    """Test storing financial analysis in vector database"""
    print(f"\n=== Testing Vector Storage for {ticker} ===")
    
    print(f"Storing financial analysis for {ticker} in vector database...")
    success = store_financial_analysis_in_vector_db(ticker)
    
    if success:
        print(f"✅ Successfully stored financial analysis for {ticker} in vector database")
    else:
        print(f"❌ Failed to store financial analysis for {ticker} in vector database")
    
    return success

def test_vector_search(ticker="AAPL", query="What is the P/E ratio?"):
    """Test searching financial analysis in vector database"""
    print(f"\n=== Testing Vector Search for {ticker} ===")
    
    print(f"Searching for '{query}' in {ticker} financial analysis...")
    results = search_financial_analysis(ticker, query)
    
    if results:
        print(f"✅ Successfully searched financial analysis")
        print(f"Found {len(results)} results")
        print("Top result:")
        print(f"  Similarity: {results[0]['similarity']:.4f}")
        print(f"  Text: {results[0]['text'][:200]}...")
    else:
        print(f"❌ Failed to search financial analysis")
    
    return results

def test_financial_insights(ticker="AAPL", query="What were the recent earnings?"):
    """Test getting financial insights"""
    print(f"\n=== Testing Financial Insights for {ticker} ===")
    
    print(f"Getting financial insights for {ticker} with query '{query}'...")
    insights = get_financial_insights(ticker, query)
    
    if insights and 'error' not in insights:
        print(f"✅ Successfully retrieved financial insights")
        print("Insights include:")
        for key in insights.keys():
            print(f"  - {key}")
        
        if 'search_results' in insights:
            print(f"\nSearch results for '{query}':")
            for i, result in enumerate(insights['search_results'][:2]):  # Show top 2 results
                print(f"Result {i+1} (similarity: {result['similarity']:.4f}):")
                print(f"  {result['text'][:200]}...")
    else:
        print(f"❌ Failed to retrieve financial insights")
        if 'error' in insights:
            print(f"Error: {insights['error']}")
    
    return insights

def main():
    """Main function to run all tests"""
    # Get ticker from command line argument or use default
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    
    print(f"Running financial analysis tests for {ticker}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Run tests
    analysis = test_financial_analyzer(ticker)
    embeddings = test_vector_embeddings()
    storage_success = test_vector_storage(ticker)
    search_results = test_vector_search(ticker, "What is the P/E ratio?")
    insights = test_financial_insights(ticker, "What were the recent earnings?")
    
    print("\n" + "=" * 50)
    print(f"Tests completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save results to file
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f"{ticker}_financial_analysis_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(results_file, 'w') as f:
        json.dump({
            'ticker': ticker,
            'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analysis': {k: str(v) for k, v in analysis.items()},
            'embedding_dimension': len(embeddings),
            'storage_success': storage_success,
            'search_results_count': len(search_results),
            'insights_keys': list(insights.keys())
        }, f, indent=2)
    
    print(f"Test results saved to: {results_file}")

if __name__ == "__main__":
    main()
