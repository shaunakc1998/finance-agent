# test_visualization.py

from tools.visualization import generate_visualization
import os

def test_price_chart():
    print("Testing price chart visualization...")
    result = generate_visualization(
        viz_type="price_chart",
        ticker="AAPL",
        period="1y",
        interval="1d",
        chart_type="line",
        moving_averages=[20, 50, 200],
        volume=True
    )
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return False
    
    print(f"Generated price chart: {result['image_path']}")
    if os.path.exists(result['image_path']):
        print("✅ Price chart test passed")
        return True
    else:
        print("❌ Price chart test failed - file not found")
        return False

def test_comparison_chart():
    print("\nTesting comparison chart visualization...")
    result = generate_visualization(
        viz_type="comparison_chart",
        tickers=["AAPL", "MSFT", "GOOGL"],
        period="1y",
        metric="price",
        chart_type="line",
        normalize=True
    )
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return False
    
    print(f"Generated comparison chart: {result['image_path']}")
    if os.path.exists(result['image_path']):
        print("✅ Comparison chart test passed")
        return True
    else:
        print("❌ Comparison chart test failed - file not found")
        return False

def test_industry_chart():
    print("\nTesting industry comparison chart...")
    result = generate_visualization(
        viz_type="industry_chart",
        ticker="AAPL",
        metrics=["pe_ratio", "price_to_book", "profit_margin", "roe", "revenue_growth"],
        chart_type="bar"
    )
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return False
    
    print(f"Generated industry chart: {result['image_path']}")
    if os.path.exists(result['image_path']):
        print("✅ Industry chart test passed")
        return True
    else:
        print("❌ Industry chart test failed - file not found")
        return False

def test_correlation_matrix():
    print("\nTesting correlation matrix...")
    result = generate_visualization(
        viz_type="correlation_matrix",
        tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
        period="1y"
    )
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return False
    
    print(f"Generated correlation matrix: {result['image_path']}")
    if os.path.exists(result['image_path']):
        print("✅ Correlation matrix test passed")
        return True
    else:
        print("❌ Correlation matrix test failed - file not found")
        return False

if __name__ == "__main__":
    print("Testing visualization tools...")
    
    # Run tests
    price_chart_result = test_price_chart()
    comparison_chart_result = test_comparison_chart()
    industry_chart_result = test_industry_chart()
    correlation_matrix_result = test_correlation_matrix()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Price Chart: {'✅ Passed' if price_chart_result else '❌ Failed'}")
    print(f"Comparison Chart: {'✅ Passed' if comparison_chart_result else '❌ Failed'}")
    print(f"Industry Chart: {'✅ Passed' if industry_chart_result else '❌ Failed'}")
    print(f"Correlation Matrix: {'✅ Passed' if correlation_matrix_result else '❌ Failed'}")
    
    # Overall result
    all_passed = all([price_chart_result, comparison_chart_result, industry_chart_result, correlation_matrix_result])
    print(f"\nOverall Result: {'✅ All tests passed' if all_passed else '❌ Some tests failed'}")
    
    # Show where to find the generated images
    if any([price_chart_result, comparison_chart_result, industry_chart_result, correlation_matrix_result]):
        viz_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations")
        print(f"\nGenerated images can be found in: {viz_dir}")
