# test_jpm_visualization.py

from tools.visualization import generate_stock_price_chart, generate_forecast_chart
import os
from tools.forecast import get_forecast

def test_jpm_price_chart():
    print("Testing JPM price chart visualization...")
    ticker = "JPM"
    
    # Test with different periods
    periods = ["3d", "5d", "1mo"]
    
    for period in periods:
        print(f"Testing period: {period}")
        image_path = generate_stock_price_chart(
            ticker=ticker,
            period=period,
            interval="1d",
            chart_type="line",
            moving_averages=[20, 50],
            volume=True
        )
        
        if image_path:
            print(f"✅ Successfully generated price chart: {image_path}")
            if os.path.exists(image_path):
                print(f"✅ Image file exists")
            else:
                print(f"❌ Image file does not exist")
        else:
            print(f"❌ Failed to generate price chart")

def test_jpm_forecast_chart():
    print("\nTesting JPM forecast chart visualization...")
    ticker = "JPM"
    
    # Get forecast data
    print("Getting forecast data...")
    forecast_data = get_forecast(ticker, forecast_days=20)
    
    if 'error' in forecast_data:
        print(f"❌ Error getting forecast data: {forecast_data['error']}")
        return
    
    print("Generating forecast chart...")
    image_path = generate_forecast_chart(
        ticker=ticker,
        forecast_data=forecast_data,
        days=20
    )
    
    if image_path:
        print(f"✅ Successfully generated forecast chart: {image_path}")
        if os.path.exists(image_path):
            print(f"✅ Image file exists")
        else:
            print(f"❌ Image file does not exist")
    else:
        print(f"❌ Failed to generate forecast chart")

if __name__ == "__main__":
    print("Testing JPM visualizations...")
    
    # Run tests
    test_jpm_price_chart()
    test_jpm_forecast_chart()
    
    print("\nTests completed.")
