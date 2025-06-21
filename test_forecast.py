# test_forecast.py
from tools.forecast import get_forecast
import json

# Test forecasts for different tickers and model types
tickers = ["NVDA", "AAPL", "MSFT", "TSLA", "AMZN"]
model_types = ["ensemble", "random_forest", "gradient_boost"]
forecast_days = [5, 10, 15]

print("Starting forecast testing...")

for ticker in tickers:
    print(f"\n=== Testing forecasts for {ticker} ===")
    
    for model_type in model_types:
        print(f"\n--- Using {model_type} model ---")
        
        for days in forecast_days:
            print(f"Generating {days}-day forecast...")
            result = get_forecast(ticker, model_type=model_type, forecast_days=days)
            
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                # Format the forecast results
                current_price = result.get('current_price', 'N/A')
                predicted_return = result.get('predicted_return', 'N/A')
                predicted_price = result.get('predicted_price', 'N/A')
                
                print(f"Current price: ${current_price}")
                print(f"Predicted {days}-day return: {predicted_return}%")
                print(f"Predicted price: ${predicted_price}")
                
                # Print confidence intervals
                if 'confidence_intervals' in result and '90%' in result['confidence_intervals']:
                    ci_90 = result['confidence_intervals']['90%']
                    print(f"90% Confidence interval: ${ci_90['lower_price']} to ${ci_90['upper_price']}")
                    print(f"90% Return range: {ci_90['lower_return']}% to {ci_90['upper_return']}%")
                
                # Print forecast metrics
                if 'forecast_metrics' in result:
                    metrics = result['forecast_metrics']
                    print(f"Direction: {metrics.get('direction', 'N/A')}")
                    print(f"Strength: {metrics.get('strength', 'N/A')}")
                    print(f"Z-score: {metrics.get('z_score', 'N/A')}")
                
                # Print top features if available
                if 'model_info' in result and 'feature_importance' in result['model_info'] and result['model_info']['feature_importance']:
                    print("\nTop 5 important features:")
                    features = result['model_info']['feature_importance']
                    for i, (feature, importance) in enumerate(list(features.items())[:5]):
                        print(f"{i+1}. {feature}: {importance}")

print("\nTesting complete!")
