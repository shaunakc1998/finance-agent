# auto_train_model.py
# Script to automatically check if a model exists for a stock and train it if it doesn't

import os
import sys
import time
from tools.forecast import PriceForecaster, train_model

def check_and_train_model(ticker, model_type="random_forest", forecast_days=10):
    """
    Check if a model exists for the given ticker and train it if it doesn't
    
    Args:
        ticker: Stock or ETF symbol
        model_type: Type of model to use
        forecast_days: Number of days to forecast into the future
        
    Returns:
        Dict with training results or None if model already exists
    """
    print(f"Checking if model exists for {ticker}...")
    
    # Initialize forecaster to check if model exists
    forecaster = PriceForecaster(ticker, model_type)
    forecaster.forecast_days = forecast_days
    
    # Check if model exists and is up to date
    if forecaster.model is not None:
        print(f"Model for {ticker} already exists and is up to date.")
        return None
    
    # Train model if it doesn't exist
    print(f"Training {model_type} model for {ticker} with {forecast_days}-day forecast...")
    result = train_model(ticker, model_type=model_type, forecast_days=forecast_days)
    
    return result

def main():
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python auto_train_model.py TICKER [MODEL_TYPE] [FORECAST_DAYS]")
        print("Example: python auto_train_model.py AAPL random_forest 10")
        sys.exit(1)
    
    # Parse arguments
    ticker = sys.argv[1].upper()
    model_type = sys.argv[2] if len(sys.argv) > 2 else "random_forest"
    forecast_days = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    # Validate model type
    valid_models = ["ensemble", "random_forest", "gradient_boost", "linear", "ridge", "neural_net", "arima"]
    if model_type not in valid_models:
        print(f"Invalid model type: {model_type}")
        print(f"Valid model types: {', '.join(valid_models)}")
        sys.exit(1)
    
    # Check and train model
    start_time = time.time()
    result = check_and_train_model(ticker, model_type, forecast_days)
    end_time = time.time()
    
    # Print results
    if result is None:
        print(f"No training needed for {ticker}.")
    elif "error" in result:
        print(f"Error training model for {ticker}: {result['error']}")
    else:
        print(f"Successfully trained model for {ticker}:")
        print(f"  Model type: {result['model_type']}")
        print(f"  Forecast days: {result['forecast_days']}")
        print(f"  RMSE: {result.get('rmse', 'N/A')}")
        print(f"  R²: {result.get('r2', 'N/A')}")
        print(f"  CV RMSE: {result.get('cv_rmse', 'N/A')}")
        print(f"  CV R²: {result.get('cv_r2', 'N/A')}")
        print(f"  Trained date: {result.get('trained_date', 'N/A')}")
    
    print(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
