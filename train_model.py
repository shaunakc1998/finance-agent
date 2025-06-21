# train_model.py
from tools.forecast import train_model

# Train models for different tickers and model types
tickers = ["NVDA", "AAPL", "MSFT", "TSLA", "AMZN"]
model_types = ["ensemble", "random_forest", "gradient_boost"]

print("Starting model training...")

for ticker in tickers:
    print(f"\n=== Training models for {ticker} ===")
    
    for model_type in model_types:
        print(f"Training {model_type} model...")
        result = train_model(ticker, model_type=model_type, forecast_days=5)
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Success! Model trained with {result.get('training_samples', 0)} samples")
            print(f"RMSE: {result.get('rmse', 'N/A')}")
            print(f"R²: {result.get('r2', 'N/A')}")
            if "cv_rmse" in result:
                print(f"Cross-validation RMSE: {result['cv_rmse']}")

print("\nTraining complete!")
