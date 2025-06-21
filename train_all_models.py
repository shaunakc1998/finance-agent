# train_all_models.py
# Script to pre-train forecast models for a comprehensive list of stocks

import os
import time
import pandas as pd
from tools.forecast import train_model
from concurrent.futures import ThreadPoolExecutor, as_completed

# List of popular stocks across different sectors
STOCKS = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "INTC", "AMD", "CSCO", "ORCL", "IBM", "ADBE", "CRM", "NFLX",
    # Healthcare
    "JNJ", "PFE", "MRK", "UNH", "ABT", "TMO", "LLY", "ABBV", "BMY", "AMGN", "MDT", "GILD", "ISRG", "REGN", "VRTX",
    # Financial
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "AXP", "V", "MA", "PYPL", "SCHW", "CME", "ICE", "SPGI",
    # Consumer
    "PG", "KO", "PEP", "WMT", "COST", "HD", "MCD", "SBUX", "NKE", "DIS", "CMCSA", "TGT", "LOW", "YUM", "MAR",
    # Industrial
    "GE", "HON", "MMM", "CAT", "DE", "BA", "LMT", "RTX", "UPS", "FDX", "UNP", "CSX", "EMR", "ETN", "ITW",
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC", "OXY", "KMI", "WMB", "ET", "BP", "SHEL", "TTE",
    # Telecom
    "T", "VZ", "TMUS", "LUMN", "CHTR", "DISH", "ATUS", "CCOI", "BAND", "CNSL", "ATNI", "GSAT", "SHEN", "GLIBA", "CABO",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "PCG", "ED", "XEL", "ES", "WEC", "CMS", "AEE", "ETR",
    # Real Estate
    "AMT", "PLD", "CCI", "EQIX", "PSA", "O", "DLR", "WELL", "AVB", "EQR", "SPG", "VTR", "BXP", "ARE", "UDR",
    # Materials
    "LIN", "APD", "ECL", "SHW", "FCX", "NEM", "NUE", "DOW", "DD", "VMC", "MLM", "CF", "MOS", "ALB", "FMC",
    # Popular ETFs
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO", "BND", "AGG", "GLD", "SLV", "XLF", "XLE", "XLK"
]

# Model types to train
MODEL_TYPES = ["random_forest", "gradient_boost"]

# Forecast periods to train for
FORECAST_DAYS = [5, 10, 20]

# Directory to store training logs
LOG_DIR = "training_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file
log_file = os.path.join(LOG_DIR, f"training_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")

# Function to train a model for a specific stock, model type, and forecast period
def train_stock_model(stock, model_type, days):
    try:
        print(f"Training {model_type} model for {stock} with {days}-day forecast...")
        result = train_model(stock, model_type=model_type, forecast_days=days)
        
        # Extract relevant metrics
        metrics = {
            "ticker": stock,
            "model_type": model_type,
            "forecast_days": days,
            "status": result.get("status", "failed"),
            "rmse": result.get("rmse", None),
            "r2": result.get("r2", None),
            "cv_rmse": result.get("cv_rmse", None),
            "cv_r2": result.get("cv_r2", None),
            "trained_date": result.get("trained_date", None),
            "error": result.get("error", None)
        }
        
        return metrics
    except Exception as e:
        print(f"Error training model for {stock}: {str(e)}")
        return {
            "ticker": stock,
            "model_type": model_type,
            "forecast_days": days,
            "status": "error",
            "error": str(e)
        }

def main():
    print(f"Starting training for {len(STOCKS)} stocks, {len(MODEL_TYPES)} model types, and {len(FORECAST_DAYS)} forecast periods...")
    print(f"Total models to train: {len(STOCKS) * len(MODEL_TYPES) * len(FORECAST_DAYS)}")
    
    # List to store all training results
    all_results = []
    
    # Use ThreadPoolExecutor to train models in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Create a list of futures
        futures = []
        
        # Submit training tasks
        for stock in STOCKS:
            for model_type in MODEL_TYPES:
                for days in FORECAST_DAYS:
                    future = executor.submit(train_stock_model, stock, model_type, days)
                    futures.append(future)
        
        # Process results as they complete
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            all_results.append(result)
            
            # Print progress
            progress = (i + 1) / len(futures) * 100
            print(f"Progress: {progress:.2f}% ({i + 1}/{len(futures)})")
            
            # Save intermediate results to CSV
            df = pd.DataFrame(all_results)
            df.to_csv(log_file, index=False)
    
    # Save final results to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(log_file, index=False)
    
    print(f"Training complete! Results saved to {log_file}")
    
    # Print summary statistics
    success_count = sum(1 for r in all_results if r["status"] == "success")
    print(f"Successfully trained {success_count} out of {len(all_results)} models ({success_count/len(all_results)*100:.2f}%)")

if __name__ == "__main__":
    main()
