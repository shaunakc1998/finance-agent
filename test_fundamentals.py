# test_fundamentals.py
from tools.fundamentals import get_fundamentals

if __name__ == "__main__":
    ticker = input("Enter a ticker symbol (e.g. AAPL): ").strip().upper()
    data = get_fundamentals(ticker)
    for k, v in data.items():
        print(f"{k}: {v}")