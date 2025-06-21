# test_technicals.py
from tools.technicals import get_technicals

if __name__ == "__main__":
    ticker = input("Enter a stock/ETF symbol: ").strip().upper()
    result = get_technicals(ticker)
    for k, v in result.items():
        print(f"{k}: {v}")