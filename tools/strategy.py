# tools/strategy.py

from typing import Dict, Union, List, Tuple
import math
import random  # For simulating historical returns

def make_strategy_decision(fundamentals: Dict, technicals: Dict, investment_type: str = "standard") -> Dict[str, Union[str, float, List[str], Dict]]:
    """
    Applies a comprehensive rules-based strategy to decide whether to buy, hold, or sell a stock or ETF.
    
    Uses multiple technical indicators and fundamental metrics to generate a more robust decision.
    Supports different investment types: "standard", "long_term", "etf_sip"

    Inputs:
        fundamentals: dict from get_fundamentals
        technicals: dict from get_technicals
        investment_type: "standard" (default), "long_term", or "etf_sip"

    Returns:
        dict: {
            action: BUY / HOLD / SELL
            reason: explanation of the rule hit
            confidence: score between 0–1
            signals: list of all signals that contributed to the decision
            price_targets: dict with entry, stop_loss, and take_profit levels
            timeframe: short_term, medium_term, or long_term
            investment_plan: (for ETFs) dict with SIP details
            historical_performance: dict with simulated historical returns
        }
    """
    # Extract all relevant metrics
    rsi = technicals.get("rsi")
    price = technicals.get("current_price")
    sma_50 = technicals.get("sma_50")
    sma_200 = technicals.get("sma_200")
    price_above_sma_50 = technicals.get("price_above_sma_50")
    price_above_sma_200 = technicals.get("price_above_sma_200")
    
    pe = fundamentals.get("pe_ratio")
    peg = fundamentals.get("peg_ratio")
    eps = fundamentals.get("eps")
    beta = fundamentals.get("beta")
    dividend_yield = fundamentals.get("dividend_yield")
    sector = fundamentals.get("sector")
    
    # Initialize signals list and scores
    buy_signals = []
    sell_signals = []
    hold_signals = []
    
    buy_score = 0
    sell_score = 0
    
    # Technical Analysis Signals
    
    # RSI Analysis
    if rsi is not None:
        if rsi < 30:
            buy_signals.append(f"RSI ({rsi}) indicates oversold conditions")
            buy_score += 1.5
        elif rsi < 40:
            buy_signals.append(f"RSI ({rsi}) is approaching oversold territory")
            buy_score += 0.5
        elif rsi > 70:
            sell_signals.append(f"RSI ({rsi}) indicates overbought conditions")
            sell_score += 1.5
        elif rsi > 60:
            sell_signals.append(f"RSI ({rsi}) is approaching overbought territory")
            sell_score += 0.5
        else:
            hold_signals.append(f"RSI ({rsi}) is in neutral territory")
    
    # Moving Average Analysis
    if price and sma_50 and sma_200:
        # Golden Cross / Death Cross
        if sma_50 > sma_200 and abs(sma_50 - sma_200) / sma_200 < 0.03:
            buy_signals.append("50-day SMA is crossing above 200-day SMA (Golden Cross forming)")
            buy_score += 2
        elif sma_50 < sma_200 and abs(sma_50 - sma_200) / sma_200 < 0.03:
            sell_signals.append("50-day SMA is crossing below 200-day SMA (Death Cross forming)")
            sell_score += 2
            
        # Price relative to moving averages
        if price_above_sma_50 and price_above_sma_200:
            buy_signals.append("Price is above both 50-day and 200-day SMAs (bullish)")
            buy_score += 1
        elif not price_above_sma_50 and not price_above_sma_200:
            sell_signals.append("Price is below both 50-day and 200-day SMAs (bearish)")
            sell_score += 1
        
        # Bounce off support or resistance
        if abs(price - sma_50) / sma_50 < 0.02 and price > sma_50:
            buy_signals.append("Price bouncing off 50-day SMA support")
            buy_score += 0.5
        elif abs(price - sma_200) / sma_200 < 0.02 and price > sma_200:
            buy_signals.append("Price bouncing off 200-day SMA support (strong)")
            buy_score += 1
    
    # Fundamental Analysis Signals
    
    # P/E Ratio Analysis
    if pe is not None:
        if pe < 15:
            buy_signals.append(f"P/E ratio ({pe}) suggests undervaluation")
            buy_score += 1
        elif pe > 30:
            sell_signals.append(f"P/E ratio ({pe}) suggests overvaluation")
            sell_score += 1
    
    # PEG Ratio Analysis
    if peg is not None:
        if peg < 1:
            buy_signals.append(f"PEG ratio ({peg}) indicates good value relative to growth")
            buy_score += 1
        elif peg > 2:
            sell_signals.append(f"PEG ratio ({peg}) indicates poor value relative to growth")
            sell_score += 0.5
    
    # Dividend Analysis
    if dividend_yield is not None:
        if dividend_yield > 4:
            buy_signals.append(f"High dividend yield ({dividend_yield}%) provides income potential")
            buy_score += 0.5
    
    # Beta Analysis (Risk)
    if beta is not None:
        if beta > 1.5:
            hold_signals.append(f"High beta ({beta}) indicates increased volatility")
        elif beta < 0.8:
            hold_signals.append(f"Low beta ({beta}) indicates reduced volatility")
    
    # Determine action based on signals
    if buy_score > sell_score:
        if buy_score > 3:
            action = "BUY"
            confidence = min(0.5 + (buy_score - sell_score) / 10, 0.95)
        else:
            action = "HOLD_BULLISH"
            confidence = 0.5 + (buy_score - sell_score) / 10
    elif sell_score > buy_score:
        if sell_score > 3:
            action = "SELL"
            confidence = min(0.5 + (sell_score - buy_score) / 10, 0.95)
        else:
            action = "HOLD_BEARISH"
            confidence = 0.5 + (sell_score - buy_score) / 10
    else:
        action = "HOLD"
        confidence = 0.5
    
    # Adjust for investment type
    if investment_type == "long_term":
        # For long-term investments, focus more on fundamentals and less on short-term technicals
        if buy_score > 0 and pe is not None and pe < 25:
            action = "BUY_LONG"
            confidence = min(0.5 + buy_score / 8, 0.9)
            reason = "Suitable for long-term investment based on fundamentals."
        elif sell_score > 3:
            action = "AVOID"
            confidence = min(0.5 + sell_score / 8, 0.9)
            reason = "Not recommended for long-term investment at current valuation."
        else:
            action = "HOLD_LONG"
            confidence = 0.6
            reason = "Maintain existing long-term position."
        
        timeframe = "long_term"
        
    elif investment_type == "etf_sip":
        # For ETF SIPs, focus on long-term potential and consistency
        action = "SIP_INVEST"
        confidence = 0.85
        reason = "Suitable for systematic investment plan (SIP)."
        timeframe = "long_term"
        
    else:
        # Standard investment timeframe determination
        if action in ["BUY", "SELL"] and confidence > 0.8:
            timeframe = "short_term"
        elif action in ["BUY", "SELL", "HOLD_BULLISH", "HOLD_BEARISH"] and confidence > 0.6:
            timeframe = "medium_term"
        else:
            timeframe = "long_term"
    
    # Calculate price targets
    price_targets = calculate_price_targets(action, price, sma_50, sma_200, rsi)
    
    # Generate investment plan for ETFs
    investment_plan = None
    if investment_type == "etf_sip":
        investment_plan = generate_sip_plan(fundamentals, technicals)
    
    # Generate historical performance simulation
    historical_performance = simulate_historical_performance(fundamentals, technicals, timeframe)
    
    # Compile the primary reason
    if action == "BUY":
        primary_signals = sorted(buy_signals, key=len, reverse=True)[:2]
        reason = " and ".join(primary_signals) + "."
    elif action == "SELL":
        primary_signals = sorted(sell_signals, key=len, reverse=True)[:2]
        reason = " and ".join(primary_signals) + "."
    elif action == "HOLD_BULLISH":
        reason = "Mixed signals with bullish bias. Consider partial position."
    elif action == "HOLD_BEARISH":
        reason = "Mixed signals with bearish bias. Consider reducing exposure."
    else:
        reason = "No strong directional signals present."
    
    # Combine all signals
    all_signals = buy_signals + sell_signals + hold_signals
    
    result = {
        "action": action,
        "reason": reason,
        "confidence": round(confidence, 2),
        "signals": all_signals,
        "price_targets": price_targets,
        "timeframe": timeframe,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "investment_type": investment_type
    }
    
    # Add investment plan if available
    if investment_plan:
        result["investment_plan"] = investment_plan
    
    # Add historical performance
    if historical_performance:
        result["historical_performance"] = historical_performance
    
    return result

def generate_sip_plan(fundamentals: Dict, technicals: Dict) -> Dict[str, Union[float, str, List]]:
    """Generate a Systematic Investment Plan (SIP) for ETFs"""
    symbol = fundamentals.get("symbol", "")
    price = technicals.get("current_price")
    
    if not price:
        # Default values if price is not available
        return {
            "recommended_frequency": "monthly",
            "monthly_investment_options": {
                "conservative": 500.0,
                "moderate": 1000.0,
                "aggressive": 2000.0
            },
            "estimated_annual_return": round(random.uniform(7.0, 12.0), 2),
            "compound_growth": {
                "5_year": round(random.uniform(40.0, 80.0), 2),
                "10_year": round(random.uniform(100.0, 200.0), 2),
                "20_year": round(random.uniform(300.0, 600.0), 2)
            },
            "dollar_cost_averaging_benefit": "Medium"
        }
    
    # Calculate recommended SIP amounts (ensure we have at least 1 share)
    monthly_small = max(1, round(500 / price)) * price
    monthly_medium = max(1, round(1000 / price)) * price
    monthly_large = max(1, round(2000 / price)) * price
    
    # Recommend frequency
    frequencies = ["weekly", "biweekly", "monthly"]
    recommended_frequency = "monthly"  # Default
    
    # Estimate annual returns (simplified)
    estimated_annual_return = random.uniform(7.0, 12.0)  # Historical average for broad market ETFs
    
    # Calculate compound growth
    five_year_growth = (1 + estimated_annual_return/100) ** 5 - 1
    ten_year_growth = (1 + estimated_annual_return/100) ** 10 - 1
    twenty_year_growth = (1 + estimated_annual_return/100) ** 20 - 1
    
    # Generate plan
    return {
        "recommended_frequency": recommended_frequency,
        "monthly_investment_options": {
            "conservative": monthly_small,
            "moderate": monthly_medium,
            "aggressive": monthly_large
        },
        "estimated_annual_return": round(estimated_annual_return, 2),
        "compound_growth": {
            "5_year": round(five_year_growth * 100, 2),
            "10_year": round(ten_year_growth * 100, 2),
            "20_year": round(twenty_year_growth * 100, 2)
        },
        "dollar_cost_averaging_benefit": "High" if technicals.get("rsi", 50) > 60 else "Medium"
    }

def simulate_historical_performance(fundamentals: Dict, technicals: Dict, timeframe: str) -> Dict[str, float]:
    """Simulate historical performance based on available data"""
    # This is a simplified simulation - in a real app, you would use actual historical data
    
    # Estimate returns based on sector and beta
    sector = fundamentals.get("sector", "Unknown")
    beta = fundamentals.get("beta", 1.0)
    
    # Base returns by timeframe
    base_returns = {
        "1_year": random.uniform(5.0, 15.0),
        "3_year": random.uniform(15.0, 45.0),
        "5_year": random.uniform(25.0, 75.0),
        "10_year": random.uniform(50.0, 150.0)
    }
    
    # Adjust based on sector performance
    sector_multipliers = {
        "Technology": 1.2,
        "Healthcare": 1.1,
        "Consumer Cyclical": 1.0,
        "Financial Services": 0.9,
        "Communication Services": 1.1,
        "Industrials": 0.95,
        "Consumer Defensive": 0.85,
        "Energy": 0.8,
        "Basic Materials": 0.9,
        "Real Estate": 0.85,
        "Utilities": 0.8
    }
    
    sector_multiplier = sector_multipliers.get(sector, 1.0)
    
    # Adjust based on beta (volatility)
    beta_adjustment = beta / 1.0
    
    # Calculate adjusted returns
    adjusted_returns = {}
    for period, base_return in base_returns.items():
        adjusted_returns[period] = round(base_return * sector_multiplier * beta_adjustment, 2)
    
    return adjusted_returns

def calculate_price_targets(action: str, price: float, sma_50: float, sma_200: float, rsi: float) -> Dict[str, float]:
    """Calculate entry, stop loss, and take profit levels based on technical indicators"""
    # Handle None values
    if price is None:
        return {"entry": None, "stop_loss": None, "take_profit": None}
    
    # If moving averages are None, use price-based estimates
    if sma_50 is None:
        sma_50 = price * 0.95  # Estimate SMA50 as 5% below current price
    
    if sma_200 is None:
        sma_200 = price * 0.90  # Estimate SMA200 as 10% below current price
    
    # Default values
    entry = price
    stop_loss = None
    take_profit = None
    
    # Calculate based on action
    if action in ["BUY", "HOLD_BULLISH"]:
        # Entry point slightly below current price for better value
        entry = round(price * 0.98, 2)
        
        # Stop loss at recent support (estimate using SMAs)
        if price > sma_50:
            stop_loss = round(min(sma_50, price * 0.92), 2)
        else:
            stop_loss = round(min(sma_200, price * 0.90), 2)
        
        # Take profit based on risk-reward ratio of 1:2
        if stop_loss:
            risk = price - stop_loss
            take_profit = round(price + (risk * 2), 2)
        else:
            take_profit = round(price * 1.15, 2)
            
    elif action in ["SELL", "HOLD_BEARISH"]:
        # Entry point slightly above current price for confirmation
        entry = round(price * 1.02, 2)
        
        # Stop loss at recent resistance (estimate using SMAs)
        if price < sma_50:
            stop_loss = round(max(sma_50, price * 1.08), 2)
        else:
            stop_loss = round(max(sma_200, price * 1.10), 2)
        
        # Take profit based on risk-reward ratio of 1:2
        if stop_loss:
            risk = stop_loss - price
            take_profit = round(price - (risk * 2), 2)
        else:
            take_profit = round(price * 0.85, 2)
    
    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit
    }
