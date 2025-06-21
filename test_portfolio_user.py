# test_portfolio_user.py

import os
import json
from datetime import datetime, timedelta
from tools.portfolio import (
    get_optimal_portfolio,
    analyze_existing_portfolio,
    get_efficient_frontier,
    get_portfolio_rebalance,
    stress_test_portfolio
)
from tools.user_profile import (
    create_user_profile,
    get_user_profile,
    update_user_preferences,
    add_investment_goal,
    get_personalized_recommendations
)

def print_section(title):
    """Print a section title"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def print_json(data):
    """Print JSON data in a readable format"""
    print(json.dumps(data, indent=2))

def test_portfolio_optimization():
    """Test portfolio optimization features"""
    print_section("Portfolio Optimization")
    
    # Define a list of tickers for a sample portfolio
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "V"]
    
    # Get optimal portfolio
    print("Getting optimal portfolio...")
    optimal_portfolio = get_optimal_portfolio(tickers, period="1y")
    
    # Print optimal weights
    print("\nOptimal Portfolio Weights:")
    if "weights" in optimal_portfolio:
        for ticker, weight in optimal_portfolio["weights"].items():
            print(f"  {ticker}: {weight:.2%}")
        
        print(f"\nExpected Annual Return: {optimal_portfolio['expected_annual_return']:.2f}%")
        print(f"Annual Volatility: {optimal_portfolio['annual_volatility']:.2f}%")
        print(f"Sharpe Ratio: {optimal_portfolio['sharpe_ratio']:.2f}")
    else:
        print("Error:", optimal_portfolio.get("error", "Unknown error"))
    
    # Define a current portfolio
    current_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    current_weights = [0.4, 0.3, 0.2, 0.1]
    
    # Analyze existing portfolio
    print("\nAnalyzing existing portfolio...")
    portfolio_analysis = analyze_existing_portfolio(current_tickers, current_weights)
    
    if "error" not in portfolio_analysis:
        print(f"\nCurrent Portfolio:")
        for ticker, weight in portfolio_analysis["weights"].items():
            print(f"  {ticker}: {weight:.2%}")
        
        print(f"\nExpected Annual Return: {portfolio_analysis['expected_annual_return']:.2f}%")
        print(f"Annual Volatility: {portfolio_analysis['annual_volatility']:.2f}%")
        print(f"Sharpe Ratio: {portfolio_analysis['sharpe_ratio']:.2f}")
        
        # Print improvement potential
        if portfolio_analysis.get("improvement_potential"):
            improvement = portfolio_analysis["improvement_potential"]
            print("\nImprovement Potential:")
            print(f"  Current Sharpe: {improvement['current_sharpe']:.2f}")
            print(f"  Optimal Sharpe: {improvement['optimal_sharpe']:.2f}")
            print(f"  Potential Improvement: {improvement['potential_improvement']:.2f}%")
    else:
        print("Error:", portfolio_analysis.get("error", "Unknown error"))
    
    # Get portfolio rebalance recommendations
    print("\nGetting portfolio rebalance recommendations...")
    rebalance = get_portfolio_rebalance(current_tickers, current_weights, optimize=True)
    
    if "error" not in rebalance:
        print("\nRecommended Trades:")
        for trade in rebalance["trades"]:
            print(f"  {trade['action']} {trade['ticker']}: {trade['current_weight']:.2%} -> {trade['target_weight']:.2%}")
        
        print(f"\nPortfolio Turnover: {rebalance['portfolio_turnover']:.2f}%")
        
        if rebalance.get("improvement_metrics"):
            metrics = rebalance["improvement_metrics"]
            print("\nImprovement Metrics:")
            print(f"  Expected Return Change: {metrics['expected_return_change']:.2f}%")
            print(f"  Volatility Change: {metrics['volatility_change']:.2f}%")
            print(f"  Sharpe Ratio Change: {metrics['sharpe_ratio_change']:.2f}")
    else:
        print("Error:", rebalance.get("error", "Unknown error"))
    
    # Perform stress test
    print("\nPerforming stress test...")
    stress_test = stress_test_portfolio(current_tickers, current_weights, scenarios=["financial_crisis_2008", "covid_crash_2020"])
    
    if "error" not in stress_test:
        print("\nStress Test Results:")
        print(f"  Scenarios Tested: {stress_test['scenarios_tested']}")
        
        if stress_test.get("summary"):
            summary = stress_test["summary"]
            print(f"  Average Return: {summary['average_return']:.2f}%")
            print(f"  Average Max Drawdown: {summary['average_max_drawdown']:.2f}%")
            print(f"  Worst Scenario: {summary['worst_scenario']['name']} ({summary['worst_scenario']['return']:.2f}%)")
            print(f"  Best Scenario: {summary['best_scenario']['name']} ({summary['best_scenario']['return']:.2f}%)")
        
        print("\nScenario Details:")
        for scenario, result in stress_test.get("scenario_results", {}).items():
            if "error" not in result:
                print(f"  {scenario}: {result['description']}")
                print(f"    Period: {result['period']}")
                print(f"    Return: {result['cumulative_return']:.2f}%")
                print(f"    Max Drawdown: {result['max_drawdown']:.2f}%")
                print(f"    Worst Day: {result['worst_day_return']:.2f}% on {result['worst_day_date']}")
    else:
        print("Error:", stress_test.get("error", "Unknown error"))

def test_user_profile():
    """Test user profile features"""
    print_section("User Profile Management")
    
    # Create a user profile
    print("Creating user profile...")
    user = create_user_profile("John Doe", "john.doe@example.com")
    
    if "error" not in user:
        user_id = user["user_id"]
        print(f"User created with ID: {user_id}")
        
        # Update user preferences
        print("\nUpdating user preferences...")
        preferences = {
            "investment_profile": {
                "risk_tolerance": "aggressive",
                "investment_horizon": "long_term",
                "investment_experience": "advanced"
            },
            "notification_preferences": {
                "email_alerts": True,
                "price_alerts": True,
                "portfolio_updates": "daily"
            }
        }
        
        update_result = update_user_preferences(user_id, preferences)
        if "error" not in update_result:
            print(f"Updated {update_result['updated_count']} preferences")
        else:
            print("Error updating preferences:", update_result.get("error"))
        
        # Add investment goals
        print("\nAdding investment goals...")
        
        # Retirement goal
        retirement_date = (datetime.now() + timedelta(days=365*30)).strftime('%Y-%m-%d')  # 30 years from now
        retirement_goal = add_investment_goal(
            user_id,
            "Retirement",
            2000000.0,  # $2M target
            retirement_date,
            current_amount=100000.0,
            priority=1
        )
        
        if "error" not in retirement_goal:
            print(f"Added retirement goal (ID: {retirement_goal['goal_id']})")
        else:
            print("Error adding retirement goal:", retirement_goal.get("error"))
        
        # House down payment goal
        house_date = (datetime.now() + timedelta(days=365*5)).strftime('%Y-%m-%d')  # 5 years from now
        house_goal = add_investment_goal(
            user_id,
            "House Down Payment",
            100000.0,  # $100K target
            house_date,
            current_amount=20000.0,
            priority=2
        )
        
        if "error" not in house_goal:
            print(f"Added house down payment goal (ID: {house_goal['goal_id']})")
        else:
            print("Error adding house goal:", house_goal.get("error"))
        
        # Get personalized recommendations
        print("\nGetting personalized recommendations...")
        recommendations = get_personalized_recommendations(user_id)
        
        if "error" not in recommendations:
            print("\nRecommended Asset Allocation:")
            for asset, allocation in recommendations["asset_allocation"].items():
                print(f"  {asset.title()}: {allocation}%")
            
            print("\nInvestment Strategy:")
            strategy = recommendations["investment_strategy"]
            print(f"  Approach: {strategy['approach'].replace('_', ' ').title()}")
            print(f"  Rebalancing: {strategy['rebalancing_frequency'].replace('_', ' ').title()}")
            print(f"  Dollar Cost Averaging: {'Yes' if strategy['dollar_cost_averaging'] else 'No'}")
            
            print("\nConsiderations:")
            for i, consideration in enumerate(strategy["considerations"], 1):
                print(f"  {i}. {consideration}")
            
            print("\nSuggested ETFs:")
            etfs = recommendations["suggested_etfs"]
            print("  Stocks:")
            for category, symbols in etfs["stocks"].items():
                if symbols:
                    print(f"    {category.replace('_', ' ').title()}: {', '.join(symbols)}")
            
            print("  Bonds:")
            for category, symbols in etfs["bonds"].items():
                if symbols:
                    print(f"    {category.replace('_', ' ').title()}: {', '.join(symbols)}")
        else:
            print("Error getting recommendations:", recommendations.get("error"))
        
        # Get user profile
        print("\nGetting user profile...")
        profile = get_user_profile(user_id)
        
        if "error" not in profile:
            print(f"User: {profile['name']} ({profile['email']})")
            print(f"Created: {profile['created_at']}")
            print(f"Last Updated: {profile['last_updated']}")
            
            print("\nInvestment Goals:")
            for goal in profile.get("investment_goals", []):
                progress = goal["current_amount"] / goal["target_amount"] * 100
                print(f"  {goal['name']}: ${goal['current_amount']:,.2f} / ${goal['target_amount']:,.2f} ({progress:.1f}%)")
                print(f"    Target Date: {goal['target_date']}")
                print(f"    Priority: {goal['priority']}")
        else:
            print("Error getting profile:", profile.get("error"))
    else:
        print("Error creating user:", user.get("error"))

def main():
    """Main function"""
    print_section("Portfolio and User Profile Test")
    
    # Test portfolio optimization
    test_portfolio_optimization()
    
    # Test user profile
    test_user_profile()

if __name__ == "__main__":
    main()
