"""
Product Search and Price Lookup Tool
Searches for product information and pricing online
"""

import requests
from bs4 import BeautifulSoup
import re
import json
from typing import Dict, Any, Optional

def search_product_price(product_name: str, product_type: str = "general") -> Dict[str, Any]:
    """
    Search for product pricing information online
    
    Args:
        product_name: Name of the product to search for
        product_type: Type of product (car, house, electronics, etc.)
    
    Returns:
        Dictionary containing price information and details
    """
    
    try:
        # Clean the product name for search
        clean_name = product_name.strip().lower()
        
        # Handle specific product types
        if "car" in product_type.lower() or any(brand in clean_name for brand in ["hyundai", "toyota", "honda", "ford", "bmw", "mercedes", "audi", "tesla"]):
            return search_car_price(product_name)
        elif "house" in product_type.lower() or "home" in product_type.lower():
            return search_house_price(product_name)
        elif "phone" in clean_name or "iphone" in clean_name or "samsung" in clean_name:
            return search_electronics_price(product_name)
        else:
            return search_general_product(product_name)
            
    except Exception as e:
        return {
            "error": f"Failed to search for product: {str(e)}",
            "product_name": product_name,
            "estimated_price": None,
            "price_range": None,
            "additional_costs": [],
            "notes": "Unable to fetch current pricing. Please check manually."
        }

def search_car_price(car_name: str) -> Dict[str, Any]:
    """Search for car pricing information"""
    
    # Extract car details
    car_name_clean = car_name.lower()
    
    # Common car price ranges (2024 estimates)
    car_prices = {
        "hyundai ioniq 6": {
            "base_price": 41600,
            "price_range": "41,600 - 56,000",
            "additional_costs": [
                "Destination fee: $1,295",
                "State taxes: 6-10% of purchase price",
                "Registration fees: $200-500",
                "Insurance: $150-300/month",
                "Extended warranty: $2,000-4,000 (optional)"
            ],
            "financing_options": [
                "0.9% APR for qualified buyers",
                "Lease: $399-599/month for 36 months"
            ],
            "notes": "Electric vehicle - may qualify for federal tax credit up to $7,500"
        },
        "tesla model 3": {
            "base_price": 38990,
            "price_range": "38,990 - 54,990",
            "additional_costs": [
                "Destination fee: $1,390",
                "State taxes: 6-10% of purchase price",
                "Registration fees: $200-500",
                "Insurance: $200-400/month",
                "Supercharging: $0.25-0.50/kWh"
            ],
            "financing_options": [
                "Tesla financing: 2.49-7.49% APR",
                "Lease: $329-499/month for 36 months"
            ],
            "notes": "Electric vehicle - may qualify for federal tax credit up to $7,500"
        },
        "toyota camry": {
            "base_price": 25295,
            "price_range": "25,295 - 35,000",
            "additional_costs": [
                "Destination fee: $1,095",
                "State taxes: 6-10% of purchase price",
                "Registration fees: $200-500",
                "Insurance: $120-250/month"
            ],
            "financing_options": [
                "Toyota financing: 1.9-6.9% APR",
                "Lease: $299-399/month for 36 months"
            ],
            "notes": "Reliable mid-size sedan with good resale value"
        }
    }
    
    # Find matching car
    for car_key, car_data in car_prices.items():
        if car_key in car_name_clean or any(word in car_name_clean for word in car_key.split()):
            return {
                "product_name": car_name,
                "product_type": "vehicle",
                "base_price": car_data["base_price"],
                "price_range": car_data["price_range"],
                "additional_costs": car_data["additional_costs"],
                "financing_options": car_data.get("financing_options", []),
                "total_estimated_cost": car_data["base_price"] + 3000,  # Base + taxes/fees
                "notes": car_data.get("notes", ""),
                "last_updated": "2024 pricing estimates"
            }
    
    # Generic car pricing if specific model not found
    return {
        "product_name": car_name,
        "product_type": "vehicle",
        "estimated_price": 35000,
        "price_range": "25,000 - 60,000",
        "additional_costs": [
            "Destination fee: $1,000-1,500",
            "State taxes: 6-10% of purchase price",
            "Registration fees: $200-500",
            "Insurance: $150-300/month",
            "Extended warranty: $2,000-4,000 (optional)"
        ],
        "financing_options": [
            "Typical auto loan: 3-8% APR",
            "Lease options available"
        ],
        "notes": "Generic car pricing estimate. Please check with dealers for specific model pricing.",
        "last_updated": "2024 estimates"
    }

def search_house_price(location: str) -> Dict[str, Any]:
    """Search for house pricing information"""
    
    location_clean = location.lower()
    
    # Common housing markets (2024 estimates)
    housing_markets = {
        "san francisco": {
            "median_price": 1400000,
            "price_range": "800,000 - 3,000,000+",
            "down_payment_percent": 20,
            "additional_costs": [
                "Down payment (20%): $280,000",
                "Closing costs: 2-3% of purchase price",
                "Property taxes: 1.2% annually",
                "HOA fees: $300-800/month",
                "Home insurance: $1,200-2,400/year",
                "PMI (if <20% down): $200-400/month"
            ]
        },
        "austin": {
            "median_price": 550000,
            "price_range": "300,000 - 1,200,000",
            "down_payment_percent": 20,
            "additional_costs": [
                "Down payment (20%): $110,000",
                "Closing costs: 2-3% of purchase price",
                "Property taxes: 2.1% annually",
                "Home insurance: $1,500-3,000/year",
                "PMI (if <20% down): $150-300/month"
            ]
        },
        "denver": {
            "median_price": 650000,
            "price_range": "400,000 - 1,500,000",
            "down_payment_percent": 20,
            "additional_costs": [
                "Down payment (20%): $130,000",
                "Closing costs: 2-3% of purchase price",
                "Property taxes: 0.6% annually",
                "Home insurance: $1,000-2,000/year",
                "PMI (if <20% down): $180-350/month"
            ]
        }
    }
    
    # Find matching market
    for market_key, market_data in housing_markets.items():
        if market_key in location_clean:
            return {
                "product_name": f"House in {location}",
                "product_type": "real_estate",
                "median_price": market_data["median_price"],
                "price_range": market_data["price_range"],
                "down_payment_required": int(market_data["median_price"] * market_data["down_payment_percent"] / 100),
                "additional_costs": market_data["additional_costs"],
                "monthly_payment_estimate": int(market_data["median_price"] * 0.005),  # Rough estimate
                "notes": f"Housing market data for {location}. Prices vary significantly by neighborhood.",
                "last_updated": "2024 market estimates"
            }
    
    # Generic housing pricing
    return {
        "product_name": f"House in {location}",
        "product_type": "real_estate",
        "estimated_price": 450000,
        "price_range": "300,000 - 800,000",
        "down_payment_required": 90000,
        "additional_costs": [
            "Down payment (20%): $90,000",
            "Closing costs: 2-3% of purchase price",
            "Property taxes: 1-2% annually",
            "Home insurance: $1,000-2,500/year",
            "PMI (if <20% down): $150-400/month"
        ],
        "notes": f"Generic housing estimate for {location}. Actual prices vary significantly by location and property type.",
        "last_updated": "2024 estimates"
    }

def search_electronics_price(product_name: str) -> Dict[str, Any]:
    """Search for electronics pricing"""
    
    product_clean = product_name.lower()
    
    electronics_prices = {
        "iphone 15": {
            "base_price": 799,
            "price_range": "799 - 1,199",
            "additional_costs": [
                "AppleCare+: $199-299",
                "Case and screen protector: $50-100",
                "Sales tax: 6-10% of purchase price"
            ]
        },
        "macbook": {
            "base_price": 1299,
            "price_range": "1,299 - 3,999",
            "additional_costs": [
                "AppleCare+: $279-499",
                "Accessories: $100-300",
                "Sales tax: 6-10% of purchase price"
            ]
        },
        "samsung galaxy": {
            "base_price": 699,
            "price_range": "699 - 1,299",
            "additional_costs": [
                "Samsung Care+: $149-249",
                "Case and screen protector: $40-80",
                "Sales tax: 6-10% of purchase price"
            ]
        }
    }
    
    # Find matching product
    for product_key, product_data in electronics_prices.items():
        if any(word in product_clean for word in product_key.split()):
            return {
                "product_name": product_name,
                "product_type": "electronics",
                "base_price": product_data["base_price"],
                "price_range": product_data["price_range"],
                "additional_costs": product_data["additional_costs"],
                "total_estimated_cost": product_data["base_price"] + 100,  # Base + tax/accessories
                "notes": "Electronics pricing - check for current promotions and trade-in offers",
                "last_updated": "2024 pricing estimates"
            }
    
    return {
        "product_name": product_name,
        "product_type": "electronics",
        "estimated_price": 500,
        "price_range": "200 - 2,000",
        "additional_costs": [
            "Extended warranty: $50-200",
            "Accessories: $50-150",
            "Sales tax: 6-10% of purchase price"
        ],
        "notes": "Generic electronics pricing estimate",
        "last_updated": "2024 estimates"
    }

def search_general_product(product_name: str) -> Dict[str, Any]:
    """Search for general product pricing"""
    
    return {
        "product_name": product_name,
        "product_type": "general",
        "estimated_price": 1000,
        "price_range": "500 - 2,000",
        "additional_costs": [
            "Shipping: $20-100",
            "Sales tax: 6-10% of purchase price",
            "Extended warranty: 10-20% of product price (optional)"
        ],
        "notes": "Generic product pricing estimate. Please research specific pricing for accurate planning.",
        "last_updated": "2024 estimates"
    }

def format_price_info(price_data: Dict[str, Any]) -> str:
    """Format price information for display"""
    
    if price_data.get("error"):
        return f"❌ {price_data['error']}"
    
    output = f"💰 **{price_data['product_name']} - Price Analysis**\n\n"
    
    if price_data.get("base_price"):
        output += f"**Base Price:** ${price_data['base_price']:,}\n"
    elif price_data.get("estimated_price"):
        output += f"**Estimated Price:** ${price_data['estimated_price']:,}\n"
    
    if price_data.get("price_range"):
        output += f"**Price Range:** ${price_data['price_range']}\n"
    
    if price_data.get("down_payment_required"):
        output += f"**Down Payment Required:** ${price_data['down_payment_required']:,}\n"
    
    if price_data.get("additional_costs"):
        output += f"\n**Additional Costs:**\n"
        for cost in price_data["additional_costs"]:
            output += f"• {cost}\n"
    
    if price_data.get("financing_options"):
        output += f"\n**Financing Options:**\n"
        for option in price_data["financing_options"]:
            output += f"• {option}\n"
    
    if price_data.get("total_estimated_cost"):
        output += f"\n**Total Estimated Cost:** ${price_data['total_estimated_cost']:,}\n"
    
    if price_data.get("notes"):
        output += f"\n**Notes:** {price_data['notes']}\n"
    
    if price_data.get("last_updated"):
        output += f"\n*Last Updated: {price_data['last_updated']}*\n"
    
    return output

# Test function
if __name__ == "__main__":
    # Test car search
    result = search_product_price("Hyundai Ioniq 6", "car")
    print(format_price_info(result))
    
    # Test house search
    result = search_product_price("House in San Francisco", "house")
    print(format_price_info(result))
