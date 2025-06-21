# tools/user_profile.py

import os
import json
import sqlite3
from typing import Dict, List, Union, Optional, Any
from datetime import datetime
import logging
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory for user profiles
PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../profiles")
os.makedirs(PROFILE_DIR, exist_ok=True)

# SQLite database for user profiles
DB_PATH = os.path.join(PROFILE_DIR, "user_profiles.db")

class UserProfileManager:
    """User profile management class"""
    
    def __init__(self):
        """Initialize the user profile manager"""
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Create users table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                email TEXT,
                created_at TEXT,
                last_updated TEXT
            )
            ''')
            
            # Create preferences table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS preferences (
                preference_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                category TEXT,
                key TEXT,
                value TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id),
                UNIQUE(user_id, category, key)
            )
            ''')
            
            # Create investment goals table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS investment_goals (
                goal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                name TEXT,
                target_amount REAL,
                current_amount REAL,
                target_date TEXT,
                priority INTEGER,
                created_at TEXT,
                last_updated TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
            ''')
            
            # Create portfolios table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolios (
                portfolio_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                name TEXT,
                description TEXT,
                created_at TEXT,
                last_updated TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
            ''')
            
            # Create portfolio holdings table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_holdings (
                holding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER,
                ticker TEXT,
                shares REAL,
                purchase_price REAL,
                purchase_date TEXT,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios (portfolio_id)
            )
            ''')
            
            # Create alerts table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                ticker TEXT,
                alert_type TEXT,
                threshold REAL,
                is_active INTEGER,
                created_at TEXT,
                triggered_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def create_user(self, name: str, email: str = None) -> Dict:
        """
        Create a new user profile
        
        Args:
            name: User's name
            email: User's email (optional)
            
        Returns:
            Dictionary with user information
        """
        try:
            # Generate a unique user ID
            user_id = str(uuid.uuid4())
            
            # Get current timestamp
            timestamp = datetime.now().isoformat()
            
            # Connect to database
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Insert user
            cursor.execute(
                "INSERT INTO users (user_id, name, email, created_at, last_updated) VALUES (?, ?, ?, ?, ?)",
                (user_id, name, email, timestamp, timestamp)
            )
            
            # Set default preferences
            default_preferences = {
                "investment_profile": {
                    "risk_tolerance": "moderate",
                    "investment_horizon": "long_term",
                    "income_level": "medium",
                    "tax_bracket": "medium",
                    "investment_experience": "intermediate"
                },
                "notification_preferences": {
                    "email_alerts": True,
                    "price_alerts": True,
                    "news_alerts": True,
                    "portfolio_updates": "weekly"
                },
                "display_preferences": {
                    "default_chart_period": "1y",
                    "default_chart_type": "line",
                    "show_beta": True,
                    "show_volatility": True,
                    "currency": "USD"
                }
            }
            
            # Insert default preferences
            for category, prefs in default_preferences.items():
                for key, value in prefs.items():
                    cursor.execute(
                        "INSERT INTO preferences (user_id, category, key, value) VALUES (?, ?, ?, ?)",
                        (user_id, category, key, json.dumps(value))
                    )
            
            conn.commit()
            conn.close()
            
            return {
                "user_id": user_id,
                "name": name,
                "email": email,
                "created_at": timestamp,
                "message": "User profile created successfully"
            }
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return {"error": f"Error creating user: {str(e)}"}
    
    def get_user(self, user_id: str) -> Dict:
        """
        Get user profile information
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with user information
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get user information
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            user = cursor.fetchone()
            
            if not user:
                return {"error": f"User not found: {user_id}"}
            
            # Convert to dictionary
            user_dict = dict(user)
            
            # Get user preferences
            cursor.execute("SELECT category, key, value FROM preferences WHERE user_id = ?", (user_id,))
            preferences = cursor.fetchall()
            
            # Organize preferences by category
            user_dict["preferences"] = {}
            for pref in preferences:
                category = pref["category"]
                key = pref["key"]
                value = json.loads(pref["value"])
                
                if category not in user_dict["preferences"]:
                    user_dict["preferences"][category] = {}
                
                user_dict["preferences"][category][key] = value
            
            # Get investment goals
            cursor.execute("SELECT * FROM investment_goals WHERE user_id = ?", (user_id,))
            goals = cursor.fetchall()
            user_dict["investment_goals"] = [dict(goal) for goal in goals]
            
            # Get portfolios
            cursor.execute("SELECT * FROM portfolios WHERE user_id = ?", (user_id,))
            portfolios = cursor.fetchall()
            
            user_dict["portfolios"] = []
            for portfolio in portfolios:
                portfolio_dict = dict(portfolio)
                portfolio_id = portfolio_dict["portfolio_id"]
                
                # Get holdings for this portfolio
                cursor.execute("SELECT * FROM portfolio_holdings WHERE portfolio_id = ?", (portfolio_id,))
                holdings = cursor.fetchall()
                portfolio_dict["holdings"] = [dict(holding) for holding in holdings]
                
                user_dict["portfolios"].append(portfolio_dict)
            
            # Get alerts
            cursor.execute("SELECT * FROM alerts WHERE user_id = ?", (user_id,))
            alerts = cursor.fetchall()
            user_dict["alerts"] = [dict(alert) for alert in alerts]
            
            conn.close()
            
            return user_dict
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return {"error": f"Error getting user: {str(e)}"}
    
    def update_preferences(self, user_id: str, preferences: Dict) -> Dict:
        """
        Update user preferences
        
        Args:
            user_id: User ID
            preferences: Dictionary of preferences to update
            
        Returns:
            Dictionary with update status
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Check if user exists
            cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
            if not cursor.fetchone():
                return {"error": f"User not found: {user_id}"}
            
            # Update last_updated timestamp
            timestamp = datetime.now().isoformat()
            cursor.execute(
                "UPDATE users SET last_updated = ? WHERE user_id = ?",
                (timestamp, user_id)
            )
            
            # Update preferences
            updated_count = 0
            for category, prefs in preferences.items():
                for key, value in prefs.items():
                    # Check if preference exists
                    cursor.execute(
                        "SELECT preference_id FROM preferences WHERE user_id = ? AND category = ? AND key = ?",
                        (user_id, category, key)
                    )
                    if cursor.fetchone():
                        # Update existing preference
                        cursor.execute(
                            "UPDATE preferences SET value = ? WHERE user_id = ? AND category = ? AND key = ?",
                            (json.dumps(value), user_id, category, key)
                        )
                    else:
                        # Insert new preference
                        cursor.execute(
                            "INSERT INTO preferences (user_id, category, key, value) VALUES (?, ?, ?, ?)",
                            (user_id, category, key, json.dumps(value))
                        )
                    updated_count += 1
            
            conn.commit()
            conn.close()
            
            return {
                "user_id": user_id,
                "updated_count": updated_count,
                "last_updated": timestamp,
                "message": "Preferences updated successfully"
            }
        except Exception as e:
            logger.error(f"Error updating preferences: {e}")
            return {"error": f"Error updating preferences: {str(e)}"}
    
    def add_investment_goal(self, user_id: str, name: str, target_amount: float, 
                           target_date: str, current_amount: float = 0.0, 
                           priority: int = 1) -> Dict:
        """
        Add an investment goal for a user
        
        Args:
            user_id: User ID
            name: Goal name
            target_amount: Target amount to reach
            target_date: Target date to reach the goal (YYYY-MM-DD)
            current_amount: Current amount saved (default: 0)
            priority: Goal priority (1-5, 1 being highest)
            
        Returns:
            Dictionary with goal information
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Check if user exists
            cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
            if not cursor.fetchone():
                return {"error": f"User not found: {user_id}"}
            
            # Get current timestamp
            timestamp = datetime.now().isoformat()
            
            # Insert goal
            cursor.execute(
                """INSERT INTO investment_goals 
                   (user_id, name, target_amount, current_amount, target_date, priority, created_at, last_updated) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, name, target_amount, current_amount, target_date, priority, timestamp, timestamp)
            )
            
            # Get the goal ID
            goal_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            return {
                "goal_id": goal_id,
                "user_id": user_id,
                "name": name,
                "target_amount": target_amount,
                "current_amount": current_amount,
                "target_date": target_date,
                "priority": priority,
                "created_at": timestamp,
                "message": "Investment goal added successfully"
            }
        except Exception as e:
            logger.error(f"Error adding investment goal: {e}")
            return {"error": f"Error adding investment goal: {str(e)}"}
    
    def update_investment_goal(self, goal_id: int, updates: Dict) -> Dict:
        """
        Update an investment goal
        
        Args:
            goal_id: Goal ID
            updates: Dictionary of fields to update
            
        Returns:
            Dictionary with update status
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Check if goal exists
            cursor.execute("SELECT * FROM investment_goals WHERE goal_id = ?", (goal_id,))
            goal = cursor.fetchone()
            
            if not goal:
                return {"error": f"Investment goal not found: {goal_id}"}
            
            # Get current timestamp
            timestamp = datetime.now().isoformat()
            
            # Build update query
            valid_fields = ["name", "target_amount", "current_amount", "target_date", "priority"]
            update_fields = []
            update_values = []
            
            for field, value in updates.items():
                if field in valid_fields:
                    update_fields.append(f"{field} = ?")
                    update_values.append(value)
            
            if not update_fields:
                return {"error": "No valid fields to update"}
            
            # Add last_updated field
            update_fields.append("last_updated = ?")
            update_values.append(timestamp)
            
            # Add goal_id for WHERE clause
            update_values.append(goal_id)
            
            # Execute update
            cursor.execute(
                f"UPDATE investment_goals SET {', '.join(update_fields)} WHERE goal_id = ?",
                tuple(update_values)
            )
            
            conn.commit()
            
            # Get updated goal
            cursor.execute("SELECT * FROM investment_goals WHERE goal_id = ?", (goal_id,))
            updated_goal = cursor.fetchone()
            
            conn.close()
            
            return {
                "goal_id": goal_id,
                "updated_fields": list(updates.keys()),
                "last_updated": timestamp,
                "goal": dict(updated_goal),
                "message": "Investment goal updated successfully"
            }
        except Exception as e:
            logger.error(f"Error updating investment goal: {e}")
            return {"error": f"Error updating investment goal: {str(e)}"}
    
    def delete_investment_goal(self, goal_id: int) -> Dict:
        """
        Delete an investment goal
        
        Args:
            goal_id: Goal ID
            
        Returns:
            Dictionary with deletion status
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Check if goal exists
            cursor.execute("SELECT goal_id FROM investment_goals WHERE goal_id = ?", (goal_id,))
            if not cursor.fetchone():
                return {"error": f"Investment goal not found: {goal_id}"}
            
            # Delete goal
            cursor.execute("DELETE FROM investment_goals WHERE goal_id = ?", (goal_id,))
            
            conn.commit()
            conn.close()
            
            return {
                "goal_id": goal_id,
                "message": "Investment goal deleted successfully"
            }
        except Exception as e:
            logger.error(f"Error deleting investment goal: {e}")
            return {"error": f"Error deleting investment goal: {str(e)}"}
    
    def add_portfolio(self, user_id: str, name: str, description: str = None) -> Dict:
        """
        Add a portfolio for a user
        
        Args:
            user_id: User ID
            name: Portfolio name
            description: Portfolio description (optional)
            
        Returns:
            Dictionary with portfolio information
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Check if user exists
            cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
            if not cursor.fetchone():
                return {"error": f"User not found: {user_id}"}
            
            # Get current timestamp
            timestamp = datetime.now().isoformat()
            
            # Insert portfolio
            cursor.execute(
                "INSERT INTO portfolios (user_id, name, description, created_at, last_updated) VALUES (?, ?, ?, ?, ?)",
                (user_id, name, description, timestamp, timestamp)
            )
            
            # Get the portfolio ID
            portfolio_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            return {
                "portfolio_id": portfolio_id,
                "user_id": user_id,
                "name": name,
                "description": description,
                "created_at": timestamp,
                "message": "Portfolio added successfully"
            }
        except Exception as e:
            logger.error(f"Error adding portfolio: {e}")
            return {"error": f"Error adding portfolio: {str(e)}"}
    
    def add_holding(self, portfolio_id: int, ticker: str, shares: float, 
                   purchase_price: float, purchase_date: str = None) -> Dict:
        """
        Add a holding to a portfolio
        
        Args:
            portfolio_id: Portfolio ID
            ticker: Stock symbol
            shares: Number of shares
            purchase_price: Purchase price per share
            purchase_date: Purchase date (YYYY-MM-DD, default: today)
            
        Returns:
            Dictionary with holding information
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Check if portfolio exists
            cursor.execute("SELECT portfolio_id FROM portfolios WHERE portfolio_id = ?", (portfolio_id,))
            if not cursor.fetchone():
                return {"error": f"Portfolio not found: {portfolio_id}"}
            
            # Set purchase date to today if not provided
            if not purchase_date:
                purchase_date = datetime.now().strftime('%Y-%m-%d')
            
            # Insert holding
            cursor.execute(
                "INSERT INTO portfolio_holdings (portfolio_id, ticker, shares, purchase_price, purchase_date) VALUES (?, ?, ?, ?, ?)",
                (portfolio_id, ticker.upper(), shares, purchase_price, purchase_date)
            )
            
            # Get the holding ID
            holding_id = cursor.lastrowid
            
            # Update portfolio last_updated timestamp
            timestamp = datetime.now().isoformat()
            cursor.execute(
                "UPDATE portfolios SET last_updated = ? WHERE portfolio_id = ?",
                (timestamp, portfolio_id)
            )
            
            conn.commit()
            conn.close()
            
            return {
                "holding_id": holding_id,
                "portfolio_id": portfolio_id,
                "ticker": ticker.upper(),
                "shares": shares,
                "purchase_price": purchase_price,
                "purchase_date": purchase_date,
                "message": "Holding added successfully"
            }
        except Exception as e:
            logger.error(f"Error adding holding: {e}")
            return {"error": f"Error adding holding: {str(e)}"}
    
    def update_holding(self, holding_id: int, updates: Dict) -> Dict:
        """
        Update a portfolio holding
        
        Args:
            holding_id: Holding ID
            updates: Dictionary of fields to update
            
        Returns:
            Dictionary with update status
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Check if holding exists
            cursor.execute("SELECT * FROM portfolio_holdings WHERE holding_id = ?", (holding_id,))
            holding = cursor.fetchone()
            
            if not holding:
                return {"error": f"Holding not found: {holding_id}"}
            
            # Build update query
            valid_fields = ["ticker", "shares", "purchase_price", "purchase_date"]
            update_fields = []
            update_values = []
            
            for field, value in updates.items():
                if field in valid_fields:
                    update_fields.append(f"{field} = ?")
                    # Convert ticker to uppercase
                    if field == "ticker":
                        update_values.append(value.upper())
                    else:
                        update_values.append(value)
            
            if not update_fields:
                return {"error": "No valid fields to update"}
            
            # Add holding_id for WHERE clause
            update_values.append(holding_id)
            
            # Execute update
            cursor.execute(
                f"UPDATE portfolio_holdings SET {', '.join(update_fields)} WHERE holding_id = ?",
                tuple(update_values)
            )
            
            # Update portfolio last_updated timestamp
            timestamp = datetime.now().isoformat()
            portfolio_id = holding["portfolio_id"]
            cursor.execute(
                "UPDATE portfolios SET last_updated = ? WHERE portfolio_id = ?",
                (timestamp, portfolio_id)
            )
            
            conn.commit()
            
            # Get updated holding
            cursor.execute("SELECT * FROM portfolio_holdings WHERE holding_id = ?", (holding_id,))
            updated_holding = cursor.fetchone()
            
            conn.close()
            
            return {
                "holding_id": holding_id,
                "updated_fields": list(updates.keys()),
                "holding": dict(updated_holding),
                "message": "Holding updated successfully"
            }
        except Exception as e:
            logger.error(f"Error updating holding: {e}")
            return {"error": f"Error updating holding: {str(e)}"}
    
    def delete_holding(self, holding_id: int) -> Dict:
        """
        Delete a portfolio holding
        
        Args:
            holding_id: Holding ID
            
        Returns:
            Dictionary with deletion status
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Check if holding exists and get portfolio_id
            cursor.execute("SELECT portfolio_id FROM portfolio_holdings WHERE holding_id = ?", (holding_id,))
            result = cursor.fetchone()
            
            if not result:
                return {"error": f"Holding not found: {holding_id}"}
            
            portfolio_id = result[0]
            
            # Delete holding
            cursor.execute("DELETE FROM portfolio_holdings WHERE holding_id = ?", (holding_id,))
            
            # Update portfolio last_updated timestamp
            timestamp = datetime.now().isoformat()
            cursor.execute(
                "UPDATE portfolios SET last_updated = ? WHERE portfolio_id = ?",
                (timestamp, portfolio_id)
            )
            
            conn.commit()
            conn.close()
            
            return {
                "holding_id": holding_id,
                "portfolio_id": portfolio_id,
                "message": "Holding deleted successfully"
            }
        except Exception as e:
            logger.error(f"Error deleting holding: {e}")
            return {"error": f"Error deleting holding: {str(e)}"}
    
    def add_alert(self, user_id: str, ticker: str, alert_type: str, threshold: float) -> Dict:
        """
        Add a price alert for a user
        
        Args:
            user_id: User ID
            ticker: Stock symbol
            alert_type: Type of alert (above, below)
            threshold: Price threshold
            
        Returns:
            Dictionary with alert information
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Check if user exists
            cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
            if not cursor.fetchone():
                return {"error": f"User not found: {user_id}"}
            
            # Validate alert type
            valid_alert_types = ["above", "below"]
            if alert_type not in valid_alert_types:
                return {"error": f"Invalid alert type: {alert_type}. Must be one of {valid_alert_types}"}
            
            # Get current timestamp
            timestamp = datetime.now().isoformat()
            
            # Insert alert
            cursor.execute(
                "INSERT INTO alerts (user_id, ticker, alert_type, threshold, is_active, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, ticker.upper(), alert_type, threshold, 1, timestamp)
            )
            
            # Get the alert ID
            alert_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            return {
                "alert_id": alert_id,
                "user_id": user_id,
                "ticker": ticker.upper(),
                "alert_type": alert_type,
                "threshold": threshold,
                "is_active": True,
                "created_at": timestamp,
                "message": "Alert added successfully"
            }
        except Exception as e:
            logger.error(f"Error adding alert: {e}")
            return {"error": f"Error adding alert: {str(e)}"}
    
    def update_alert(self, alert_id: int, updates: Dict) -> Dict:
        """
        Update a price alert
        
        Args:
            alert_id: Alert ID
            updates: Dictionary of fields to update
            
        Returns:
            Dictionary with update status
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Check if alert exists
            cursor.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,))
            alert = cursor.fetchone()
            
            if not alert:
                return {"error": f"Alert not found: {alert_id}"}
            
            # Build update query
            valid_fields = ["ticker", "alert_type", "threshold", "is_active"]
            update_fields = []
            update_values = []
            
            for field, value in updates.items():
                if field in valid_fields:
                    # Validate alert type
                    if field == "alert_type":
                        valid_alert_types = ["above", "below"]
                        if value not in valid_alert_types:
                            return {"error": f"Invalid alert type: {value}. Must be one of {valid_alert_types}"}
                    
                    update_fields.append(f"{field} = ?")
                    # Convert ticker to uppercase
                    if field == "ticker":
                        update_values.append(value.upper())
                    else:
                        update_values.append(value)
            
            if not update_fields:
                return {"error": "No valid fields to update"}
            
            # Add alert_id for WHERE clause
            update_values.append(alert_id)
            
            # Execute update
            cursor.execute(
                f"UPDATE alerts SET {', '.join(update_fields)} WHERE alert_id = ?",
                tuple(update_values)
            )
            
            conn.commit()
            
            # Get updated alert
            cursor.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,))
            updated_alert = cursor.fetchone()
            
            conn.close()
            
            return {
                "alert_id": alert_id,
                "updated_fields": list(updates.keys()),
                "alert": dict(updated_alert),
                "message": "Alert updated successfully"
            }
        except Exception as e:
            logger.error(f"Error updating alert: {e}")
            return {"error": f"Error updating alert: {str(e)}"}
    
    def delete_alert(self, alert_id: int) -> Dict:
        """
        Delete a price alert
        
        Args:
            alert_id: Alert ID
            
        Returns:
            Dictionary with deletion status
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Check if alert exists
            cursor.execute("SELECT alert_id FROM alerts WHERE alert_id = ?", (alert_id,))
            if not cursor.fetchone():
                return {"error": f"Alert not found: {alert_id}"}
            
            # Delete alert
            cursor.execute("DELETE FROM alerts WHERE alert_id = ?", (alert_id,))
            
            conn.commit()
            conn.close()
            
            return {
                "alert_id": alert_id,
                "message": "Alert deleted successfully"
            }
        except Exception as e:
            logger.error(f"Error deleting alert: {e}")
            return {"error": f"Error deleting alert: {str(e)}"}
    
    def get_investment_recommendations(self, user_id: str) -> Dict:
        """
        Get personalized investment recommendations based on user profile
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with investment recommendations
        """
        try:
            # Get user profile
            user = self.get_user(user_id)
            
            if "error" in user:
                return user
            
            # Extract preferences
            preferences = user.get("preferences", {})
            investment_profile = preferences.get("investment_profile", {})
            
            # Get risk tolerance
            risk_tolerance = investment_profile.get("risk_tolerance", "moderate")
            
            # Get investment horizon
            investment_horizon = investment_profile.get("investment_horizon", "long_term")
            
            # Get investment experience
            investment_experience = investment_profile.get("investment_experience", "intermediate")
            
            # Generate recommendations based on profile
            recommendations = {
                "risk_tolerance": risk_tolerance,
                "investment_horizon": investment_horizon,
                "investment_experience": investment_experience,
                "asset_allocation": self._get_asset_allocation(risk_tolerance, investment_horizon),
                "suggested_etfs": self._get_suggested_etfs(risk_tolerance, investment_horizon),
                "investment_strategy": self._get_investment_strategy(risk_tolerance, investment_horizon, investment_experience)
            }
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting investment recommendations: {e}")
            return {"error": f"Error getting investment recommendations: {str(e)}"}
    
    def _get_asset_allocation(self, risk_tolerance: str, investment_horizon: str) -> Dict:
        """
        Get recommended asset allocation based on risk tolerance and investment horizon
        
        Args:
            risk_tolerance: Risk tolerance (conservative, moderate, aggressive)
            investment_horizon: Investment horizon (short_term, medium_term, long_term)
            
        Returns:
            Dictionary with asset allocation percentages
        """
        # Default allocation (moderate risk, long-term horizon)
        allocation = {
            "stocks": 60,
            "bonds": 30,
            "cash": 5,
            "alternatives": 5
        }
        
        # Adjust based on risk tolerance
        if risk_tolerance == "conservative":
            allocation["stocks"] -= 20
            allocation["bonds"] += 15
            allocation["cash"] += 5
        elif risk_tolerance == "aggressive":
            allocation["stocks"] += 20
            allocation["bonds"] -= 15
            allocation["alternatives"] += 5
            allocation["cash"] -= 10
        
        # Adjust based on investment horizon
        if investment_horizon == "short_term":
            allocation["stocks"] -= 20
            allocation["bonds"] -= 10
            allocation["cash"] += 30
        elif investment_horizon == "medium_term":
            allocation["stocks"] -= 10
            allocation["bonds"] += 5
            allocation["cash"] += 5
        
        # Ensure no negative allocations
        for asset in allocation:
            allocation[asset] = max(0, allocation[asset])
        
        # Normalize to ensure sum is 100%
        total = sum(allocation.values())
        for asset in allocation:
            allocation[asset] = round(allocation[asset] / total * 100)
        
        return allocation
    
    def _get_suggested_etfs(self, risk_tolerance: str, investment_horizon: str) -> Dict:
        """
        Get suggested ETFs based on risk tolerance and investment horizon
        
        Args:
            risk_tolerance: Risk tolerance (conservative, moderate, aggressive)
            investment_horizon: Investment horizon (short_term, medium_term, long_term)
            
        Returns:
            Dictionary with suggested ETFs by category
        """
        # Default ETF suggestions
        etfs = {
            "stocks": {
                "us_large_cap": ["VTI", "SPY", "IVV"],
                "us_mid_cap": ["VO", "IJH"],
                "us_small_cap": ["VB", "IJR"],
                "international_developed": ["VEA", "EFA"],
                "international_emerging": ["VWO", "IEMG"],
                "sector": []
            },
            "bonds": {
                "us_aggregate": ["AGG", "BND"],
                "us_treasury": ["IEI", "GOVT"],
                "us_corporate": ["LQD", "VCIT"],
                "international": ["BNDX"],
                "high_yield": []
            },
            "alternatives": {
                "real_estate": ["VNQ", "IYR"],
                "commodities": ["GLD", "IAU"],
                "other": []
            },
            "cash_equivalents": ["SHV", "BIL"]
        }
        
        # Adjust based on risk tolerance
        if risk_tolerance == "conservative":
            etfs["bonds"]["us_treasury"].append("SHY")  # Short-term Treasury
            etfs["stocks"]["sector"] = ["VPU", "XLP"]  # Utilities, Consumer Staples
            etfs["bonds"]["high_yield"] = []  # No high yield for conservative
        elif risk_tolerance == "moderate":
            etfs["bonds"]["high_yield"] = ["HYG"]  # Some high yield
            etfs["stocks"]["sector"] = ["VPU", "XLP", "XLV"]  # Add Healthcare
        elif risk_tolerance == "aggressive":
            etfs["bonds"]["high_yield"] = ["HYG", "JNK"]  # More high yield
            etfs["stocks"]["sector"] = ["XLK", "QQQ", "XLF", "XLI"]  # Tech, Financials, Industrials
            etfs["alternatives"]["other"] = ["ARKK"]  # Innovation ETF
        
        # Adjust based on investment horizon
        if investment_horizon == "short_term":
            etfs["bonds"]["us_treasury"] = ["SHY", "BIL"]  # Short-term only
            etfs["bonds"]["us_corporate"] = ["VCSH"]  # Short-term corporate
            etfs["alternatives"]["other"] = []  # No speculative alternatives
        elif investment_horizon == "medium_term":
            etfs["bonds"]["us_treasury"].append("IEF")  # Add intermediate-term
        elif investment_horizon == "long_term":
            etfs["stocks"]["us_small_cap"].append("VBR")  # Add small-cap value
            etfs["alternatives"]["other"].append("SCHD")  # Add dividend ETF
        
        return etfs
    
    def _get_investment_strategy(self, risk_tolerance: str, investment_horizon: str, 
                               investment_experience: str) -> Dict:
        """
        Get investment strategy recommendations based on user profile
        
        Args:
            risk_tolerance: Risk tolerance (conservative, moderate, aggressive)
            investment_horizon: Investment horizon (short_term, medium_term, long_term)
            investment_experience: Investment experience (beginner, intermediate, advanced)
            
        Returns:
            Dictionary with investment strategy recommendations
        """
        # Default strategy
        strategy = {
            "approach": "buy_and_hold",
            "rebalancing_frequency": "annually",
            "dollar_cost_averaging": True,
            "tax_strategy": "tax_efficient_etfs",
            "recommended_accounts": ["401k", "ira", "taxable"],
            "considerations": []
        }
        
        # Adjust based on risk tolerance
        if risk_tolerance == "conservative":
            strategy["approach"] = "income_focused"
            strategy["considerations"].append("Focus on capital preservation and income generation")
            strategy["considerations"].append("Consider bond ladder strategy for stable income")
        elif risk_tolerance == "moderate":
            strategy["approach"] = "balanced_growth"
            strategy["considerations"].append("Balance between growth and income")
            strategy["considerations"].append("Consider dividend growth stocks for income and appreciation")
        elif risk_tolerance == "aggressive":
            strategy["approach"] = "growth_focused"
            strategy["rebalancing_frequency"] = "semi_annually"
            strategy["considerations"].append("Focus on long-term capital appreciation")
            strategy["considerations"].append("Consider overweighting growth sectors like technology")
        
        # Adjust based on investment horizon
        if investment_horizon == "short_term":
            strategy["approach"] = "capital_preservation"
            strategy["rebalancing_frequency"] = "quarterly"
            strategy["dollar_cost_averaging"] = False
            strategy["considerations"].append("Prioritize liquidity and capital preservation")
            strategy["considerations"].append("Avoid investments with high volatility")
        elif investment_horizon == "medium_term":
            strategy["considerations"].append("Balance between short-term needs and long-term growth")
        elif investment_horizon == "long_term":
            strategy["considerations"].append("Take advantage of compound growth over time")
            strategy["considerations"].append("Consider higher equity allocation for better long-term returns")
        
        # Adjust based on investment experience
        if investment_experience == "beginner":
            strategy["approach"] = "simple_diversified"
            strategy["considerations"].append("Start with broad market index ETFs")
            strategy["considerations"].append("Focus on education and building good investment habits")
        elif investment_experience == "intermediate":
            strategy["considerations"].append("Consider factor-based investing strategies")
            strategy["considerations"].append("Explore tax-loss harvesting opportunities")
        elif investment_experience == "advanced":
            strategy["considerations"].append("Consider more sophisticated asset allocation strategies")
            strategy["considerations"].append("Explore alternative investments for diversification")
        
        return strategy

# Helper functions for user profile management
def create_user_profile(name: str, email: str = None) -> Dict:
    """
    Create a new user profile
    
    Args:
        name: User's name
        email: User's email (optional)
        
    Returns:
        Dictionary with user information
    """
    try:
        profile_manager = UserProfileManager()
        result = profile_manager.create_user(name, email)
        return result
    except Exception as e:
        logger.error(f"Error creating user profile: {e}")
        return {"error": f"Error creating user profile: {str(e)}"}

def get_user_profile(user_id: str) -> Dict:
    """
    Get user profile information
    
    Args:
        user_id: User ID
        
    Returns:
        Dictionary with user information
    """
    try:
        profile_manager = UserProfileManager()
        result = profile_manager.get_user(user_id)
        return result
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        return {"error": f"Error getting user profile: {str(e)}"}

def update_user_preferences(user_id: str, preferences: Dict) -> Dict:
    """
    Update user preferences
    
    Args:
        user_id: User ID
        preferences: Dictionary of preferences to update
        
    Returns:
        Dictionary with update status
    """
    try:
        profile_manager = UserProfileManager()
        result = profile_manager.update_preferences(user_id, preferences)
        return result
    except Exception as e:
        logger.error(f"Error updating user preferences: {e}")
        return {"error": f"Error updating user preferences: {str(e)}"}

def add_investment_goal(user_id: str, name: str, target_amount: float, 
                       target_date: str, current_amount: float = 0.0, 
                       priority: int = 1) -> Dict:
    """
    Add an investment goal for a user
    
    Args:
        user_id: User ID
        name: Goal name
        target_amount: Target amount to reach
        target_date: Target date to reach the goal (YYYY-MM-DD)
        current_amount: Current amount saved (default: 0)
        priority: Goal priority (1-5, 1 being highest)
        
    Returns:
        Dictionary with goal information
    """
    try:
        profile_manager = UserProfileManager()
        result = profile_manager.add_investment_goal(
            user_id, name, target_amount, target_date, current_amount, priority
        )
        return result
    except Exception as e:
        logger.error(f"Error adding investment goal: {e}")
        return {"error": f"Error adding investment goal: {str(e)}"}

def get_personalized_recommendations(user_id: str) -> Dict:
    """
    Get personalized investment recommendations based on user profile
    
    Args:
        user_id: User ID
        
    Returns:
        Dictionary with investment recommendations
    """
    try:
        profile_manager = UserProfileManager()
        result = profile_manager.get_investment_recommendations(user_id)
        return result
    except Exception as e:
        logger.error(f"Error getting personalized recommendations: {e}")
        return {"error": f"Error getting personalized recommendations: {str(e)}"}
