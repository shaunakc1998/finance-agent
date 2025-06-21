# tools/forecast.py

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Union, List, Tuple, Optional
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

# XGBoost is not used in this version to avoid dependency issues
XGBOOST_AVAILABLE = False

# Try to import optional dependencies
try:
    from sklearn.neural_network import MLPRegressor
    NEURAL_NET_AVAILABLE = True
except ImportError:
    NEURAL_NET_AVAILABLE = False

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Directory to store trained models
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models")
os.makedirs(MODEL_DIR, exist_ok=True)

class PriceForecaster:
    """Class to handle price forecasting for stocks and ETFs"""
    
    def __init__(self, ticker: str, model_type: str = "ensemble"):
        """
        Initialize the forecaster for a specific ticker
        
        Args:
            ticker: Stock or ETF symbol
            model_type: Type of model to use ('ensemble', 'random_forest', 'gradient_boost', 
                        'linear', 'ridge', 'neural_net', 'arima')
        """
        self.ticker = ticker.upper()
        self.model_type = model_type
        self.model = None
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.last_trained = None
        self.forecast_days = 5  # Default forecast period
        self.features = []
        self.feature_importance = {}
        self.validation_metrics = {}
        self.model_file = os.path.join(MODEL_DIR, f"{self.ticker}_{self.model_type}_model.pkl")
        self.scaler_file = os.path.join(MODEL_DIR, f"{self.ticker}_scaler.pkl")
        self.metrics_file = os.path.join(MODEL_DIR, f"{self.ticker}_metrics.pkl")
        self.market_regime = "normal"  # Can be "normal", "bullish", "bearish", "volatile"
        
        # Try to load existing model
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load model from disk if it exists"""
        if os.path.exists(self.model_file) and os.path.exists(self.scaler_file):
            try:
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.features = model_data['features']
                    self.last_trained = model_data['trained_date']
                    self.forecast_days = model_data['forecast_days']
                    self.feature_importance = model_data.get('feature_importance', {})
                
                with open(self.scaler_file, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                # Load validation metrics if available
                if os.path.exists(self.metrics_file):
                    with open(self.metrics_file, 'rb') as f:
                        self.validation_metrics = pickle.load(f)
                
                # Check if model is too old (more than 3 days for volatile stocks, 7 for others)
                days_threshold = 3 if self._is_volatile_stock() else 7
                if (datetime.now() - self.last_trained).days > days_threshold:
                    print(f"Model for {self.ticker} is outdated, retraining...")
                    return False
                
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        return False
    
    def _is_volatile_stock(self) -> bool:
        """Check if the stock is volatile based on beta or recent price movements"""
        try:
            stock = yf.Ticker(self.ticker)
            info = stock.info
            
            # Check beta if available
            if 'beta' in info and info['beta'] is not None:
                if info['beta'] > 1.5:  # High beta indicates volatility
                    return True
            
            # Check recent price volatility
            hist = stock.history(period="1mo")
            if not hist.empty:
                daily_returns = hist['Close'].pct_change().dropna()
                volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
                if volatility > 0.4:  # 40% annualized volatility is quite high
                    return True
            
            return False
        except Exception:
            return False  # Default to False if we can't determine
    
    def _save_model(self):
        """Save model to disk"""
        try:
            model_data = {
                'model': self.model,
                'features': self.features,
                'trained_date': self.last_trained,
                'forecast_days': self.forecast_days,
                'feature_importance': self.feature_importance
            }
            
            with open(self.model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            with open(self.scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            with open(self.metrics_file, 'wb') as f:
                pickle.dump(self.validation_metrics, f)
                
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def _get_price_history(self, period: str = "3y") -> pd.DataFrame:
        """Get historical price data and calculate technical indicators"""
        try:
            # Get price history
            stock = yf.Ticker(self.ticker)
            df = stock.history(period=period)
            
            if df.empty:
                raise ValueError(f"No price data found for {self.ticker}")
            
            # Calculate technical indicators
            # RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Moving Averages
            for window in [5, 10, 20, 50, 100, 200]:
                df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
                
            # Exponential Moving Averages
            for window in [5, 10, 20, 50]:
                df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
            
            # MACD
            df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # Price momentum
            for days in [1, 3, 5, 10, 21]:
                df[f'Return_{days}d'] = df['Close'].pct_change(days)
            
            # Volatility
            for window in [5, 10, 21]:
                df[f'Volatility_{window}d'] = df['Return_1d'].rolling(window=window).std() * np.sqrt(252)  # Annualized
            
            # Volume indicators
            df['Volume_Change'] = df['Volume'].pct_change(1)
            df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_10']
            df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            df['BB_Std'] = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
            df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            df['BB_Pct'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # Price relative to moving averages
            for ma in [10, 20, 50, 200]:
                df[f'Price_to_SMA_{ma}'] = df['Close'] / df[f'SMA_{ma}'] - 1
            
            # Market regime features
            df['Bull_Market'] = (df['Close'] > df['SMA_200']).astype(int)
            df['Correction'] = ((df['Close'] / df['Close'].rolling(window=60).max() - 1) < -0.1).astype(int)
            
            # Trend strength
            df['ADX'] = self._calculate_adx(df)
            
            # Target variable - future price change
            df[f'Future_Return_{self.forecast_days}d'] = df['Close'].pct_change(self.forecast_days).shift(-self.forecast_days)
            df[f'Future_Price_{self.forecast_days}d'] = df['Close'].shift(-self.forecast_days)
            
            # Add fundamental data if available
            try:
                self._add_fundamental_features(df, stock)
            except Exception as e:
                print(f"Could not add fundamental features: {e}")
            
            # Add market data
            try:
                self._add_market_features(df)
            except Exception as e:
                print(f"Could not add market features: {e}")
            
            # Drop NaN values
            df = df.dropna()
            
            # Detect market regime
            self._detect_market_regime(df)
            
            return df
        except Exception as e:
            print(f"Error getting price history: {e}")
            return pd.DataFrame()
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX) - a trend strength indicator"""
        try:
            # Calculate True Range
            df['TR'] = np.maximum(
                np.maximum(
                    df['High'] - df['Low'],
                    np.abs(df['High'] - df['Close'].shift(1))
                ),
                np.abs(df['Low'] - df['Close'].shift(1))
            )
            
            # Calculate Directional Movement
            df['DM_plus'] = np.where(
                (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
                np.maximum(df['High'] - df['High'].shift(1), 0),
                0
            )
            
            df['DM_minus'] = np.where(
                (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
                np.maximum(df['Low'].shift(1) - df['Low'], 0),
                0
            )
            
            # Calculate smoothed averages
            df['ATR'] = df['TR'].rolling(window=period).mean()
            df['DI_plus'] = 100 * (df['DM_plus'].rolling(window=period).mean() / df['ATR'])
            df['DI_minus'] = 100 * (df['DM_minus'].rolling(window=period).mean() / df['ATR'])
            
            # Calculate Directional Index
            df['DX'] = 100 * np.abs(df['DI_plus'] - df['DI_minus']) / (df['DI_plus'] + df['DI_minus'])
            
            # Calculate ADX
            adx = df['DX'].rolling(window=period).mean()
            return adx
        except Exception:
            # Return zeros if calculation fails
            return pd.Series(0, index=df.index)
    
    def _add_fundamental_features(self, df: pd.DataFrame, stock: yf.Ticker) -> None:
        """Add fundamental data features if available"""
        try:
            # Get quarterly financials
            financials = stock.quarterly_financials
            if financials is not None and not financials.empty:
                # Get the most recent quarter
                latest_quarter = financials.columns[0]
                
                # Create a date range for the entire price history
                date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
                
                # Create a DataFrame with fundamental data
                fundamental_df = pd.DataFrame(index=date_range)
                
                # Add quarterly financial metrics
                try:
                    revenue = financials.loc['Total Revenue', latest_quarter]
                    fundamental_df['Revenue'] = revenue
                except:
                    pass
                
                try:
                    net_income = financials.loc['Net Income', latest_quarter]
                    fundamental_df['Net_Income'] = net_income
                except:
                    pass
                
                # Resample to match price data frequency and forward fill
                fundamental_df = fundamental_df.resample('D').ffill()
                
                # Merge with price data
                fundamental_df = fundamental_df.reindex(df.index)
                for col in fundamental_df.columns:
                    df[col] = fundamental_df[col]
                
                # Add derived fundamental metrics
                if 'Revenue' in df.columns and 'Net_Income' in df.columns:
                    df['Profit_Margin'] = df['Net_Income'] / df['Revenue']
                
                # Add PE ratio if available
                try:
                    info = stock.info
                    if 'trailingPE' in info and info['trailingPE'] is not None:
                        df['PE_Ratio'] = info['trailingPE']
                    if 'priceToBook' in info and info['priceToBook'] is not None:
                        df['PB_Ratio'] = info['priceToBook']
                except:
                    pass
        except Exception as e:
            print(f"Error adding fundamental features: {e}")
    
    def _add_market_features(self, df: pd.DataFrame) -> None:
        """Add market index data for comparison"""
        try:
            # Convert index to datetime if needed
            if not isinstance(df.index, pd.DatetimeIndex):
                print(f"Warning: Index is not DatetimeIndex, converting for {self.ticker}")
                df.index = pd.to_datetime(df.index)
            
            # Get S&P 500 data with explicit date format
            start_date = df.index.min().strftime('%Y-%m-%d')
            end_date = df.index.max().strftime('%Y-%m-%d')
            
            spy = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
            
            if not spy.empty:
                # Calculate S&P 500 returns
                spy_returns = spy['Close'].pct_change()
                
                # Handle timezone differences by using date string matching
                df_dates = df.index.strftime('%Y-%m-%d')
                spy_dates = spy.index.strftime('%Y-%m-%d')
                
                # Create a mapping from dates to spy returns
                spy_returns_dict = {date: ret for date, ret in zip(spy_dates, spy_returns.values)}
                
                # Create a new series with matched dates
                matched_spy_returns = pd.Series([spy_returns_dict.get(date, np.nan) for date in df_dates], index=df.index)
                
                # Add to dataframe
                df['SPY_Return_1d'] = matched_spy_returns
                
                # Calculate correlation with market (safely)
                df['Market_Correlation'] = df['Return_1d'].rolling(window=20).corr(matched_spy_returns)
                
                # Calculate relative strength (safely)
                relative_strength = pd.Series(index=df.index)
                spy_returns_sum = matched_spy_returns.rolling(window=10).sum()
                
                for i in range(len(df)):
                    idx = df.index[i]
                    if not pd.isna(spy_returns_sum.iloc[i]) and spy_returns_sum.iloc[i] != -1:
                        relative_strength.iloc[i] = (1 + df['Return_10d'].iloc[i]) / (1 + spy_returns_sum.iloc[i]) - 1
                    else:
                        relative_strength.iloc[i] = 0
                        
                df['Relative_Strength'] = relative_strength
        except Exception as e:
            print(f"Error adding market features: {e}")
    
    def _detect_market_regime(self, df: pd.DataFrame) -> None:
        """Detect the current market regime based on recent price action"""
        try:
            # Make sure we have enough data
            if len(df) < 60:
                self.market_regime = "normal"
                return
                
            # Use the last 60 days of data (safely)
            recent = df.iloc[-min(60, len(df)):]
            
            # Calculate volatility
            volatility = recent['Return_1d'].std() * np.sqrt(252)
            
            # Calculate trend
            trend = recent['Close'].iloc[-1] / recent['Close'].iloc[0] - 1
            
            # Calculate average ADX if available
            avg_adx = recent['ADX'].mean() if 'ADX' in recent.columns else 0
            
            # Determine market regime
            if volatility > 0.4:  # High volatility
                self.market_regime = "volatile"
            elif trend > 0.1 and recent['Close'].iloc[-1] > recent['SMA_200'].iloc[-1]:
                self.market_regime = "bullish"
            elif trend < -0.1 and recent['Close'].iloc[-1] < recent['SMA_200'].iloc[-1]:
                self.market_regime = "bearish"
            else:
                self.market_regime = "normal"
            
            print(f"Detected market regime for {self.ticker}: {self.market_regime}")
        except Exception as e:
            print(f"Error detecting market regime: {e}")
            self.market_regime = "normal"  # Default to normal
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for model training"""
        # Define core features to use
        core_features = [
            'RSI', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200',
            'EMA_5', 'EMA_10', 'EMA_20',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'Return_1d', 'Return_3d', 'Return_5d', 'Return_10d',
            'Volatility_5d', 'Volatility_10d', 'Volatility_21d',
            'Volume_Change', 'Volume_Ratio', 'OBV',
            'BB_Width', 'BB_Pct',
            'Price_to_SMA_10', 'Price_to_SMA_50', 'Price_to_SMA_200',
            'Bull_Market', 'Correction', 'ADX',
            'Close', 'Volume'
        ]
        
        # Add market features if available
        market_features = [col for col in ['SPY_Return_1d', 'Market_Correlation', 'Relative_Strength'] 
                          if col in df.columns]
        
        # Add fundamental features if available
        fundamental_features = [col for col in ['Revenue', 'Net_Income', 'Profit_Margin', 'PE_Ratio', 'PB_Ratio'] 
                               if col in df.columns]
        
        # Combine all available features
        available_features = [f for f in core_features if f in df.columns]
        available_features.extend(market_features)
        available_features.extend(fundamental_features)
        
        # Update features list
        self.features = available_features
        
        # Prepare features and target
        X = df[self.features].values
        y = df[f'Future_Return_{self.forecast_days}d'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_model(self, forecast_days: int = 5) -> Dict[str, Union[str, float]]:
        """
        Train a new forecasting model
        
        Args:
            forecast_days: Number of days to forecast into the future
            
        Returns:
            Dict with training results
        """
        self.forecast_days = forecast_days
        
        try:
            # Get price history with technical indicators
            df = self._get_price_history(period="3y")
            
            if df.empty:
                return {"error": f"No data available for {self.ticker}"}
            
            # Prepare features and target
            X_scaled, y = self._prepare_features(df)
            
            # Create time series cross-validation splits
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Initialize model based on type
            if self.model_type == "ensemble":
                # Create an ensemble of models
                base_models = [
                    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                    ('gbr', GradientBoostingRegressor(n_estimators=100, random_state=42)),
                    ('ridge', Ridge(alpha=1.0, random_state=42))
                ]
                
                # Add neural net if available
                if NEURAL_NET_AVAILABLE:
                    base_models.append(('mlp', MLPRegressor(hidden_layer_sizes=(100, 50), 
                                                          max_iter=1000, random_state=42)))
                
                self.model = VotingRegressor(estimators=base_models)
                
            elif self.model_type == "random_forest":
                self.model = RandomForestRegressor(n_estimators=200, max_depth=10, 
                                                 min_samples_split=5, random_state=42)
                
            elif self.model_type == "gradient_boost":
                self.model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, 
                                                     max_depth=5, random_state=42)
                
            elif self.model_type == "linear":
                self.model = LinearRegression()
                
            elif self.model_type == "ridge":
                self.model = Ridge(alpha=1.0, random_state=42)
                
            elif self.model_type == "neural_net":
                if NEURAL_NET_AVAILABLE:
                    self.model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
                else:
                    # Fall back to RandomForest if neural net is not available
                    print(f"Neural network not available, using RandomForest for {self.ticker}")
                    self.model_type = "random_forest"
                    self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            elif self.model_type == "arima":
                if STATSMODELS_AVAILABLE:
                    # For ARIMA, we use a different approach - train on the raw price series
                    # We'll still use the ML model as a backup
                    try:
                        # Get just the closing prices
                        prices = df['Close'].values
                        # Fit ARIMA model
                        arima_model = sm.tsa.ARIMA(prices, order=(5,1,0))
                        arima_results = arima_model.fit()
                        # Store the ARIMA model
                        self.arima_model = arima_results
                        # Also train a backup ML model
                        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
                        self.model.fit(X_scaled, y)
                    except Exception as e:
                        print(f"Error fitting ARIMA model: {e}, using RandomForest instead")
                        self.model_type = "random_forest"
                        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
                else:
                    # Fall back to RandomForest if statsmodels is not available
                    print(f"Statsmodels not available, using RandomForest for {self.ticker}")
                    self.model_type = "random_forest"
                    self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                return {"error": f"Unknown model type: {self.model_type}"}
            
            # Train model
            self.model.fit(X_scaled, y)
            
            # Perform cross-validation to assess model quality
            cv_scores = {}
            try:
                # Use time series cross-validation
                cv_predictions = np.zeros_like(y)
                for train_idx, test_idx in tscv.split(X_scaled):
                    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    # Train on each fold
                    fold_model = clone(self.model)
                    fold_model.fit(X_train, y_train)
                    
                    # Predict on test fold
                    cv_predictions[test_idx] = fold_model.predict(X_test)
                
                # Calculate metrics
                cv_mse = mean_squared_error(y, cv_predictions)
                cv_rmse = np.sqrt(cv_mse)
                cv_mae = mean_absolute_error(y, cv_predictions)
                cv_r2 = r2_score(y, cv_predictions)
                
                cv_scores = {
                    "cv_rmse": round(cv_rmse, 4),
                    "cv_mae": round(cv_mae, 4),
                    "cv_r2": round(cv_r2, 4)
                }
                
                # Store validation metrics
                self.validation_metrics = cv_scores
            except Exception as e:
                print(f"Error during cross-validation: {e}")
            
            # Extract feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                self.feature_importance = {}
                
                for i, feature in enumerate(self.features):
                    self.feature_importance[feature] = round(float(importance[i]), 4)
                
                # Sort by importance
                self.feature_importance = {k: v for k, v in sorted(
                    self.feature_importance.items(), key=lambda item: item[1], reverse=True
                )}
            
            # Record training time
            self.last_trained = datetime.now()
            
            # Save model
            self._save_model()
            
            # Calculate training metrics
            train_pred = self.model.predict(X_scaled)
            mse = mean_squared_error(y, train_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, train_pred)
            r2 = r2_score(y, train_pred)
            
            result = {
                "ticker": self.ticker,
                "model_type": self.model_type,
                "forecast_days": self.forecast_days,
                "training_samples": len(y),
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "r2": round(r2, 4),
                "trained_date": self.last_trained.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "success"
            }
            
            # Add cross-validation scores if available
            if cv_scores:
                result.update(cv_scores)
            
            return result
            
        except Exception as e:
            return {"error": f"Error training model: {str(e)}"}
    
    def get_forecast(self) -> Dict[str, Union[str, float, Dict]]:
        """
        Generate price forecast with confidence intervals
        
        Returns:
            Dict with forecast results
        """
        try:
            # Check if model exists
            if self.model is None:
                # Try to train a new model
                training_result = self.train_model(self.forecast_days)
                if "error" in training_result:
                    return training_result
            
            # Get recent price data
            stock = yf.Ticker(self.ticker)
            recent_data = stock.history(period="60d")
            
            if recent_data.empty:
                return {"error": f"No recent data available for {self.ticker}"}
            
            # Calculate all the same technical indicators as in training
            # RSI
            delta = recent_data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            recent_data['RSI'] = 100 - (100 / (1 + rs))
            
            # Moving Averages
            for window in [5, 10, 20, 50, 100, 200]:
                recent_data[f'SMA_{window}'] = recent_data['Close'].rolling(window=window).mean()
                
            # Exponential Moving Averages
            for window in [5, 10, 20, 50]:
                recent_data[f'EMA_{window}'] = recent_data['Close'].ewm(span=window, adjust=False).mean()
            
            # MACD
            recent_data['MACD'] = recent_data['Close'].ewm(span=12, adjust=False).mean() - recent_data['Close'].ewm(span=26, adjust=False).mean()
            recent_data['MACD_Signal'] = recent_data['MACD'].ewm(span=9, adjust=False).mean()
            recent_data['MACD_Hist'] = recent_data['MACD'] - recent_data['MACD_Signal']
            
            # Price momentum
            for days in [1, 3, 5, 10, 21]:
                recent_data[f'Return_{days}d'] = recent_data['Close'].pct_change(days)
            
            # Volatility
            for window in [5, 10, 21]:
                recent_data[f'Volatility_{window}d'] = recent_data['Return_1d'].rolling(window=window).std() * np.sqrt(252)  # Annualized
            
            # Volume indicators
            recent_data['Volume_Change'] = recent_data['Volume'].pct_change(1)
            recent_data['Volume_MA_10'] = recent_data['Volume'].rolling(window=10).mean()
            recent_data['Volume_Ratio'] = recent_data['Volume'] / recent_data['Volume_MA_10']
            recent_data['OBV'] = (np.sign(recent_data['Close'].diff()) * recent_data['Volume']).fillna(0).cumsum()
            
            # Bollinger Bands
            recent_data['BB_Middle'] = recent_data['Close'].rolling(window=20).mean()
            recent_data['BB_Std'] = recent_data['Close'].rolling(window=20).std()
            recent_data['BB_Upper'] = recent_data['BB_Middle'] + (recent_data['BB_Std'] * 2)
            recent_data['BB_Lower'] = recent_data['BB_Middle'] - (recent_data['BB_Std'] * 2)
            recent_data['BB_Width'] = (recent_data['BB_Upper'] - recent_data['BB_Lower']) / recent_data['BB_Middle']
            recent_data['BB_Pct'] = (recent_data['Close'] - recent_data['BB_Lower']) / (recent_data['BB_Upper'] - recent_data['BB_Lower'])
            
            # Price relative to moving averages
            for ma in [10, 20, 50, 200]:
                if f'SMA_{ma}' in recent_data.columns:
                    recent_data[f'Price_to_SMA_{ma}'] = recent_data['Close'] / recent_data[f'SMA_{ma}'] - 1
            
            # Market regime features
            if 'SMA_200' in recent_data.columns:
                recent_data['Bull_Market'] = (recent_data['Close'] > recent_data['SMA_200']).astype(int)
            else:
                recent_data['Bull_Market'] = 0
                
            # Correction (if we have enough data)
            if len(recent_data) >= 60:
                recent_data['Correction'] = ((recent_data['Close'] / recent_data['Close'].rolling(window=60).max() - 1) < -0.1).astype(int)
            else:
                recent_data['Correction'] = 0
            
            # Trend strength
            recent_data['ADX'] = self._calculate_adx(recent_data)
            
            # Try to add market features
            try:
                # Get S&P 500 data for the same period
                spy = yf.download('^GSPC', period="60d", progress=False)
                if not spy.empty:
                    # Calculate S&P 500 returns
                    spy_returns = spy['Close'].pct_change()
                    
                    # Handle timezone differences by using date string matching
                    recent_data_dates = recent_data.index.strftime('%Y-%m-%d')
                    spy_dates = spy.index.strftime('%Y-%m-%d')
                    
                    # Create a mapping from dates to spy returns
                    spy_returns_dict = {date: ret for date, ret in zip(spy_dates, spy_returns.values)}
                    
                    # Create a new series with matched dates
                    matched_spy_returns = pd.Series([spy_returns_dict.get(date, np.nan) for date in recent_data_dates], index=recent_data.index)
                    
                    # Add to dataframe (ensure it's a scalar value)
                    recent_data['SPY_Return_1d'] = matched_spy_returns.astype(float)
                    
                    # Calculate correlation with market (safely)
                    if len(recent_data) >= 20:
                        recent_data['Market_Correlation'] = recent_data['Return_1d'].rolling(window=min(20, len(recent_data))).corr(matched_spy_returns)
                    else:
                        recent_data['Market_Correlation'] = 0
                    
                    # Calculate relative strength (safely)
                    recent_data['Relative_Strength'] = 0  # Default value
                    
                    if 'Return_10d' in recent_data.columns and len(recent_data) >= 10:
                        spy_returns_sum = matched_spy_returns.rolling(window=min(10, len(recent_data))).sum()
                        
                        for i in range(len(recent_data)):
                            if i < len(spy_returns_sum) and not pd.isna(spy_returns_sum.iloc[i]) and spy_returns_sum.iloc[i] != -1:
                                if not pd.isna(recent_data['Return_10d'].iloc[i]):
                                    try:
                                        recent_data.loc[recent_data.index[i], 'Relative_Strength'] = (1 + recent_data['Return_10d'].iloc[i]) / (1 + spy_returns_sum.iloc[i]) - 1
                                    except Exception as e:
                                        print(f"Error calculating relative strength at index {i}: {e}")
            except Exception as e:
                print(f"Could not add market features for forecast: {e}")
            
            # Get the most recent complete data point (safely)
            recent_data_clean = recent_data.dropna(subset=['Close'])
            if len(recent_data_clean) == 0:
                return {"error": f"No complete data available for {self.ticker}"}
                
            latest_data = recent_data_clean.iloc[-1]
            
            # Extract features (safely handling missing features)
            feature_values = []
            for feature in self.features:
                if feature in latest_data:
                    # Ensure the value is a scalar
                    value = latest_data[feature]
                    if isinstance(value, (list, tuple, np.ndarray)):
                        print(f"Warning: Feature {feature} is not a scalar for {self.ticker}, using first element")
                        try:
                            value = float(value[0])
                        except (IndexError, TypeError, ValueError):
                            print(f"Warning: Could not convert {feature} to scalar, using 0")
                            value = 0.0
                    # Convert to float to ensure consistent types
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        print(f"Warning: Could not convert {feature} value to float, using 0")
                        value = 0.0
                    feature_values.append(value)
                else:
                    print(f"Warning: Feature {feature} not found in latest data for {self.ticker}, using 0")
                    feature_values.append(0.0)
            
            # Ensure all values are scalars and create a 2D array
            X = np.array(feature_values, dtype=float).reshape(1, -1)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Handle NaN values
            if np.isnan(X_scaled).any():
                print(f"Warning: NaN values detected in scaled features for {self.ticker}, filling with 0")
                X_scaled = np.nan_to_num(X_scaled, nan=0.0)
            
            # Make prediction
            try:
                predicted_return = self.model.predict(X_scaled)[0]
                
                # Sanity check on prediction (cap extreme values)
                if np.isnan(predicted_return) or np.isinf(predicted_return):
                    print(f"Warning: Invalid prediction value for {self.ticker}, using 0")
                    predicted_return = 0.0
                elif abs(predicted_return) > 0.5:  # Cap at 50% in either direction
                    print(f"Warning: Extreme prediction value {predicted_return} for {self.ticker}, capping at +/-50%")
                    # For stable blue-chip stocks, use a more conservative cap
                    if self.ticker in ["JNJ", "PG", "KO", "PEP", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "BRK-B", "JPM", "V", "WMT", "HD", "UNH"]:
                        # Use a more reasonable cap for stable stocks (10% for blue chips)
                        predicted_return = np.sign(predicted_return) * min(abs(predicted_return), 0.1)
                        print(f"  Further limiting prediction for blue-chip stock {self.ticker} to {predicted_return}")
                    else:
                        predicted_return = np.sign(predicted_return) * 0.5
                
                # Additional sanity check based on historical volatility
                historical_returns = recent_data['Close'].pct_change(self.forecast_days).dropna()
                std_dev = historical_returns.std()
                mean_return = historical_returns.mean()
                
                # For newly trained models, be more conservative
                if self.last_trained and (datetime.now() - self.last_trained).total_seconds() < 3600:  # Model trained in the last hour
                    print(f"New model detected for {self.ticker}, applying conservative prediction adjustment")
                    
                    # For new models, bias towards historical mean with a small adjustment
                    # Use 80% historical mean + 20% model prediction
                    adjusted_return = (0.8 * mean_return) + (0.2 * predicted_return)
                    print(f"  Adjusting prediction from {predicted_return} to {adjusted_return} (80% hist mean + 20% model)")
                    predicted_return = adjusted_return
                
                # If prediction is more than 3 standard deviations from the mean, cap it
                elif abs(predicted_return - mean_return) > 3 * std_dev:
                    # Cap at 3 standard deviations from the mean
                    max_reasonable_return = mean_return + (3 * std_dev * np.sign(predicted_return - mean_return))
                    print(f"Warning: Prediction {predicted_return} is > 3 std devs from mean {mean_return} for {self.ticker}")
                    print(f"  Capping at {max_reasonable_return} (3 std devs from mean)")
                    predicted_return = max_reasonable_return
                
                # For stocks with limited data, be more conservative
                if len(historical_returns) < 100:
                    print(f"Limited historical data for {self.ticker} ({len(historical_returns)} samples), applying conservative adjustment")
                    # Bias more towards zero/small positive return for stocks with limited data
                    conservative_return = 0.5 * predicted_return + 0.5 * 0.01  # Bias towards 1% return
                    print(f"  Adjusting prediction from {predicted_return} to {conservative_return}")
                    predicted_return = conservative_return
            except Exception as e:
                print(f"Error making prediction: {e}, using 0")
                predicted_return = 0.0
            
            # Calculate confidence interval using historical standard deviation
            historical_returns = recent_data['Close'].pct_change(self.forecast_days).dropna()
            std_dev = historical_returns.std()
            
            # Calculate z-scores for different confidence levels
            z_90 = 1.645  # 90% confidence
            z_95 = 1.96   # 95% confidence
            
            # Calculate confidence intervals
            lower_90 = predicted_return - (z_90 * std_dev)
            upper_90 = predicted_return + (z_90 * std_dev)
            lower_95 = predicted_return - (z_95 * std_dev)
            upper_95 = predicted_return + (z_95 * std_dev)
            
            # Calculate predicted prices
            current_price = latest_data['Close']
            predicted_price = current_price * (1 + predicted_return)
            lower_price_90 = current_price * (1 + lower_90)
            upper_price_90 = current_price * (1 + upper_90)
            lower_price_95 = current_price * (1 + lower_95)
            upper_price_95 = current_price * (1 + upper_95)
            
            # Calculate average returns
            avg_return_5d = recent_data['Return_5d'].mean()
            avg_return_10d = recent_data['Return_10d'].mean()
            
            # Calculate z-score of prediction compared to historical distribution
            z_score = (predicted_return - historical_returns.mean()) / std_dev
            
            # Determine forecast strength
            if abs(z_score) < 0.5:
                strength = "weak"
            elif abs(z_score) < 1.0:
                strength = "moderate"
            else:
                strength = "strong"
            
            # Determine direction
            if predicted_return > 0:
                direction = "bullish"
            else:
                direction = "bearish"
            
            # Format the result
            result = {
                "ticker": self.ticker,
                "current_price": round(current_price, 2),
                "forecast_days": self.forecast_days,
                "predicted_return": round(predicted_return * 100, 2),
                "predicted_price": round(predicted_price, 2),
                "confidence_intervals": {
                    "90%": {
                        "lower_return": round(lower_90 * 100, 2),
                        "upper_return": round(upper_90 * 100, 2),
                        "lower_price": round(lower_price_90, 2),
                        "upper_price": round(upper_price_90, 2)
                    },
                    "95%": {
                        "lower_return": round(lower_95 * 100, 2),
                        "upper_return": round(upper_95 * 100, 2),
                        "lower_price": round(lower_price_95, 2),
                        "upper_price": round(upper_price_95, 2)
                    }
                },
                "historical_metrics": {
                    "avg_return_5d": round(avg_return_5d * 100, 2),
                    "avg_return_10d": round(avg_return_10d * 100, 2),
                    "return_volatility": round(std_dev * 100, 2)
                },
                "forecast_metrics": {
                    "z_score": round(z_score, 2),
                    "strength": strength,
                    "direction": direction
                },
                "model_info": {
                    "model_type": self.model_type,
                    "last_trained": self.last_trained.strftime("%Y-%m-%d") if self.last_trained else "Never",
                    "feature_importance": self._get_feature_importance() if hasattr(self.model, 'feature_importances_') else None
                },
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Error generating forecast: {str(e)}"}
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the model if available"""
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        importance = self.model.feature_importances_
        feature_importance = {}
        
        for i, feature in enumerate(self.features):
            feature_importance[feature] = round(float(importance[i]), 4)
        
        # Sort by importance
        feature_importance = {k: v for k, v in sorted(
            feature_importance.items(), key=lambda item: item[1], reverse=True
        )}
        
        return feature_importance


def train_model(ticker: str, model_type: str = "random_forest", forecast_days: int = 5) -> Dict[str, Union[str, float]]:
    """
    Train a forecasting model for a specific ticker
    
    Args:
        ticker: Stock or ETF symbol
        model_type: Type of model to use ('random_forest', 'linear', 'xgboost')
        forecast_days: Number of days to forecast into the future
        
    Returns:
        Dict with training results
    """
    forecaster = PriceForecaster(ticker, model_type)
    return forecaster.train_model(forecast_days)


def get_forecast(ticker: str, model_type: str = "random_forest", forecast_days: int = 5) -> Dict[str, Union[str, float, Dict]]:
    """
    Get price forecast for a specific ticker
    
    Args:
        ticker: Stock or ETF symbol
        model_type: Type of model to use ('random_forest', 'linear', 'xgboost')
        forecast_days: Number of days to forecast into the future
        
    Returns:
        Dict with forecast results including:
        - predicted_return: Forecasted percentage return
        - predicted_price: Forecasted price
        - confidence_intervals: Lower and upper bounds at different confidence levels
        - forecast_metrics: Z-score, strength, and direction of the forecast
    """
    forecaster = PriceForecaster(ticker, model_type)
    forecaster.forecast_days = forecast_days
    return forecaster.get_forecast()
