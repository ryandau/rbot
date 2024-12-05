from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from typing import List, Dict, Optional
import logging
from pydantic import BaseModel
from datetime import datetime
import aiohttp
import hmac
import hashlib
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the data directory
DATA_DIR = 'data'

# Ensure the data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Settings(BaseModel):
    POLL_INTERVAL: int = int(os.getenv('POLL_INTERVAL', 30))
    INITIAL_INVESTMENT: float = float(os.getenv('INITIAL_INVESTMENT', 500))  # AUD
    RISK_THRESHOLD: float = float(os.getenv('RISK_THRESHOLD', 0.65))
    MAX_HISTORY_DAYS: int = int(os.getenv('MAX_HISTORY_DAYS', 14))
    PRICE_BUFFER: float = float(os.getenv('PRICE_BUFFER', 300))  # AUD
    NTFY_TOPIC: str = os.getenv('NTFY_TOPIC')

    # API keys
    COINSPOT_API_KEY: str = os.getenv('COINSPOT_API_KEY')
    COINSPOT_API_SECRET: str = os.getenv('COINSPOT_API_SECRET')

    # Trading thresholds
    MIN_VOLATILITY_THRESHOLD: float = float(os.getenv('MIN_VOLATILITY_THRESHOLD', 0.0002))
    MAX_VOLATILITY_THRESHOLD: float = float(os.getenv('MAX_VOLATILITY_THRESHOLD', 0.02))
    CONFIDENCE_THRESHOLD: float = float(os.getenv('CONFIDENCE_THRESHOLD', 0.42))
    SIGNAL_AGREEMENT_REQUIRED: int = int(os.getenv('SIGNAL_AGREEMENT_REQUIRED', 2))

    # Protection settings
    MAX_DRAWDOWN_PCT: float = float(os.getenv('MAX_DRAWDOWN_PCT', 10.0))
    MAX_DECLINE_RATE_PCT: float = float(os.getenv('MAX_DECLINE_RATE_PCT', 2.5))
    MAX_TOTAL_EXPOSURE: float = float(os.getenv('MAX_TOTAL_EXPOSURE', 750.0))  # AUD
    PRICE_VALIDATION_THRESHOLD: float = float(os.getenv('PRICE_VALIDATION_THRESHOLD', 1.0))
    STOP_LOSS_PCT: float = float(os.getenv('STOP_LOSS_PCT', 5.0))

    # Technical analysis
    SMA_SHORT_PERIOD: int = int(os.getenv('SMA_SHORT_PERIOD', 5))
    SMA_LONG_PERIOD: int = int(os.getenv('SMA_LONG_PERIOD', 14))
    EMA_ALPHA: float = float(os.getenv('EMA_ALPHA', 0.2))

class NotificationManager:
    def __init__(self, ntfy_topic: str):
        self.ntfy_url = f"https://ntfy.sh/{ntfy_topic}"
        self.session = None

    async def get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def send_notification(self, title: str, message: str, priority: int = 3):
        try:
            headers = {
                "Title": title,
                "Priority": str(priority),
                "Tags": "bitcoin,trading"
            }
            session = await self.get_session()
            async with session.post(
                self.ntfy_url,
                data=message.encode(encoding='utf-8'),
                headers=headers
            ) as response:
                if response.status != 200:
                    logger.error(f"Failed to send notification: {response.status}")
        except Exception as e:
            logger.error(f"Error sending notification: {e}")

class PriceLevel(BaseModel):
    price: float
    allocation: float
    triggered: bool = False
    last_checked: datetime = datetime.now()

# Initial settings
settings = Settings()

# Price levels
price_levels: Dict[float, PriceLevel] = {}

class MarketAnalysis:
    def __init__(self, max_history: int = 14, price_history: List[float] = None):
        self.price_history = price_history if price_history is not None else []
        self.max_history = max_history
        self.last_update = None
        self.last_price = None
        self.total_price_points = len(self.price_history)
        self.min_data_points = 5
        self.logger = logging.getLogger(__name__)
        self.history_file = os.path.join(DATA_DIR, 'price_history.json')

        # Technical analysis config from settings
        self.sma_short_period = settings.SMA_SHORT_PERIOD
        self.sma_long_period = settings.SMA_LONG_PERIOD
        self.ema_alpha = settings.EMA_ALPHA

    def load_history(self) -> List[float]:
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                return data['price_history'][-self.max_history:]
        except:
            return []

    def save_history(self):
        try:
            with open(self.history_file, 'w') as f:
                json.dump({
                    'price_history': self.price_history,
                    'last_update': str(datetime.now())
                }, f)
        except Exception as e:
            self.logger.error(f"Error saving history: {e}")

    async def update_history(self, price: float):
        """Update price history with validation"""
        if price <= 0:
            self.logger.error(f"Invalid price for history update: {price}")
            return

        current_time = datetime.now()
        time_passed = (current_time - self.last_update).total_seconds() if self.last_update else None
        time_passed_str = f"{time_passed:.1f}s" if time_passed is not None else "0s"

        # Validate price movement
        if self.price_history:
            price_change = abs(price - self.price_history[0]) / self.price_history[0]
            if price_change > settings.PRICE_VALIDATION_THRESHOLD:
                self.logger.warning(f"Large price movement detected: {price_change:.2%}")

        should_update = (
            not self.price_history or
            price != self.last_price or
            (time_passed and time_passed >= settings.POLL_INTERVAL)
        )

        if should_update:
            self.price_history.insert(0, price)
            if len(self.price_history) > self.max_history:
                self.price_history.pop()

            self.last_update = current_time
            self.last_price = price
            self.total_price_points += 1
            self.save_history()

            # Log the update
            self.logger.info(
                f"Price history updated - New: {price}, "
                f"Points: {len(self.price_history)}, "
                f"Time passed: {time_passed_str}, "
                f"History: {self.price_history[:5]}"
            )
        else:
            self.logger.debug(
                f"Update skipped - Current: {price}, "
                f"Last: {self.last_price}, "
                f"Time passed: {time_passed_str}"
            )

    def calculate_sma(self, period: int) -> Optional[float]:
        try:
            if not self.price_history or len(self.price_history) < period:
                self.logger.debug(f"Price history state: {self.price_history}")
                return None

            values = self.price_history[:period]
            self.logger.debug(f"Calculating SMA with values: {values}")
            sma = sum(values) / period
            return sma

        except Exception as e:
            self.logger.error(f"Error calculating SMA({period}): {e}")
            return None

    def calculate_ema(self, period: int) -> Optional[float]:
        """Calculate Exponential Moving Average iteratively"""
        try:
            if len(self.price_history) < period:
                return None

            # First EMA value is SMA
            ema = sum(self.price_history[:period]) / period

            # Calculate EMA iteratively for remaining prices
            alpha = self.ema_alpha  # Smoothing factor
            for price in self.price_history[:period-1]:
                ema = price * alpha + ema * (1 - alpha)

            return ema
        except Exception as e:
            self.logger.error(f"Error calculating EMA({period}): {e}")
            return None

    def calculate_momentum(self, period: int = 10) -> Optional[float]:
        """Calculate momentum indicator"""
        try:
            if len(self.price_history) < period:
                return None
            return (self.price_history[0] - self.price_history[period - 1]) / self.price_history[period - 1]
        except Exception as e:
            self.logger.error(f"Error calculating momentum: {e}")
            return None

    def calculate_linear_regression(self) -> tuple:
        """Calculate linear regression slope and R-squared"""
        try:
            if len(self.price_history) < self.min_data_points:
                return None, None

            x = list(range(len(self.price_history)))
            y = self.price_history

            n = len(x)
            x_mean = sum(x) / n
            y_mean = sum(y) / n

            # Calculate slope
            numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
            slope = numerator / denominator if denominator != 0 else 0

            # Calculate R-squared
            y_pred = [slope * (x[i] - x_mean) + y_mean for i in range(n)]
            ss_tot = sum((y[i] - y_mean) ** 2 for i in range(n))
            ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
            r_squared = max(0, 1 - (ss_res / ss_tot)) if ss_tot != 0 else 0

            return slope, r_squared

        except Exception as e:
            self.logger.error(f"Error calculating linear regression: {e}")
            return None, None

    def calculate_volatility(self) -> float:
        """Calculate volatility using rolling standard deviation of returns with proper scaling"""
        try:
            if len(self.price_history) < 2:
                return 0.0

            # Calculate percentage changes
            changes = [
                (self.price_history[i] - self.price_history[i+1]) / self.price_history[i+1]
                for i in range(len(self.price_history)-1)
            ]

            # Calculate weighted standard deviation with more weight on recent changes
            weights = [1 - (i / len(changes)) for i in range(len(changes))]
            weighted_changes = [c * w for c, w in zip(changes, weights)]

            mean = sum(weighted_changes) / sum(weights)
            variance = sum(w * ((c - mean) ** 2) for c, w in zip(changes, weights)) / sum(weights)

            # Scale the volatility to match your thresholds
            volatility = abs(variance ** 0.5)  # Taking absolute value for safety

            self.logger.debug(f"Calculated volatility: {volatility:.6f} from {len(changes)} changes")
            return volatility

        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return 0.0

    def determine_trend(self, trend_signals: Dict, trend_score: float, volatility: float) -> str:
        """Determine trend with more nuanced classification"""
        try:
            # Count agreeing signals
            signal_values = list(trend_signals.values())
            agreement = len([s for s in signal_values if s == signal_values[0] and s != 0])

            # Normalize trend score to -1 to 1 range
            norm_score = max(min(trend_score, 1), -1)

            # Factor in volatility
            vol_factor = min(volatility / settings.MIN_VOLATILITY_THRESHOLD, 1)
            adjusted_score = norm_score * vol_factor

            # Determine trend based on adjusted score and agreement
            if abs(adjusted_score) < 0.1 or agreement < 2:
                return "neutral"
            elif adjusted_score > 0:
                return "strong_upward" if adjusted_score > 0.5 and agreement >= 2 else "upward"
            else:
                return "strong_downward" if adjusted_score < -0.5 and agreement >= 2 else "downward"

        except Exception as e:
            self.logger.error(f"Error determining trend: {e}")
            return "neutral"

    def analyze_trend_signals(self, trend_signals: dict, trend: str) -> tuple[int, bool]:
        """
        Analyze trend signals more comprehensively.
        Returns (number of agreeing signals, whether they meet the requirement)
        """
        try:
            signal_values = list(trend_signals.values())

            # Count agreeing signals based on trend direction
            if trend in ["upward", "strong_upward"]:
                agreeing = sum(1 for s in signal_values if s == 1)
            elif trend in ["downward", "strong_downward"]:
                agreeing = sum(1 for s in signal_values if s == -1)
            else:  # neutral
                # For neutral trend, count most common signal
                from collections import Counter
                signal_counts = Counter(signal_values)
                agreeing = max(signal_counts.values())

            meets_requirement = agreeing >= settings.SIGNAL_AGREEMENT_REQUIRED

            return agreeing, meets_requirement

        except Exception as e:
            self.logger.error(f"Error analyzing trend signals: {e}")
            return 0, False

    async def analyze_market_conditions(self) -> dict:
        """Analyze current market conditions with improved calculations"""
        if len(self.price_history) < self.min_data_points:
            return self._get_default_conditions()

        try:
            # Calculate indicators
            sma_short = self.calculate_sma(self.sma_short_period)
            sma_long = self.calculate_sma(self.sma_long_period)
            ema_short = self.calculate_ema(self.sma_short_period)
            slope, r_squared = self.calculate_linear_regression()
            momentum = self.calculate_momentum()

            # Calculate volatility with improved method
            volatility = self.calculate_volatility()

            # Determine trend signals
            trend_signals = {
                'sma_signal': (
                    1 if (sma_short is not None and sma_long is not None and sma_short > sma_long)
                    else -1 if (sma_short is not None and sma_long is not None and sma_short < sma_long)
                    else 0
                ),
                'momentum_signal': (
                    1 if (momentum is not None and momentum > 0)
                    else -1 if (momentum is not None and momentum < 0)
                    else 0
                ),
                'regression_signal': (
                    1 if (slope is not None and slope > 0)
                    else -1 if (slope is not None and slope < 0)
                    else 0
                )
            }

            # Calculate weighted trend score
            weights = {'sma_signal': 0.35, 'momentum_signal': 0.35, 'regression_signal': 0.30}
            trend_score = sum(signal * weights[name] for name, signal in trend_signals.items())

            # Determine trend with improved classification
            trend = self.determine_trend(trend_signals, trend_score, volatility)

            # Calculate confidence with more factors
            signal_agreement = len(set(trend_signals.values()))
            signal_confidence = (3 - signal_agreement) / 3
            trend_strength = abs(trend_score)
            regression_quality = r_squared if r_squared is not None else 0
            vol_confidence = min(volatility / settings.MAX_VOLATILITY_THRESHOLD, 1)

            confidence = (signal_confidence * 0.3 +
                        trend_strength * 0.3 +
                        regression_quality * 0.2 +
                        vol_confidence * 0.2)

            # Ensure confidence is between 0 and 1
            confidence = max(0, min(1, confidence))

            return {
                'trend': trend,
                'confidence': confidence,
                'volatility': volatility,
                'risk_level': sum({
                    'volatility': min(1, volatility * 10),
                    'trend_reversal': 0.5 if getattr(self, '_previous_trend', trend) != trend else 0,
                    'signal_disagreement': signal_agreement / 3,
                }.values()) / 3,
                'indicators': {
                    'sma_short': sma_short,
                    'sma_long': sma_long,
                    'ema_short': ema_short,
                    'momentum': momentum,
                    'regression_slope': slope,
                    'r_squared': r_squared
                },
                'trend_signals': trend_signals,
                'trend_change': trend != getattr(self, '_previous_trend', trend),
                'waiting_for_data': False,
                'price_points': self.total_price_points,
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"Error in market analysis: {e}", exc_info=True)
            return self._get_default_conditions()

    def _get_default_conditions(self) -> dict:
        """Return default market conditions when analysis cannot be performed"""
        return {
            'trend': "neutral",
            'confidence': 0.0,
            'volatility': 0.0,
            'risk_level': 0.0,
            'indicators': {
                'sma_short': None,
                'sma_long': None,
                'ema_short': None,
                'momentum': None,
                'regression_slope': None,
                'r_squared': None
            },
            'trend_signals': {
                'sma_signal': 0,
                'momentum_signal': 0,
                'regression_signal': 0
            },
            'trend_change': False,
            'waiting_for_data': True,
            'price_points': self.total_price_points,
            'timestamp': datetime.now()
        }

class CoinspotAPI:
    """Handles authenticated requests to Coinspot API."""
    def __init__(self, api_key: str, api_secret: str):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_endpoint = 'https://www.coinspot.com.au/api/v2'  # Base API endpoint
        self.session = aiohttp.ClientSession()

    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def request(self, path: str, data: dict, read_only: bool = False) -> dict:
        """Make an authenticated request to the Coinspot API."""
        try:
            # Add nonce if not present
            if 'nonce' not in data:
                data['nonce'] = int(datetime.now().timestamp() * 1000)

            post_data = json.dumps(data, separators=(',', ':'))

            # Create HMAC signature
            sign = hmac.new(
                self.api_secret.encode('utf-8'),
                post_data.encode('utf-8'),
                hashlib.sha512
            ).hexdigest()

            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'key': self.api_key,
                'sign': sign
            }

            # Construct URL based on read_only flag
            if read_only:
                url = f"{self.api_endpoint}/ro{path}"
            else:
                url = f"{self.api_endpoint}{path}"

            self.logger.info(f"Making request to {url}")
            self.logger.info(f"Request data: {post_data}")

            async with self.session.post(url, data=post_data, headers=headers) as response:
                response_text = await response.text()
                self.logger.info(f"API Response: {response_text}")

                if response.status == 200:
                    result = json.loads(response_text)
                    if result.get('status') == 'ok':
                        return result
                    else:
                        self.logger.error(f"API Error: {result}")
                        return None
                else:
                    self.logger.error(f"HTTP Error: Status {response.status}")
                    self.logger.error(f"Response: {response_text}")
                    return None

        except Exception as e:
            self.logger.error(f"Error in CoinspotAPI request: {e}", exc_info=True)
            return None

class BTCTrader:
    def __init__(self):
        # Initialize basic attributes
        self.position_locks = {}
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.last_btc_price = None
        self.peak_portfolio_value = 0.0
        self.last_prices = []  # Store recent prices for decline rate

        # Update file paths
        self.state_file = os.path.join(DATA_DIR, 'trader_state.json')
        self.history_file = os.path.join(DATA_DIR, 'price_history.json')

        # Setup logging first
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [BTCTrader] %(message)s')
        file_handler = logging.FileHandler('btc_trader.log')
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Avoid adding multiple handlers if they already exist
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            self.logger.addHandler(file_handler)
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            self.logger.addHandler(console_handler)

        self.logger.setLevel(logging.INFO)
        self.logger.info("BTCTrader logger initialized")

        # Initialize with default values before loading state
        self.price_history = []
        self.active_positions = []
        self.price_alerts = []

        # Load previous state
        self.load_state()

        # Initialize components after loading state
        self.market_analysis = MarketAnalysis(
            settings.MAX_HISTORY_DAYS,
            price_history=self.price_history
        )
        self.notifications = NotificationManager(settings.NTFY_TOPIC)
        self.price_buffer = settings.PRICE_BUFFER

        # Initialize CoinspotAPI with credentials
        self.coinspot_api = CoinspotAPI(settings.COINSPOT_API_KEY, settings.COINSPOT_API_SECRET)

    def load_state(self):
        """Load previous trading state"""
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            self.price_history = state.get('price_history', [])
            # Initialize market analysis ONCE with proper history
            if not hasattr(self, 'market_analysis'):
                self.market_analysis = MarketAnalysis(
                    settings.MAX_HISTORY_DAYS,
                    price_history=self.price_history
                )
            else:
                # Update existing market analysis
                self.market_analysis.price_history = self.price_history

            self.logger.info(
                f"State loaded successfully: "
                f"{len(self.price_history)} price points, "
                f"{len(self.active_positions)} active positions"
            )

        except FileNotFoundError:
            self.logger.info("No previous state found, starting fresh")
            self.price_history = []
            self.active_positions = []
            self.price_alerts = []
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            self.price_history = []
            self.active_positions = []
            self.price_alerts = []

    def save_state(self):
        try:
            state = {
                'price_history': self.market_analysis.price_history,
                'last_btc_price': self.last_btc_price,
                'active_positions': self.active_positions,
                'price_alerts': self.price_alerts,
                'last_update': str(datetime.now()),
                'total_price_points': self.market_analysis.total_price_points,
                'peak_portfolio_value': self.peak_portfolio_value
            }

            # Write to temporary file first
            temp_file = f"{self.state_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(state, f)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename
            os.replace(temp_file, self.state_file)

            self.logger.info(f"State saved: {len(self.market_analysis.price_history)} points")

        except Exception as e:
            self.logger.error(f"Error saving state: {e}")

    async def validate_state_consistency(self):
        """Validate state consistency"""
        try:
            # Check position-level matching
            for position in self.active_positions:
                if not any(
                    abs(float(position['price_level']) - float(price)) < 0.0001
                    for price in price_levels.keys()
                ):
                    self.logger.warning(
                        f"Invalid position found for level {position['price_level']}"
                    )
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating state consistency: {e}")
            return False

    async def sync_historical_positions(self):
        """Sync positions with CoinSpot order history"""
        try:
            self.logger.info("Starting historical position sync...")

            # First, ensure we have price levels
            if not price_levels:
                self.logger.error("No price levels configured")
                return False

            data = {
                'cointype': 'BTC',
                'limit': 500
            }

            history = await self.coinspot_api.request('/my/orders/completed', data, read_only=True)

            if not history or 'buyorders' not in history:
                self.logger.error("Failed to fetch order history")
                return False

            buy_orders = history.get('buyorders', [])
            sell_orders = history.get('sellorders', [])

            # Track sold positions by creating a set of sold amounts
            sold_amounts = set()
            for sell in sell_orders:
                try:
                    sold_amounts.add(float(sell['amount']))
                except (ValueError, KeyError):
                    continue

            # Sort price levels for better matching
            sorted_levels = sorted(price_levels.items(), key=lambda x: x[0], reverse=True)

            # Process active positions
            active_positions = []
            positions_by_level = {}  # Track positions per level

            for order in buy_orders:
                try:
                    entry_price = float(order['rate'])
                    btc_amount = float(order['amount'])

                    # Skip if amount has been sold
                    if btc_amount in sold_amounts:
                        self.logger.info(f"Skipping sold position: {btc_amount} BTC")
                        continue

                    # Find appropriate price level
                    matched_level = None
                    for level_price, level in sorted_levels:
                        # Check if this position fits within this level's range
                        next_level_up = None
                        for higher_price, _ in sorted_levels:
                            if higher_price > level_price:
                                next_level_up = higher_price
                                break

                        # Calculate price range for this level
                        upper_bound = next_level_up if next_level_up else float('inf')
                        lower_bound = level_price

                        if lower_bound <= entry_price < upper_bound:
                            matched_level = level_price
                            break

                    if matched_level is not None:
                        # Check allocation limits
                        current_allocation = sum(
                            pos['aud_total']
                            for pos in positions_by_level.get(matched_level, [])
                        )
                        max_allocation = settings.INITIAL_INVESTMENT * price_levels[matched_level].allocation

                        if current_allocation + float(order['audtotal']) <= max_allocation:
                            position = {
                                "price_level": matched_level,
                                "entry_price": entry_price,
                                "btc_amount": btc_amount,
                                "timestamp": order['solddate'],
                                "order_id": order.get('id'),
                                "aud_total": float(order.get('audtotal', 0))
                            }

                            # Add to tracking structures
                            if matched_level not in positions_by_level:
                                positions_by_level[matched_level] = []
                            positions_by_level[matched_level].append(position)
                            active_positions.append(position)

                            # Mark level as triggered
                            price_levels[matched_level].triggered = True

                            self.logger.info(
                                f"Matched position: Level {matched_level}, "
                                f"Entry: {entry_price}, Amount: {btc_amount} BTC"
                            )
                        else:
                            self.logger.info(
                                f"Skipping position due to allocation limit: "
                                f"Level {matched_level}, Amount: {btc_amount} BTC"
                            )
                    else:
                        self.logger.info(
                            f"No matching level found for position: "
                            f"Price {entry_price}, Amount: {btc_amount} BTC"
                        )

                except Exception as e:
                    self.logger.error(f"Error processing order: {e}")
                    continue

            # Update state
            self.active_positions = active_positions
            self.save_state()

            # Log summary
            self.logger.info(
                f"Sync completed: {len(active_positions)} positions matched "
                f"across {len(positions_by_level)} price levels"
            )

            return True

        except Exception as e:
            self.logger.error(f"Error in sync_historical_positions: {e}")
            return False

    async def verify_position_triggers(self):
        """Verify triggers match actual positions"""
        try:
            changes = 0
            for price, level in price_levels.items():
                has_position = any(
                    pos['price_level'] == price
                    for pos in self.active_positions
                )

                # Fix mismatched triggers
                if level.triggered != has_position:
                    level.triggered = has_position
                    changes += 1
                    self.logger.info(
                        f"Fixed trigger state for level {price}: {has_position}"
                    )

            if changes:
                self.logger.warning(f"Fixed {changes} mismatched triggers")

            return changes

        except Exception as e:
            self.logger.error(f"Error verifying triggers: {e}")
            return 0

    async def check_drawdown(self, current_price: float) -> bool:
        """Check if current drawdown exceeds maximum allowed"""
        try:
            # Calculate total portfolio value
            portfolio_value = sum(
                pos["btc_amount"] * current_price
                for pos in self.active_positions
            )

            # Update peak value
            self.peak_portfolio_value = max(
                self.peak_portfolio_value,
                portfolio_value
            )

            if self.peak_portfolio_value > 0:
                drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value * 100
                if drawdown > settings.MAX_DRAWDOWN_PCT:
                    self.logger.warning(
                        f"Maximum drawdown exceeded: {drawdown:.2f}% > {settings.MAX_DRAWDOWN_PCT}%"
                    )
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Error in check_drawdown: {e}")
            return False

    async def check_decline_rate(self, current_price: float) -> bool:
        """Check if price is declining too rapidly"""
        try:
            self.last_prices.insert(0, (datetime.now(), current_price))
            self.last_prices = self.last_prices[:10]  # Keep last 10 prices

            if len(self.last_prices) >= 2:
                time_diff = (self.last_prices[0][0] - self.last_prices[-1][0]).total_seconds()
                price_diff = (self.last_prices[0][1] - self.last_prices[-1][1]) / self.last_prices[-1][1] * 100

                decline_rate = (price_diff / time_diff) * 300  # Normalize to 5-minute rate

                if decline_rate < -settings.MAX_DECLINE_RATE_PCT:
                    self.logger.warning(
                        f"Price declining too rapidly: {decline_rate:.2f}% per 5min"
                    )
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Error in check_decline_rate: {e}")
            return False

    async def check_total_exposure(self, new_position_size: float) -> bool:
        """Check if new position would exceed maximum exposure"""
        try:
            current_exposure = sum(
                float(pos.get('btc_amount', 0)) * self.last_btc_price
                for pos in self.active_positions
            )

            if (current_exposure + new_position_size) > settings.MAX_TOTAL_EXPOSURE:
                self.logger.warning(
                    f"Maximum exposure exceeded: {current_exposure + new_position_size:.2f} > {settings.MAX_TOTAL_EXPOSURE}"
                )
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error in check_total_exposure: {e}")
            return False

    async def close_position(self, position: dict, current_price: float):
        """Close a position when stop loss is triggered"""
        try:
            # Log the stop loss trigger
            self.logger.warning(
                f"Closing position due to stop loss: "
                f"Entry: ${position['entry_price']:.2f}, "
                f"Current: ${current_price:.2f}, "
                f"Loss: {((current_price - position['entry_price']) / position['entry_price'] * 100):.2f}%"
            )

            # Place sell order
            try:
                data = {
                    'cointype': 'BTC',
                    'amount': str(position['btc_amount']),
                    'amounttype': 'btc'  # Selling BTC amount
                }
                result = await self.coinspot_api.request('/my/sell/now', data)
                if not result:
                    self.logger.error("Failed to execute stop loss sell order")
                    return  # Don't remove position if sell fails
            except Exception as e:
                self.logger.error(f"Error placing sell order: {e}")
                return

            # Remove from active positions
            self.active_positions.remove(position)

            # Create alert for stop loss
            stop_loss_alert = {
                "type": "stop_loss_triggered",
                "price_level": position['price_level'],
                "entry_price": position['entry_price'],
                "exit_price": current_price,
                "btc_amount": position['btc_amount'],
                "loss_percentage": ((current_price - position['entry_price']) / position['entry_price'] * 100),
                "timestamp": str(datetime.now()),
                "order_id": position['order_id']
            }
            self.price_alerts.append(stop_loss_alert)

            # Save state after position closure
            self.save_state()

            # Send notification
            await self.notifications.send_notification(
                title="⚠️ Stop Loss Triggered",
                message=(
                    f"Position closed:\n"
                    f"Entry: ${position['entry_price']:.2f}\n"
                    f"Exit: ${current_price:.2f}\n"
                    f"Loss: {((current_price - position['entry_price']) / position['entry_price'] * 100):.2f}%\n"
                    f"Amount: {position['btc_amount']:.8f} BTC"
                ),
                priority=5
            )

        except Exception as e:
            self.logger.error(f"Error closing position: {e}")

    async def check_stop_losses(self, current_price: float):
        """Check and execute stop losses if needed"""
        try:
            for position in self.active_positions[:]:  # Copy list to allow modification
                entry_price = float(position['entry_price'])
                loss_pct = (current_price - entry_price) / entry_price * 100

                if loss_pct < -settings.STOP_LOSS_PCT:
                    self.logger.warning(
                        f"Stop loss triggered for position entered at {entry_price}"
                    )
                    # Here you would add logic to sell the position
                    await self.close_position(position, current_price)
        except Exception as e:
            self.logger.error(f"Error in check_stop_losses: {e}")

    async def validate_new_position(self, level: PriceLevel, current_price: float) -> bool:
        """Validate all conditions before taking new position"""
        try:
            # Add validation for negative/zero prices
            if current_price <= 0:
                self.logger.error(f"Invalid price: {current_price}")
                return False

            # FIXED: Add explicit price validation
            # Only allow buying when price is below the level price
            if current_price >= level.price:  # Changed condition
                self.logger.warning(f"Invalid trigger: Price ${current_price} at or above level ${level.price}")
                return False

            # Existing entry condition check
            if not await self.check_entry_conditions(level, current_price):
                return False

            # New protection checks
            checks = await asyncio.gather(
                self.check_drawdown(current_price),
                self.check_decline_rate(current_price),
                self.check_total_exposure(settings.INITIAL_INVESTMENT * level.allocation)
            )

            return all(checks)

        except Exception as e:
            self.logger.error(f"Error in validate_new_position: {e}")
            return False

    async def monitor_prices(self):
        """Monitor BTC prices and execute trading logic"""
        self.logger.info("Starting price monitoring...")

        while self.running:
            try:
                current_price = await self.get_btc_price()
                if current_price > 0:
                    # Check stop losses first
                    await self.check_stop_losses(current_price)

                    # Then check price levels for new positions
                    await self.check_price_levels(current_price)

                    # Update market analysis
                    await self.market_analysis.update_history(current_price)

                # Wait for next interval
                await asyncio.sleep(settings.POLL_INTERVAL)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(settings.POLL_INTERVAL * 2)  # Wait longer on error

    async def get_level_lock(self, price_level: float) -> asyncio.Lock:
        """Get or create lock for price level"""
        if price_level not in self.position_locks:
            self.position_locks[price_level] = asyncio.Lock()
        return self.position_locks[price_level]

    async def check_price_levels(self, current_price: float):
        """Check price levels with proper locking"""
        try:
            current_time = datetime.now()

            for level in price_levels.values():
                level.last_checked = current_time
                buffer_price = level.price - self.price_buffer

                # FIXED: Changed the price validation logic
                # Only proceed if current price is BELOW the level price
                if current_price >= level.price:  # Changed from current_price > level.price
                    self.logger.info(f"Price {current_price} above level {level.price} - skipping")
                    continue

                # Only trigger if price has dropped to or below buffer price
                if not level.triggered and current_price <= buffer_price:
                    # Validate before executing
                    if await self.validate_new_position(level, current_price):
                        allocation_amount = settings.INITIAL_INVESTMENT * level.allocation
                        # Set triggered BEFORE executing to prevent race
                        level.triggered = True
                        try:
                            await self.execute_buy_signal(level, current_price, allocation_amount)
                        except Exception as e:
                            level.triggered = False  # Reset on failure
                            self.logger.error(f"Failed to execute buy signal: {e}")

        except Exception as e:
            self.logger.error(f"Error in check_price_levels: {e}")

    async def check_entry_conditions(self, level: PriceLevel, current_price: float) -> bool:
        """Evaluate entry conditions with improved signal agreement and buffer checking."""
        try:
            # Check basic conditions first
            if len(self.market_analysis.price_history) < self.market_analysis.min_data_points:
                self.logger.info("Not enough price history to evaluate entry conditions.")
                return False

            # Get market conditions
            market_conditions = await self.market_analysis.analyze_market_conditions()

            # Check price buffer condition first
            buffer_price = level.price - self.price_buffer
            if current_price > buffer_price:
                self.logger.info(f"Price {current_price} above buffer price {buffer_price}")
                return False

            # Check volatility thresholds with proper scaling
            volatility = market_conditions['volatility']
            if volatility < settings.MIN_VOLATILITY_THRESHOLD:
                self.logger.info(f"Volatility too low: {volatility:.6f} < {settings.MIN_VOLATILITY_THRESHOLD}")
                return False
            if volatility > settings.MAX_VOLATILITY_THRESHOLD:
                self.logger.info(f"Volatility too high: {volatility:.6f} > {settings.MAX_VOLATILITY_THRESHOLD}")
                return False

            # Check confidence threshold
            confidence = market_conditions['confidence']
            if confidence < settings.CONFIDENCE_THRESHOLD:
                self.logger.info(f"Confidence too low: {confidence:.3f} < {settings.CONFIDENCE_THRESHOLD}")
                return False

            # Improved signal agreement check
            trend_signals = market_conditions['trend_signals']
            agreeing, meets_requirement = self.market_analysis.analyze_trend_signals(
                trend_signals,
                market_conditions['trend']
            )

            if not meets_requirement:
                self.logger.info(f"Insufficient signal agreement: {agreeing} signals agree")
                return False

            # Risk assessment
            risk_level = market_conditions['risk_level']
            if risk_level > settings.RISK_THRESHOLD:
                self.logger.info(f"Risk too high: {risk_level:.3f} > {settings.RISK_THRESHOLD}")
                return False

            # Log complete analysis
            self.logger.info(f"""
            Entry Conditions Analysis:
            - Price: {current_price:.2f} (Buffer: {buffer_price:.2f})
            - Volatility: {volatility:.6f} (min: {settings.MIN_VOLATILITY_THRESHOLD}, max: {settings.MAX_VOLATILITY_THRESHOLD})
            - Confidence: {confidence:.3f} (threshold: {settings.CONFIDENCE_THRESHOLD})
            - Agreeing Signals: {agreeing} (required: {settings.SIGNAL_AGREEMENT_REQUIRED})
            - Risk Level: {risk_level:.3f} (threshold: {settings.RISK_THRESHOLD})
            - Trend Signals: {trend_signals}
            All conditions met: True
            """)

            return True

        except Exception as e:
            self.logger.error(f"Error in check_entry_conditions: {e}")
            return False

    async def place_buy_order(self, allocation_amount: float) -> dict:
        """Place a market buy order via Coinspot API."""
        try:
            data = {
                'cointype': 'BTC',
                'amount': str(round(allocation_amount, 2)),  # Amount in AUD
                'amounttype': 'aud'
            }
            result = await self.coinspot_api.request('/my/buy/now', data)
            if result:
                self.logger.info(f"Buy order successful: {result}")
                return result
            else:
                self.logger.error("Buy order failed.")
                return None
        except Exception as e:
            self.logger.error(f"Error in place_buy_order: {e}")
            return None

    async def get_btc_price(self) -> float:
        """Get current BTC price in AUD from Coinspot using bid/ask pricing"""
        try:
            # Correct the API URL
            url = 'https://www.coinspot.com.au/pubapi/latest'
            headers = {'User-Agent': 'Mozilla/5.0'}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.logger.debug(f"API Response Data: {data}")

                        if data.get('status') == 'ok' and 'prices' in data:
                            btc_data = data['prices'].get('btc', {})

                            # Try to get 'last' price
                            last_price_str = btc_data.get('last')
                            if last_price_str:
                                try:
                                    price = float(last_price_str.replace(',', ''))
                                    if price > 0:
                                        self.last_btc_price = price
                                        await self.market_analysis.update_history(price)
                                        self.save_state()
                                        self.logger.info(f"BTC Price Update: ${price:.2f} AUD")
                                        return price
                                except (ValueError, TypeError) as e:
                                    self.logger.error(f"Failed to parse 'last' price: {e}", exc_info=True)

                            # If 'last' fails, try bid/ask average
                            bid_str = btc_data.get('bid', '0')
                            ask_str = btc_data.get('ask', '0')
                            if bid_str and ask_str:
                                try:
                                    bid = float(bid_str.replace(',', ''))
                                    ask = float(ask_str.replace(',', ''))
                                    if bid > 0 and ask > 0:
                                        price = (bid + ask) / 2
                                        self.last_btc_price = price
                                        await self.market_analysis.update_history(price)
                                        self.save_state()
                                        self.logger.info(f"BTC Price Update: ${price:.2f} AUD (bid/ask avg)")
                                        return price
                                except (ValueError, TypeError) as e:
                                    self.logger.error(f"Failed to parse bid/ask prices: {e}", exc_info=True)

                        if self.last_btc_price:
                            self.logger.warning(f"Using last known price: ${self.last_btc_price:.2f}")
                            return self.last_btc_price

                        self.logger.error("Invalid API response structure")
                        return 0

                    else:
                        self.logger.error(f"HTTP Error: Status {response.status}")
                        if self.last_btc_price:
                            return self.last_btc_price
                        return 0

        except Exception as e:
            self.logger.error(f"Request error: {str(e)}", exc_info=True)
            if self.last_btc_price:
                return self.last_btc_price
            return 0

    async def execute_buy_signal(self, level: PriceLevel, current_price: float, allocation_amount: float):
        """Execute buy signal with proper transaction handling"""
        try:
            # Place order
            order_result = await self.place_buy_order(allocation_amount)
            if not order_result:
                self.logger.error("Order failed - skipping position creation")
                raise Exception("Order placement failed")

            # Create position record
            new_position = {
                "price_level": level.price,
                "entry_price": current_price,
                "btc_amount": float(order_result.get('amount', 0)),
                "timestamp": str(datetime.now()),
                "order_id": order_result.get('id')
            }

            # Update state
            self.active_positions.append(new_position)
            self.save_state()  # Save immediately after state update

            # Create and save alert
            new_alert = {
                "type": "buy_executed",
                "price_level": level.price,
                "entry_price": current_price,
                "allocation": allocation_amount,
                "btc_amount": new_position['btc_amount'],
                "timestamp": new_position['timestamp'],
                "order_id": order_result.get('id')
            }
            self.price_alerts.append(new_alert)

            await self.notifications.send_notification(
                title="BTC Buy Order Executed",
                message=(
                    f"Level: {level.price}\n"
                    f"Amount: {new_position['btc_amount']:.8f} BTC\n"
                    f"Cost: ${allocation_amount:.2f} AUD\n"
                    f"Order ID: {order_result.get('id')}"
                ),
                priority=5
            )

        except Exception as e:
            self.logger.error(f"Error in execute_buy_signal: {e}")
            raise  # Re-raise to trigger trigger reset

    async def cleanup(self):
        """Cleanup resources before shutdown"""
        try:
            self.logger.info("Starting cleanup...")

            # Save final state
            self.save_state()

            # Close API sessions
            if self.notifications:
                await self.notifications.close()

            if self.coinspot_api:
                await self.coinspot_api.close()

            self.logger.info("Cleanup completed successfully")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    async def cleanup_duplicate_positions(self):
        """Clean up any duplicate positions"""
        try:
            # Group positions by price level
            from collections import defaultdict
            positions_by_level = defaultdict(list)

            for pos in self.active_positions:
                positions_by_level[pos['price_level']].append(pos)

            # Keep only the earliest position for each level
            cleaned_positions = []
            for level, positions in positions_by_level.items():
                if len(positions) > 1:
                    self.logger.warning(f"Found {len(positions)} positions at level {level}")
                    # Sort by timestamp and keep earliest
                    positions.sort(key=lambda x: x['timestamp'])
                    cleaned_positions.append(positions[0])

                    # Log removed positions
                    for removed in positions[1:]:
                        self.logger.info(
                            f"Removing duplicate position: Level {level}, "
                            f"Entry: {removed['entry_price']}, "
                            f"Time: {removed['timestamp']}"
                        )
                else:
                    cleaned_positions.append(positions[0])

            # Update active positions
            self.active_positions = cleaned_positions
            self.save_state()

            return len(self.active_positions)

        except Exception as e:
            self.logger.error(f"Error in cleanup_duplicate_positions: {e}")
            return len(self.active_positions)

    async def reconcile_state(self):
        """Reconcile price levels and positions"""
        try:
            changed = False

            # Check for triggered levels without positions
            for price, level in price_levels.items():
                if level.triggered:
                    position_exists = any(
                        pos['price_level'] == price
                        for pos in self.active_positions
                    )

                    if not position_exists:
                        self.logger.warning(
                            f"Found triggered level {price} without position - resetting"
                        )
                        level.triggered = False
                        changed = True

            # Check for positions without triggered levels
            for position in self.active_positions[:]:
                level_exists = any(
                    price == position['price_level']
                    for price in price_levels.keys()
                )

                if not level_exists:
                    self.logger.warning(
                        f"Found position without price level - removing"
                    )
                    self.active_positions.remove(position)
                    changed = True

            if changed:
                self.save_state()
                await self.notifications.send_notification(
                    title="State Reconciled",
                    message="Fixed mismatched positions and triggers",
                    priority=3
                )

            return changed

        except Exception as e:
            self.logger.error(f"Error in reconcile_state: {e}")
            return False

# Configure FastAPI lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global trader
    trader = BTCTrader()

    # Just initialize with clean state
    trader.running = True
    monitoring_task = asyncio.create_task(trader.monitor_prices())
    logger.info("AUD Trading bot started")

    yield

    if trader:
        trader.running = False
        monitoring_task.cancel()
        await trader.cleanup()
        logger.info("AUD Trading bot stopped")

# Instantiate FastAPI app
app = FastAPI(
    title="Bitcoin AUD Trading Bot",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status")
async def get_status():
    if not trader:
        return {"error": "Trader not initialized"}

    current_price = trader.last_btc_price if trader.last_btc_price else await trader.get_btc_price()
    market_conditions = await trader.market_analysis.analyze_market_conditions()

    return {
        "btc_price_aud": current_price,
        "market_conditions": market_conditions,
        "price_levels": {
            price: {
                "allocation": level.allocation,
                "triggered": level.triggered,
                "last_checked": level.last_checked,
                "buffer_price": level.price - trader.price_buffer
            }
            for price, level in price_levels.items()
        },
        "active_positions": trader.active_positions,
        "recent_alerts": trader.price_alerts[-5:] if trader.price_alerts else [],
    }

@app.post("/update_levels")
async def update_levels(new_levels: Dict[float, PriceLevel]):
    """Update price levels"""
    global price_levels
    price_levels = new_levels
    logger.info(f"Updated price levels: {new_levels}")
    return {"status": "success", "new_levels": price_levels}

@app.post("/update_settings")
async def update_settings(new_settings: Settings):
    """Update bot settings and reinitialize analysis"""
    global settings, trader
    settings = new_settings

    # Update market analysis with new settings
    trader.market_analysis.sma_short_period = settings.SMA_SHORT_PERIOD
    trader.market_analysis.sma_long_period = settings.SMA_LONG_PERIOD
    trader.market_analysis.ema_alpha = settings.EMA_ALPHA

    logger.info(f"Updated settings: {settings.dict()}")
    return {"status": "success", "new_settings": settings.dict()}

@app.post("/reset_triggers")
async def reset_triggers():
    """Reset all triggered flags"""
    for level in price_levels.values():
        level.triggered = False
    logger.info("Reset all price triggers")
    return {"status": "success", "message": "All triggers reset"}

@app.get("/sync_positions", response_model=dict)
async def sync_positions():
    """Sync positions with exchange history"""
    try:
        success = await trader.sync_historical_positions()
        if success:
            # Verify triggers after sync
            fixed_triggers = await trader.verify_position_triggers()

            return {
                "status": "success",
                "positions_recovered": len(trader.active_positions),
                "triggers_fixed": fixed_triggers
            }
        else:
            return {"status": "error", "message": "Failed to sync positions"}

    except Exception as e:
        logger.error(f"Error in position sync: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/verify_state", response_model=dict)
async def verify_state():
    """Verify and report on current trading state"""
    try:
        fixed_triggers = await trader.verify_position_triggers()

        return {
            "status": "success",
            "active_positions": len(trader.active_positions),
            "triggered_levels": sum(
                1 for level in price_levels.values()
                if level.triggered
            ),
            "triggers_fixed": fixed_triggers,
            "state_valid": await trader.validate_state_consistency()
        }

    except Exception as e:
        logger.error(f"Error in state verification: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/set_triggers", response_model=dict)
async def set_triggers(trigger_data: Dict[str, bool]):
    """Manually set triggered state for price levels"""
    try:
        updates = 0
        for price_str, should_trigger in trigger_data.items():
            price = float(price_str)
            if price in price_levels:
                price_levels[price].triggered = should_trigger
                updates += 1

        await trader.save_state()

        return {
            "status": "success",
            "updates": updates,
            "current_state": {
                str(price): level.triggered
                for price, level in price_levels.items()
            }
        }
    except Exception as e:
        logger.error(f"Error setting triggers: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/test_api")
async def test_api():
    """Test CoinSpot API connectivity"""
    try:
        # Test basic status
        status_result = await trader.coinspot_api.request('/status', {
            'nonce': int(datetime.now().timestamp() * 1000)
        })

        # Test read-only balances endpoint
        balances_result = await trader.coinspot_api.request('/mybalances', {
            'nonce': int(datetime.now().timestamp() * 1000)
        }, read_only=True)

        # Get detailed debug info
        debug_info = {
            "status_result": status_result,
            "balances_result": balances_result,
            "api_key_length": len(trader.coinspot_api.api_key) if trader.coinspot_api.api_key else 0,
            "api_secret_set": bool(trader.coinspot_api.api_secret)
        }

        return {
            "status": "success",
            "api_connected": True,
            "status_check": status_result,
            "balances_check": balances_result is not None,
            "debug_info": debug_info
        }

    except Exception as e:
        logger.error(f"API test failed: {e}")
        return {
            "status": "error",
            "message": str(e),
            "debug_info": {
                "error_type": type(e).__name__,
                "error_details": str(e)
            }
        }

@app.get("/check_balances")
async def check_balances():
    """Check account balances"""
    try:
        result = await trader.coinspot_api.request('/my/balances', {
            'nonce': int(datetime.now().timestamp() * 1000)
        }, read_only=True)

        if result:
            return {
                "status": "success",
                "balances": result.get('balances'),
                "raw_response": result  # Include raw response for debugging
            }
        else:
            return {
                "status": "error",
                "message": "Failed to get balances",
                "raw_response": None
            }

    except Exception as e:
        logger.error(f"Balance check failed: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/verify_credentials")
async def verify_credentials():
    """Verify API credentials are properly set"""
    return {
        "api_key_set": bool(settings.COINSPOT_API_KEY),
        "api_secret_set": bool(settings.COINSPOT_API_SECRET),
        "api_key_length": len(settings.COINSPOT_API_KEY) if settings.COINSPOT_API_KEY else 0,
        "api_secret_length": len(settings.COINSPOT_API_SECRET) if settings.COINSPOT_API_SECRET else 0
    }

@app.get("/positions")
async def get_positions():
    """Get detailed position information"""
    try:
        # Get current BTC price
        current_price = trader.last_btc_price if trader.last_btc_price else await trader.get_btc_price()

        positions_info = []
        total_btc = 0
        total_aud_value = 0

        for pos in trader.active_positions:
            btc_amount = float(pos['btc_amount'])
            entry_price = float(pos['entry_price'])
            current_value = btc_amount * current_price
            profit_loss = ((current_price - entry_price) / entry_price) * 100

            position_detail = {
                "price_level": pos['price_level'],
                "entry_price": entry_price,
                "current_price": current_price,
                "btc_amount": btc_amount,
                "aud_value": current_value,
                "profit_loss_pct": profit_loss,
                "timestamp": pos['timestamp'],
                "order_id": pos['order_id']
            }

            positions_info.append(position_detail)
            total_btc += btc_amount
            total_aud_value += current_value

        return {
            "status": "success",
            "current_btc_price": current_price,
            "total_positions": len(positions_info),
            "total_btc_amount": total_btc,
            "total_aud_value": total_aud_value,
            "positions": positions_info
        }

    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/debug_sync")
async def debug_sync():
    """Debug endpoint to check raw order history"""
    try:
        # Get raw order history
        data = {
            'cointype': 'BTC',
            'limit': 500
        }

        history = await trader.coinspot_api.request('/my/orders/completed', data, read_only=True)

        # Get current balances
        balances = await trader.coinspot_api.request('/my/balances', {
            'nonce': int(datetime.now().timestamp() * 1000)
        }, read_only=True)

        return {
            "status": "success",
            "raw_history": history,
            "balances": balances,
            "active_positions": trader.active_positions,
            "price_levels": {
                str(price): {
                    "triggered": level.triggered,
                    "allocation": level.allocation
                }
                for price, level in price_levels.items()
            }
        }
    except Exception as e:
        logger.error(f"Debug sync error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/verify_sync")
async def verify_sync():
    """Verify sync status and current positions"""
    try:
        # Get current balances
        balances = await trader.coinspot_api.request('/my/balances', {
            'nonce': int(datetime.now().timestamp() * 1000)
        }, read_only=True)

        # Get BTC balance
        btc_balance = 0
        if balances and 'balances' in balances:
            for balance in balances['balances']:
                if balance.get('cointype') == 'BTC':
                    btc_balance = float(balance.get('balance', 0))
                    break

        return {
            "status": "success",
            "btc_balance": btc_balance,
            "active_positions": len(trader.active_positions),
            "active_positions_detail": trader.active_positions,
            "triggered_levels": sum(1 for level in price_levels.values() if level.triggered),
            "price_levels": len(price_levels),
            "last_sync": getattr(trader, 'last_sync', None)
        }

    except Exception as e:
        logger.error(f"Verify sync error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/initialize_levels")
async def initialize_levels():
    """Initialize price levels based on recent trading history"""
    try:
        global price_levels

        # Get current price
        current_price = await trader.get_btc_price()
        if not current_price:
            return {"status": "error", "message": "Could not get current price"}

        # Define price levels with 5% intervals down from current price
        # Adjust these values based on your strategy
        intervals = [
            (0.95, 0.15),    # 5% down, 15% allocation
            (0.90, 0.20),    # 10% down, 20% allocation
            (0.85, 0.25),    # 15% down, 25% allocation
            (0.80, 0.40)     # 20% down, 40% allocation
        ]

        new_levels = {}
        for interval, allocation in intervals:
            price = round(current_price * interval, 2)
            new_levels[price] = PriceLevel(
                price=price,
                allocation=allocation,
                triggered=False,
                last_checked=datetime.now()
            )

        price_levels = new_levels

        # After setting levels, try to sync positions
        await trader.sync_historical_positions()

        return {
            "status": "success",
            "current_price": current_price,
            "price_levels": {
                str(price): {
                    "allocation": level.allocation,
                    "triggered": level.triggered
                }
                for price, level in price_levels.items()
            }
        }

    except Exception as e:
        logger.error(f"Error initializing levels: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/recover_position")
async def recover_position(position_data: dict):
    """Manually recover a position"""
    try:
        if not price_levels:
            return {"status": "error", "message": "No price levels configured"}

        entry_price = float(position_data['entry_price'])
        btc_amount = float(position_data['btc_amount'])

        # Find closest price level
        closest_level = min(
            price_levels.keys(),
            key=lambda x: abs(float(x) - entry_price)
        )

        new_position = {
            "price_level": closest_level,
            "entry_price": entry_price,
            "btc_amount": btc_amount,
            "timestamp": position_data.get('timestamp', str(datetime.now())),
            "order_id": position_data.get('order_id', f"manual_{int(time.time())}")
        }

        trader.active_positions.append(new_position)
        price_levels[closest_level].triggered = True
        trader.save_state()

        return {
            "status": "success",
            "recovered_position": new_position
        }

    except Exception as e:
        logger.error(f"Error recovering position: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "running", "timestamp": datetime.now()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
