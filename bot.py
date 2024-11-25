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

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Settings(BaseModel):
    POLL_INTERVAL: int = int(os.getenv('POLL_INTERVAL', 20))
    INITIAL_INVESTMENT: float = float(os.getenv('INITIAL_INVESTMENT', 500))  # AUD
    RISK_THRESHOLD: float = float(os.getenv('RISK_THRESHOLD', 0.7))
    MAX_HISTORY_DAYS: int = int(os.getenv('MAX_HISTORY_DAYS', 14))
    PRICE_BUFFER: float = float(os.getenv('PRICE_BUFFER', 500))  # AUD
    NTFY_TOPIC: str = os.getenv('NTFY_TOPIC')
    COINSPOT_API_KEY: str = os.getenv('COINSPOT_API_KEY')
    COINSPOT_API_SECRET: str = os.getenv('COINSPOT_API_SECRET')
    
    # New trading thresholds
    MIN_VOLATILITY_THRESHOLD: float = float(os.getenv('MIN_VOLATILITY_THRESHOLD', 0.001))
    MAX_VOLATILITY_THRESHOLD: float = float(os.getenv('MAX_VOLATILITY_THRESHOLD', 0.03))
    CONFIDENCE_THRESHOLD: float = float(os.getenv('CONFIDENCE_THRESHOLD', 0.35))
    SIGNAL_AGREEMENT_REQUIRED: int = int(os.getenv('SIGNAL_AGREEMENT_REQUIRED', 2))

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
        self.history_file = 'price_history.json'

        # Configuration for trend analysis
        self.sma_short_period = 5
        self.sma_long_period = 14
        self.ema_alpha = 0.2

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
        """Update price history with new price point."""
        if price <= 0:
            self.logger.error(f"Invalid price for history update: {price}")
            return

        current_time = datetime.now()
        time_passed = (current_time - self.last_update).total_seconds() if self.last_update else None

        # Update if: first price, price changed, or enough time passed
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
            
            # Save after update
            self.save_history()
            
            self.logger.info(
                f"Price history updated - New: {price}, "
                f"Points: {len(self.price_history)}, "
                f"Time passed: {time_passed:.1f if time_passed else 0}s, "
                f"History: {self.price_history[:5]}"
            )
        else:
            self.logger.debug(
                f"Update skipped - Current: {price}, "
                f"Last: {self.last_price}, "
                f"Time passed: {time_passed:.1f if time_passed else 0}s"
            )

    def calculate_sma(self, period: int) -> Optional[float]:
        """Calculate Simple Moving Average with proper error handling"""
        try:
            if not self.price_history or len(self.price_history) < period:
                self.logger.debug(f"Insufficient data for SMA({period}): {len(self.price_history)} points")
                return None
                
            sma = sum(self.price_history[:period]) / period
            self.logger.debug(f"Calculated SMA({period}): {sma:.2f}")
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
        self.api_endpoint = 'https://www.coinspot.com.au/api/v2'
        self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session:
            await self.session.close()

    async def request(self, path: str, data: dict) -> dict:
        """Make an authenticated request to the Coinspot API."""
        try:
            nonce = int(datetime.now().timestamp() * 1000)
            data['nonce'] = nonce
            post_data = json.dumps(data, separators=(',', ':'))

            # Create the HMAC signature
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

            url = f"{self.api_endpoint}{path}"

            async with self.session.post(url, data=post_data, headers=headers) as response:
                response_text = await response.text()
                self.logger.info(f"API Response: {response_text}")
                if response.status == 200:
                    result = json.loads(response_text)
                    if result.get('status') == 'ok':
                        return result
                    else:
                        self.logger.error(f"API Error: {result.get('message')}")
                        return None
                else:
                    self.logger.error(f"HTTP Error: Status {response.status}")
                    return None
        except Exception as e:
            self.logger.error(f"Error in CoinspotAPI request: {e}")
            return None

class BTCTrader:
    def __init__(self):
        self.position_locks = {}
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.last_btc_price = None
        
        # File paths for persistence
        self.state_file = 'trader_state.json'
        self.history_file = 'price_history.json'
        
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
        
        # Load previous state
        self.load_state()
        
        # Initialize components after loading state
        self.market_analysis = MarketAnalysis(
            settings.MAX_HISTORY_DAYS,
            price_history=self.price_history if hasattr(self, 'price_history') else []
        )
        self.notifications = NotificationManager(settings.NTFY_TOPIC)
        self.price_buffer = settings.PRICE_BUFFER
        
        # Initialize with loaded state or defaults
        self.price_alerts = getattr(self, 'price_alerts', [])
        self.active_positions = getattr(self, 'active_positions', [])
        self.last_btc_price = getattr(self, 'last_btc_price', None)

        # Initialize CoinspotAPI with credentials
        self.coinspot_api = CoinspotAPI(settings.COINSPOT_API_KEY, settings.COINSPOT_API_SECRET)

    async def get_level_lock(self, price_level: float) -> asyncio.Lock:
        """Get or create lock for price level"""
        if price_level not in self.position_locks:
            self.position_locks[price_level] = asyncio.Lock()
        return self.position_locks[price_level]

    async def monitor_prices(self):
        """Monitor BTC prices and check entry conditions"""
        self.logger.info("Starting price monitoring...")

        while self.running:
            try:
                current_price = await self.get_btc_price()
                if current_price > 0:
                    await self.check_price_levels(current_price)
                await asyncio.sleep(settings.POLL_INTERVAL)
                self.logger.info(
                    f"Monitoring - Current Price: ${current_price:.2f}, "
                    f"History Points: {len(self.market_analysis.price_history)}"
                )
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(settings.POLL_INTERVAL * 2)  # Back off on error

    async def check_price_levels(self, current_price: float):
        """Check price levels with proper locking"""
        try:
            current_time = datetime.now()
            
            for level in price_levels.values():
                level.last_checked = current_time
                buffer_price = level.price - self.price_buffer

                if not level.triggered and current_price <= buffer_price:
                    # Get lock for this level
                    lock = await self.get_level_lock(level.price)
                    
                    async with lock:  # Ensure atomic operation
                        # Recheck conditions after acquiring lock
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

    def save_state(self):
        """Save complete trading state including price history"""
        try:
            state = {
                'price_history': self.market_analysis.price_history,
                'last_btc_price': self.last_btc_price,
                'active_positions': self.active_positions,
                'price_alerts': self.price_alerts,
                'last_update': str(datetime.now()),
                'total_price_points': self.market_analysis.total_price_points
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f)
                
            self.logger.info(f"State saved successfully: {len(self.market_analysis.price_history)} price points")
            
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")

    def load_state(self):
        """Load previous trading state"""
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                
            self.price_history = state.get('price_history', [])
            self.last_btc_price = state.get('last_btc_price')
            self.active_positions = state.get('active_positions', [])
            self.price_alerts = state.get('price_alerts', [])
            
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

    async def get_btc_price(self) -> float:
        """Get current BTC price in AUD from Coinspot using bid/ask pricing"""
        try:
            url = 'https://www.coinspot.com.au/pubapi/v2/latest'
            headers = {'User-Agent': 'Mozilla/5.0'}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('status') == 'ok' and 'prices' in data:
                            btc_data = data['prices'].get('btc', {})
                            
                            # Get 'last' price first, fallback to bid/ask average
                            if 'last' in btc_data:
                                try:
                                    price = float(btc_data['last'].replace(',', ''))
                                    if price > 0:
                                        self.last_btc_price = price
                                        await self.market_analysis.update_history(price)
                                        self.save_state()
                                        self.logger.info(f"BTC Price Update: ${price:.2f} AUD")
                                        return price
                                except (ValueError, TypeError) as e:
                                    self.logger.debug(f"Failed to parse 'last' price: {e}")
                            
                            # If 'last' fails, try bid/ask average
                            try:
                                bid = float(btc_data.get('bid', '0').replace(',', ''))
                                ask = float(btc_data.get('ask', '0').replace(',', ''))
                                if bid > 0 and ask > 0:
                                    price = (bid + ask) / 2
                                    self.last_btc_price = price
                                    await self.market_analysis.update_history(price)
                                    self.save_state()
                                    self.logger.info(f"BTC Price Update: ${price:.2f} AUD (bid/ask avg)")
                                    return price
                            except (ValueError, TypeError) as e:
                                self.logger.debug(f"Failed to parse bid/ask prices: {e}")
                        
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
            self.logger.error(f"Request error: {str(e)}")
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

# Configure FastAPI lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global trader
    trader = BTCTrader()
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
    """Update bot settings"""
    global settings
    settings = new_settings
    logger.info(f"Updated settings: {settings.dict()}")
    return {"status": "success", "new_settings": settings.dict()}

@app.post("/reset_triggers")
async def reset_triggers():
    """Reset all triggered flags"""
    for level in price_levels.values():
        level.triggered = False
    logger.info("Reset all price triggers")
    return {"status": "success", "message": "All triggers reset"}

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "running", "timestamp": datetime.now()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
