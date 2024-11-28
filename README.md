# rbot

Bitcoin trading bot for the Australian market using CoinSpot API.

## Setup

1. Clone and install:
```bash
git clone https://github.com/ryandau/rbot.git
cd rbot
pip install -r requirements.txt
```

2. Create `.env` file:
```env
# Price levels
POLL_INTERVAL=30
INITIAL_INVESTMENT=500.0
RISK_THRESHOLD=0.65
MAX_HISTORY_DAYS=14
PRICE_BUFFER=300.0
NTFY_TOPIC=your_ntfy_topic

# API keys 
COINSPOT_API_KEY=your_api_key
COINSPOT_API_SECRET=your_api_secret

# Trading thresholds
MIN_VOLATILITY_THRESHOLD=0.0002
MAX_VOLATILITY_THRESHOLD=0.02 
CONFIDENCE_THRESHOLD=0.42
SIGNAL_AGREEMENT_REQUIRED=2

# Protection settings
MAX_DRAWDOWN_PCT=10.0
MAX_DECLINE_RATE_PCT=2.5 
MAX_TOTAL_EXPOSURE=750.0
PRICE_VALIDATION_THRESHOLD=1.0
STOP_LOSS_PCT=5.0
```

## Run

```bash
python bot.py
```

## API Endpoints

### Get Status
```bash
curl http://127.0.0.1:8000/status
```

### Update Price Levels
```bash
curl -X POST http://127.0.0.1:8000/update_levels \
-H "Content-Type: application/json" \
-d '{
    "147500.0": {"price": 147500.0, "allocation": 0.10, "triggered": true},
    "145000.0": {"price": 145000.0, "allocation": 0.15, "triggered": true},
    "142500.0": {"price": 142500.0, "allocation": 0.15, "triggered": true},
    "141500.0": {"price": 141500.0, "allocation": 0.15},
    "140000.0": {"price": 140000.0, "allocation": 0.15},
    "137500.0": {"price": 137500.0, "allocation": 0.15},
    "135000.0": {"price": 135000.0, "allocation": 0.15}
}'
```

### Update Settings
```bash
curl -X POST http://127.0.0.1:8000/update_settings \
-H "Content-Type: application/json" \
-d '{
    "POLL_INTERVAL": 30,
    "INITIAL_INVESTMENT": 500.0,
    "RISK_THRESHOLD": 0.65,
    "MAX_HISTORY_DAYS": 14,
    "PRICE_BUFFER": 300.0,
    "MIN_VOLATILITY_THRESHOLD": 0.0002,
    "MAX_VOLATILITY_THRESHOLD": 0.02,
    "CONFIDENCE_THRESHOLD": 0.42,
    "SIGNAL_AGREEMENT_REQUIRED": 2,
    "MAX_DRAWDOWN_PCT": 10.0,
    "MAX_DECLINE_RATE_PCT": 2.5,
    "MAX_TOTAL_EXPOSURE": 750.0,
    "PRICE_VALIDATION_THRESHOLD": 1.0,
    "STOP_LOSS_PCT": 5.0
}'
```

### Reset Triggers
```bash
curl -X POST http://127.0.0.1:8000/reset_triggers
```

### Health Check
```bash
curl http://127.0.0.1:8000/health
```

## Features

* Real-time BTC/AUD price monitoring
* Technical analysis with multiple indicators:
  - SMA, EMA, momentum
  - Linear regression analysis
  - Volatility calculations
* Configurable price levels and entry points
* Risk management system:
  - Maximum drawdown protection
  - Decline rate monitoring
  - Stop-loss automation
  - Exposure limits
* Real-time notifications via ntfy.sh
* State persistence
* API endpoints for:
  - Status monitoring
  - Settings updates
  - Price level management
  - Health checks
* CoinSpot API integration
* Automatic position tracking

## Disclaimer

This software is provided "as is" without warranty. The author(s) are not liable for any:
- Trading losses or financial decisions
- Technical malfunctions
- Data inaccuracies
- Direct or indirect damages

This is not financial advice. Use at your own risk.
