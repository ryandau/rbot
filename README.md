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
POLL_INTERVAL=20
INITIAL_INVESTMENT=500
RISK_THRESHOLD=0.65
MAX_HISTORY_DAYS=14
PRICE_BUFFER=200
NTFY_TOPIC=your_ntfy_topic
COINSPOT_API_KEY=your_api_key
COINSPOT_API_SECRET=your_api_secret
MIN_VOLATILITY_THRESHOLD=0.0002
MAX_VOLATILITY_THRESHOLD=0.015
CONFIDENCE_THRESHOLD=0.42
SIGNAL_AGREEMENT_REQUIRED=3
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
  "70000.0": {
    "price": 70000.0,
    "allocation": 0.25
  },
  "65000.0": {
    "price": 65000.0,
    "allocation": 0.5
  }
}'
```

### Update Settings
```bash
curl -X POST http://127.0.0.1:8000/update_settings \
-H "Content-Type: application/json" \
-d '{
  "POLL_INTERVAL": 30,
  "INITIAL_INVESTMENT": 1000,
  "RISK_THRESHOLD": 0.6,
  "PRICE_BUFFER": 400,
  "MIN_VOLATILITY_THRESHOLD": 0.002,
  "MAX_VOLATILITY_THRESHOLD": 0.025,
  "CONFIDENCE_THRESHOLD": 0.4
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

- Real-time BTC/AUD price monitoring
- Technical analysis with multiple indicators
- Configurable price levels and entry points
- Risk management system
- Real-time notifications via ntfy.sh
- State persistence

## Disclaimer

This bot is provided for educational and research purposes only. Cryptocurrency trading carries significant risks, and you should never invest more than you can afford to lose. The creators of this bot are not responsible for any financial losses incurred through its use.
