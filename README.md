# rbot

Bitcoin DCA (Dollar Cost Average) trading bot for the Australian market using CoinSpot API. Features technical analysis, risk management, and automated trading strategies.

Licensed under [MIT License](LICENSE)

## Overview

rbot implements an advanced trading strategy for Bitcoin purchases on CoinSpot, featuring:
- Multi-level price entry points
- Technical analysis validation
- Risk management systems
- Real-time monitoring
- Push notifications

## Requirements

- Python 3.8+
- fastapi
- aiohttp
- pydantic
- python-dotenv
- uvicorn
- hmac
- typing
- json

## Setup

1. Clone and install:
```bash
git clone https://github.com/ryandau/rbot.git
cd rbot
pip install -r requirements.txt
```

2. Set up credentials:
- Get your [CoinSpot API credentials](https://www.coinspot.com.au/my/api)
- Create a [ntfy.sh](https://ntfy.sh/) topic for notifications
- Copy `.env.example` to `.env` and update with your credentials

3. Create data directory:
```bash
mkdir -p data
```

4. Run:
```bash
python bot.py
```

## Market Analysis

The bot employs multiple technical indicators and analysis methods:

- Moving Averages
  - Short-term SMA (8 periods)
  - Long-term SMA (21 periods)
  - EMA with configurable alpha
- Momentum Analysis
  - Price momentum calculation
  - Trend strength evaluation
- Statistical Analysis
  - Linear regression
  - R-squared validation
  - Volatility measurement
- Signal Agreement
  - Multiple signal validation
  - Confidence scoring
  - Risk level assessment

## Risk Management

Comprehensive risk management features:

1. Position Protection
   - Maximum drawdown limits
   - Stop-loss automation
   - Rapid decline protection
   - Exposure limits

2. Entry Validation
   - Price buffers
   - Volatility thresholds
   - Technical confirmation
   - Signal agreement requirements

3. Portfolio Management
   - Position sizing
   - Allocation controls
   - Total exposure limits
   - Investment pacing

## API Endpoints

### Core Operations

#### Status Check
```bash
GET http://127.0.0.1:8000/status
```
Returns current market conditions, positions, and bot status.

#### Health Check
```bash
GET http://127.0.0.1:8000/health
```
Confirms bot is running.

### Position Management

#### Get Positions
```bash
GET http://127.0.0.1:8000/positions
```
Returns detailed position information including P/L.

#### Sync Positions
```bash
GET http://127.0.0.1:8000/sync_positions
```
Synchronizes positions with exchange history.

#### Recover Position
```bash
POST http://127.0.0.1:8000/recover_position
```
Manually recover a position into tracking.

### Configuration

#### Update Price Levels
```bash
POST http://127.0.0.1:8000/update_levels
Content-Type: application/json

{
    "165000.0": {"price": 165000.0, "allocation": 0.15, "triggered": false},
    "160000.0": {"price": 160000.0, "allocation": 0.15, "triggered": false},
    "155000.0": {"price": 155000.0, "allocation": 0.15, "triggered": false}
}
```

#### Update Settings
```bash
POST http://127.0.0.1:8000/update_settings
Content-Type: application/json

{
    "POLL_INTERVAL": 20,
    "INITIAL_INVESTMENT": 500.0,
    "RISK_THRESHOLD": 0.65
    // ... other settings
}
```

#### Reset Triggers
```bash
POST http://127.0.0.1:8000/reset_triggers
```
Resets all price level triggers.

### Monitoring

#### Check Balances
```bash
GET http://127.0.0.1:8000/check_balances
```
Returns current account balances.

#### Verify Credentials
```bash
GET http://127.0.0.1:8000/verify_credentials
```
Validates API credentials.

#### Verify State
```bash
GET http://127.0.0.1:8000/verify_state
```
Checks bot state consistency.

#### Debug Sync
```bash
GET http://127.0.0.1:8000/debug_sync
```
Detailed synchronization information.

## Configuration

The bot can be configured through:

1. Environment Variables
   - Set in `.env` file
   - System environment variables
   - Runtime updates via API

2. Price Levels
   - Multiple entry points
   - Individual allocations
   - Buffer zones
   - Trigger status

3. Trading Parameters
   - Investment amounts
   - Risk thresholds
   - Technical indicators
   - Protection settings

## Data Storage

The bot maintains state in the `data/` directory:
- `trader_state.json`: Current bot state
- `price_history.json`: Historical price data

## Notifications

Real-time notifications via ntfy.sh for:
- Trade executions
- Stop losses
- Error conditions
- State changes

## Security Notes

- API keys stored locally
- No external data transmission except to CoinSpot
- Secure notifications via ntfy.sh
- No sensitive data logging

## Disclaimer

This software is provided "as is" without warranty. The author(s) are not liable for any:
- Trading losses or financial decisions
- Technical malfunctions
- Data inaccuracies
- Direct or indirect damages

This is not financial advice. Use at your own risk.

## License

MIT License - see [LICENSE](LICENSE) for details
