# rbot Trading Dashboard

A real-time React dashboard for monitoring the rbot Bitcoin trading bot.

<img width="884" alt="image" src="https://github.com/user-attachments/assets/3ae94358-11d4-4c84-a157-0d3d16c75e45" />

## Overview

The dashboard provides:
- Real-time price monitoring
- Technical analysis visualization
- Position tracking
- Market conditions monitoring
- Trading signals analysis

## Prerequisites

- Node.js >= 14.0.0
- npm
- Access to rbot API (running on 127.0.0.1:8000)
- SSH access for tunneling

## Installation

1. Create and navigate to project directory:
```bash
mkdir -p ~/trading-dashboard
cd ~/trading-dashboard
```

2. Clone or create required files:
```bash
# Create directories
mkdir public

# Create files
touch server.js
touch public/index.html
touch public/app.js
touch package.json
```

3. Copy the provided files:
- `server.js` - Main server file
- `public/index.html` - Dashboard HTML template
- `public/app.js` - React application
- `package.json` - Project configuration

4. Install dependencies:
```bash
npm install
```

## SSH Tunnel Setup

To access the bot API running on 127.0.0.1:8000, set up an SSH tunnel:

1. On your local machine:
```bash
ssh -L 8000:127.0.0.1:8000 username@your-server
```

2. In a separate terminal, set up tunnel for dashboard:
```bash
ssh -L 3000:127.0.0.1:3000 username@your-server
```

## Running the Dashboard

1. Start the dashboard server:
```bash
npm start
```

2. Access the dashboard:
```
http://127.0.0.1:3000
```

## File Structure
```
trading-dashboard/
├── server.js           # Express server configuration
├── package.json        # Project dependencies
├── node_modules/       # Installed packages
└── public/            
    ├── index.html     # Dashboard HTML template
    └── app.js         # React application
```

## Configuration

### server.js
- Port: 3000
- CORS enabled for all origins
- Serves static files from public directory

### package.json
```json
{
  "type": "module",
  "scripts": {
    "start": "node server.js",
    "dev": "nodemon server.js"
  }
}
```

## Development

For development with auto-reload:
```bash
npm install --save-dev nodemon
npm run dev
```

## API Integration

The dashboard connects to the rbot API running on `http://127.0.0.1:8000` with endpoints:
- `/status` - Current market conditions and positions
- `/positions` - Active trading positions
- Other endpoints as needed

## Troubleshooting

1. Connection Issues
```bash
# Check SSH tunnels
lsof -i :8000
lsof -i :3000

# Restart tunnels if needed
```

2. API Access
```bash
# Test API access
curl http://127.0.0.1:8000/status
```

3. Common Issues:
- "Cannot find module" - Run `npm install`
- Connection refused - Check SSH tunnels
- CORS errors - Verify server.js CORS settings
- React loading issues - Check browser console

## License

MIT License - see LICENSE for details
