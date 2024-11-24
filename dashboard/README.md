# React Dashboard

![image](https://github.com/user-attachments/assets/cb2ae3f0-3309-41a9-a1f9-13021b32ca2a)

## Description
This is a React dashboard for rbot served by an Express.js backend that provides static file serving and CORS support. The server is configured to serve a single-page application (SPA) by redirecting all routes to index.html.

## Prerequisites
- Node.js (version 14.0.0 or higher)
- npm (Node Package Manager)

## Project Structure
```
rbot/
├── dashboard/
│   └── public/
│       ├── app.js
│       └── index.html
└── server.js
```

## Installation

1. Install dependencies:
```bash
npm install
```

Required dependencies:
- express
- path
- url

## Configuration

The server is configured with:
- Port: 3000 (configurable via PORT constant)
- CORS: Enabled for all origins
- Static file serving from 'public' directory
- All routes redirect to index.html (SPA support)

## Usage

1. Start the server:
```bash
node server.js
```

2. Access the dashboard:
- Local: http://127.0.0.1:3000
- Network: http://<your-ip-address>:3000

## Server Features

### Static File Serving
- Serves static files from the `public` directory
- Configured using `express.static('public')`

### CORS Support
- Allows cross-origin requests from any domain
- Configures necessary CORS headers:
  - Access-Control-Allow-Origin: *
  - Access-Control-Allow-Headers: Origin, X-Requested-With, Content-Type, Accept

### SPA Support
- Redirects all routes to index.html
- Enables client-side routing for React application

## Development

### Modifying the Server
- Port can be changed by updating the `PORT` constant
- CORS settings can be modified in the middleware
- Static file directory can be changed in `express.static()`

### Adding Routes
Additional Express routes should be added before the catch-all route:
```javascript
// Add new routes here
app.get('/api/example', (req, res) => {
  // Route handler
});

// Keep this last
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});
```
