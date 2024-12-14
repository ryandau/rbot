// Card components
const Card = ({ children, className = "" }) => 
  React.createElement("div", { 
    className: `bg-white shadow rounded-lg p-4 ${className}`
  }, children);

const CardHeader = ({ children }) => 
  React.createElement("div", { className: "mb-4" }, children);

const CardTitle = ({ children, className = "" }) => 
  React.createElement("h2", { 
    className: `text-lg font-semibold ${className}`
  }, children);

const CardContent = ({ children }) => 
  React.createElement("div", null, children);

const Alert = ({ children, variant = "default" }) => {
  const variantClasses = {
    default: "bg-blue-100 text-blue-800",
    destructive: "bg-red-100 text-red-800",
    success: "bg-green-100 text-green-800"
  };
  
  return React.createElement("div", {
    className: `p-4 rounded-lg ${variantClasses[variant]}`
  }, children);
};

const AlertTitle = ({ children }) => 
  React.createElement("h3", { 
    className: "font-semibold mb-1" 
  }, children);

const AlertDescription = ({ children }) => 
  React.createElement("div", { 
    className: "text-sm" 
  }, children);

// Icons as SVG components
const TrendingUp = () => 
  React.createElement("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: 24,
    height: 24,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 2,
    strokeLinecap: "round",
    strokeLinejoin: "round",
    className: "inline-block"
  }, [
    React.createElement("polyline", { 
      key: "line1",
      points: "23 6 13.5 15.5 8.5 10.5 1 18" 
    }),
    React.createElement("polyline", { 
      key: "line2",
      points: "17 6 23 6 23 12" 
    })
  ]);

const TrendingDown = () => 
  React.createElement("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: 24,
    height: 24,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 2,
    strokeLinecap: "round",
    strokeLinejoin: "round",
    className: "inline-block"
  }, [
    React.createElement("polyline", { 
      key: "line1",
      points: "23 18 13.5 8.5 8.5 13.5 1 6" 
    }),
    React.createElement("polyline", { 
      key: "line2",
      points: "17 18 23 18 23 12" 
    })
  ]);

const Activity = () => 
  React.createElement("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: 24,
    height: 24,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 2,
    strokeLinecap: "round",
    strokeLinejoin: "round",
    className: "inline-block"
  }, 
    React.createElement("polyline", { 
      points: "22 12 18 12 15 21 9 3 6 12 2 12" 
    })
  );

const LiveTradingDashboard = () => {
  const [data, setData] = React.useState(null);
  const [error, setError] = React.useState(null);

  React.useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/status');
        const newData = await response.json();
        setData(newData);
      } catch (err) {
        console.error('Fetch error:', err);
        setError('Failed to fetch data');
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 20000);
    return () => clearInterval(interval);
  }, []);

  const formatPrice = (price) => {
    if (!price && price !== 0) return '-';
    return new Intl.NumberFormat('en-AU', {
      style: 'currency',
      currency: 'AUD',
      minimumFractionDigits: 2
    }).format(price);
  };

  const getPriceColor = (currentPrice, levelPrice) => {
    const diff = currentPrice - levelPrice;
    if (Math.abs(diff) < 1000) return 'text-yellow-500';
    return diff > 0 ? 'text-green-500' : 'text-blue-500';
  };

  if (error) {
    return React.createElement("div", { className: "max-w-4xl mx-auto p-4" },
      React.createElement(Alert, { variant: "destructive" }, [
        React.createElement(AlertTitle, { key: "title" }, "Error"),
        React.createElement(AlertDescription, { key: "desc" }, error)
      ])
    );
  }

  if (!data) {
    return React.createElement("div", { 
      className: "max-w-4xl mx-auto p-4 text-center" 
    }, [
      React.createElement(Activity, { key: "icon" }),
      React.createElement("p", { 
        key: "text",
        className: "mt-2" 
      }, "Loading dashboard data...")
    ]);
  }

  const levelData = Object.entries(data.price_levels || {})
    .map(([price, details]) => ({
      price: parseFloat(price),
      ...details
    }))
    .sort((a, b) => b.price - a.price);

  return React.createElement("div", {
    className: "max-w-4xl mx-auto p-4 space-y-4"
  }, [
    // Market Status Alert
    React.createElement(Alert, {
      key: "status",
      variant: data.market_conditions.trend === "downward" ? "destructive" : "default"
    }, [
      React.createElement("div", {
        key: "header",
        className: "flex items-center gap-2"
      }, [
        data.market_conditions.trend === "downward" ? 
          React.createElement(TrendingDown, { key: "icon" }) : 
          React.createElement(TrendingUp, { key: "icon" }),
        React.createElement(AlertTitle, { key: "title" }, 
          `Market Status: ${data.market_conditions.trend.toUpperCase()}`
        )
      ]),
      React.createElement(AlertDescription, { key: "price" },
        `Current Price: ${formatPrice(data.btc_price_aud)}`
      )
    ]),

    // Market Conditions and Signal Analysis Grid
    React.createElement("div", {
      key: "grid",
      className: "grid grid-cols-1 md:grid-cols-2 gap-4"
    }, [
      // Market Conditions Card
      React.createElement(Card, { key: "conditions" }, [
        React.createElement(CardHeader, { key: "header" },
          React.createElement(CardTitle, { 
            className: "flex items-center gap-2" 
          }, [
            React.createElement(Activity, { key: "icon" }),
            "Market Conditions"
          ])
        ),
        React.createElement(CardContent, { key: "content" },
          React.createElement("div", { 
            className: "grid grid-cols-2 gap-4" 
          }, [
            // Confidence
            React.createElement("div", { 
              key: "confidence",
              className: "p-2 border rounded"
            }, [
              React.createElement("div", { 
                key: "label",
                className: "text-sm text-gray-600" 
              }, "Confidence"),
              React.createElement("div", {
                key: "value",
                className: `text-lg font-semibold ${
                  data.market_conditions.confidence >= 0.42 ? 
                    'text-green-500' : 'text-red-500'
                }`
              }, `${(data.market_conditions.confidence * 100).toFixed(1)}%`)
            ]),
            // Volatility
            React.createElement("div", {
              key: "volatility",
              className: "p-2 border rounded"
            }, [
              React.createElement("div", {
                key: "label",
                className: "text-sm text-gray-600"
              }, "Volatility"),
              React.createElement("div", {
                key: "value",
                className: "text-lg font-semibold"
              }, `${(data.market_conditions.volatility * 100).toFixed(4)}%`)
            ])
          ])
        )
      ]),

      // Signal Analysis Card
      React.createElement(Card, { key: "signals" }, [
        React.createElement(CardHeader, { key: "header" },
          React.createElement(CardTitle, {}, "Signal Analysis")
        ),
        React.createElement(CardContent, { key: "content" },
          React.createElement("div", { className: "space-y-2" },
            Object.entries(data.market_conditions.trend_signals)
              .map(([signal, value]) =>
                React.createElement("div", {
                  key: signal,
                  className: "flex justify-between p-2 border rounded"
                }, [
                  React.createElement("span", {
                    key: "label",
                    className: "font-medium"
                  }, signal.replace('_signal', '').toUpperCase()),
                  React.createElement("span", {
                    key: "value",
                    className: value === 1 ? 'text-green-500' : 
                              value === -1 ? 'text-red-500' : 
                              'text-gray-500'
                  }, value === 1 ? '↑' : value === -1 ? '↓' : '−')
                ])
              )
          )
        )
      ])
    ]),

    // Entry Levels Card
    React.createElement(Card, { key: "levels" }, [
      React.createElement(CardHeader, { key: "header" },
        React.createElement(CardTitle, {}, "Entry Levels")
      ),
      React.createElement(CardContent, { key: "content" },
        React.createElement("div", { className: "space-y-2" },
          levelData.map(level =>
            React.createElement("div", {
              key: level.price,
              className: "flex justify-between p-2 border rounded"
            }, [
              React.createElement("div", { key: "left" }, [
                React.createElement("span", {
                  key: "price",
                  className: "font-medium"
                }, formatPrice(level.price)),
                React.createElement("span", {
                  key: "allocation",
                  className: "ml-2 text-gray-500"
                }, `(${(level.allocation * 100)}%)`)
              ]),
              React.createElement("div", {
                key: "right",
                className: getPriceColor(data.btc_price_aud, level.price)
              }, formatPrice(data.btc_price_aud - level.price))
            ])
          )
        )
      )
    ])
  ]);
};

// Render the app
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(React.createElement(LiveTradingDashboard));
