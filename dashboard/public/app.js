// Card components
const Card = ({ children, className = "" }) => (
  <div className={`card ${className}`}>{children}</div>
);

const CardHeader = ({ children }) => (
  <div className="mb-4">{children}</div>
);

const CardTitle = ({ children, className = "" }) => (
  <h2 className={`text-lg font-semibold ${className}`}>{children}</h2>
);

const CardContent = ({ children }) => (
  <div>{children}</div>
);

const Alert = ({ children, variant = "default" }) => (
  <div className={`alert alert-${variant}`}>{children}</div>
);

const AlertTitle = ({ children }) => (
  <h3 className="font-semibold mb-1">{children}</h3>
);

const AlertDescription = ({ children }) => (
  <div>{children}</div>
);

// Icons
const TrendingUp = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline>
    <polyline points="17 6 23 6 23 12"></polyline>
  </svg>
);
          
const TrendingDown = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="23 18 13.5 8.5 8.5 13.5 1 6"></polyline>
    <polyline points="17 18 23 18 23 12"></polyline>
  </svg>
);
                  
const Activity = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
  </svg>        
);              
                  
const LiveTradingDashboard = () => {
  const [data, setData] = React.useState(null);
  const [error, setError] = React.useState(null);
                
  React.useEffect(() => {
    const fetchData = async () => {
      try {     
        const response = await fetch('http://127.0.0.1:8000/status');
        const newData = await response.json();
        console.log('Fetched data:', newData);
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
          
  if (!data) return <div className="text-center p-4">Loading...</div>;
  if (error) return <div className="text-center p-4 text-red-500">Error: {error}</div>;
                
  const formatPrice = (price) => price.toLocaleString('en-AU', {
    style: 'currency',
    currency: 'AUD',
    minimumFractionDigits: 2
  });

  const getPriceColor = (currentPrice, levelPrice) => {
    const diff = currentPrice - levelPrice;
    if (Math.abs(diff) < 1000) return 'text-yellow-500';
    return diff > 0 ? 'text-green-500' : 'text-blue-500';
  };    
      
  const levelData = Object.entries(data.price_levels).map(([price, details]) => ({
    price: parseFloat(price),
    ...details
  }));    
        
  return (
    <div className="space-y-4 max-w-4xl mx-auto">
      <Alert variant={data.market_conditions.trend === "downward" ? "destructive" : "default"}>
        <div className="flex items-center gap-2">
          {data.market_conditions.trend === "downward" ?
            <TrendingDown /> : 
            <TrendingUp />
          }         
          <AlertTitle>Market Status: {data.market_conditions.trend.toUpperCase()}</AlertTitle>
        </div>  
        <AlertDescription>
          Current Price: {formatPrice(data.btc_price_aud)}
        </AlertDescription>
      </Alert>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity />
              Market Conditions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              <div className="p-2 border rounded">
                <div className="text-sm text-gray-600">Confidence</div>
                <div className={`text-lg font-semibold ${
                  data.market_conditions.confidence >= 0.42 ? 'text-green-500' : 'text-red-500'
                }`}>
                  {(data.market_conditions.confidence * 100).toFixed(1)}%
                </div>
              </div>
              <div className="p-2 border rounded">
                <div className="text-sm text-gray-600">Volatility</div>
                <div className="text-lg font-semibold">
                  {(data.market_conditions.volatility * 100).toFixed(4)}%
                </div>
              </div>
              <div className="p-2 border rounded">
                <div className="text-sm text-gray-600">Risk Level</div>
                <div className="text-lg font-semibold">
                  {(data.market_conditions.risk_level * 100).toFixed(1)}%
                </div>
              </div>
              <div className="p-2 border rounded">
                <div className="text-sm text-gray-600">R-Squared</div>
                <div className="text-lg font-semibold">
                  {(data.market_conditions.indicators.r_squared * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Signal Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {Object.entries(data.market_conditions.trend_signals).map(([signal, value]) => (
                <div key={signal} className="flex justify-between p-2 border rounded">
                  <span className="font-medium">
                    {signal.replace('_signal', '').toUpperCase()}
                  </span>
                  <span className={value === 1 ? 'text-green-500' : 'text-red-500'}>
                    {value === 1 ? '↑' : '↓'}
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Entry Levels</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {levelData.sort((a, b) => b.price - a.price).map((level) => (
              <div key={level.price} className="flex justify-between p-2 border rounded">
                <div>
                  <span className="font-medium">{formatPrice(level.price)}</span>
                  <span className="ml-2 text-gray-500">
                    ({(level.allocation * 100)}%)
                  </span>
                </div>
                <div className={getPriceColor(data.btc_price_aud, level.price)}>
                  {(data.btc_price_aud - level.price).toFixed(2)} AUD
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

// Render the app
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<LiveTradingDashboard />);
                                         
