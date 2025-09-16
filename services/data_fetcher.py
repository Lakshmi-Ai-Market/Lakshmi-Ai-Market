import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class DataFetcher:
    """Handles fetching market data from multiple sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_yahoo_data(self, symbol: str, interval: str = '1d', period: int = 60) -> Optional[Dict]:
        """
        Fetch data from Yahoo Finance (Primary source - No API key required)
        
        Args:
            symbol: Stock symbol
            interval: Time interval
            period: Number of days
            
        Returns:
            Dictionary containing market data or None if failed
        """
        try:
            # Convert symbol to Yahoo Finance format
            yahoo_symbol = self._convert_to_yahoo_symbol(symbol)
            
            logger.info(f"Fetching Yahoo Finance data for {yahoo_symbol}")
            
            # Use yfinance library for reliable data fetching
            ticker = yf.Ticker(yahoo_symbol)
            
            # Calculate period string for yfinance
            if period <= 7:
                period_str = "7d"
            elif period <= 30:
                period_str = "1mo"
            elif period <= 90:
                period_str = "3mo"
            elif period <= 180:
                period_str = "6mo"
            else:
                period_str = "1y"
            
            # Fetch historical data
            hist = ticker.history(period=period_str, interval=interval)
            
            if hist.empty:
                logger.warning(f"No data returned for {yahoo_symbol}")
                return None
            
            # Get current info
            info = ticker.info
            
            # Convert to our format
            chart_data = []
            for index, row in hist.iterrows():
                chart_data.append({
                    'time': index.isoformat(),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume']) if pd.notna(row['Volume']) else 0
                })
            
            if len(chart_data) < 2:
                logger.warning(f"Insufficient data points for {yahoo_symbol}")
                return None
            
            # Calculate price change
            current_price = chart_data[-1]['close']
            prev_price = chart_data[-2]['close']
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100
            
            # Calculate average volume
            volumes = [item['volume'] for item in chart_data if item['volume'] > 0]
            avg_volume = sum(volumes) / len(volumes) if volumes else 0
            
            result = {
                'symbol': symbol,
                'yahoo_symbol': yahoo_symbol,
                'current_price': current_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'volume': chart_data[-1]['volume'],
                'avg_volume': avg_volume,
                'data_points': len(chart_data),
                'chart': chart_data,
                'info': {
                    'name': info.get('longName', symbol),
                    'currency': info.get('currency', 'INR'),
                    'market_cap': info.get('marketCap'),
                    'pe_ratio': info.get('trailingPE'),
                    'dividend_yield': info.get('dividendYield')
                },
                'last_updated': datetime.now().isoformat(),
                'source': 'Yahoo Finance'
            }
            
            logger.info(f"Successfully fetched {len(chart_data)} data points for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Yahoo Finance fetch failed for {symbol}: {str(e)}")
            return None
    
    def _convert_to_yahoo_symbol(self, symbol: str) -> str:
        """Convert symbol to Yahoo Finance format"""
        symbol_map = {
            'NIFTY': '^NSEI',
            'BANKNIFTY': '^NSEBANK',
            'SENSEX': '^BSESN',
            'FINNIFTY': 'NIFTY_FIN_SERVICE.NS',
            'MIDCPNIFTY': 'NIFTY_MID_SELECT.NS'
        }
        return symbol_map.get(symbol, symbol)
    
    def fetch_alpha_vantage_data(self, symbol: str, api_key: str) -> Optional[Dict]:
        """Fetch data from Alpha Vantage (requires API key)"""
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': api_key,
                'outputsize': 'full'
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Time Series (Daily)' not in data:
                logger.warning(f"No time series data in Alpha Vantage response for {symbol}")
                return None
            
            # Process Alpha Vantage data
            time_series = data['Time Series (Daily)']
            chart_data = []
            
            for date_str, values in sorted(time_series.items()):
                chart_data.append({
                    'time': datetime.strptime(date_str, '%Y-%m-%d').isoformat(),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': int(values['5. volume'])
                })
            
            # Take last 60 days
            chart_data = chart_data[-60:]
            
            if len(chart_data) < 2:
                return None
            
            current_price = chart_data[-1]['close']
            prev_price = chart_data[-2]['close']
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'price_change': current_price - prev_price,
                'price_change_pct': ((current_price - prev_price) / prev_price) * 100,
                'volume': chart_data[-1]['volume'],
                'avg_volume': sum(item['volume'] for item in chart_data) / len(chart_data),
                'data_points': len(chart_data),
                'chart': chart_data,
                'source': 'Alpha Vantage'
            }
            
        except Exception as e:
            logger.error(f"Alpha Vantage fetch failed for {symbol}: {str(e)}")
            return None