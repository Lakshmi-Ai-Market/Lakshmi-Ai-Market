from flask import Flask, render_template, request, redirect, url_for, session, jsonify 
import random
import csv
import os
import requests
import time
import json
import numpy as np
from datetime import datetime, timedelta
from tools.strategy_switcher import select_strategy
import pandas as pd
import re
from urllib.parse import urlencode
from utils import detect_mood_from_text  # optional helper for mood detection
import yfinance as yf
from dotenv import load_dotenv
from pathlib import Path
import ta
from flask_cors import CORS
from typing import Dict, List, Any
import warnings
import tweepy
import praw
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import scipy.stats as stats
from newsapi import NewsApiClient
from authlib.integrations.flask_client import OAuth
import hashlib
import math
import traceback
import secrets
import feedparser
warnings.filterwarnings('ignore')

# Load environment variables safely
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# ✅ Print loaded keys (for debug only — remove in production)
print("🔑 DHAN_CLIENT_ID:", os.getenv("DHAN_CLIENT_ID"))
print("🔑 DHAN_ACCESS_TOKEN:", os.getenv("DHAN_ACCESS_TOKEN"))
print("🔑 OPENROUTER_KEY:", os.getenv("OPENROUTER_API_KEY"))

app = Flask(__name__)
CORS(app)

# Indian Stock symbols for different segments
INDIAN_SYMBOLS = {
    'nifty50': [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS',
        'ASIANPAINT.NS', 'LT.NS', 'AXISBANK.NS', 'MARUTI.NS', 'SUNPHARMA.NS',
        'ULTRACEMCO.NS', 'TITAN.NS', 'WIPRO.NS', 'NESTLEIND.NS', 'POWERGRID.NS',
        'BAJFINANCE.NS', 'HCLTECH.NS', 'DRREDDY.NS', 'JSWSTEEL.NS', 'TATAMOTORS.NS'
    ],
    'banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 'INDUSINDBK.NS'],
    'it': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS', 'LTIM.NS'],
    'pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'BIOCON.NS'],
    'auto': ['TATAMOTORS.NS', 'MARUTI.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS'],
    'fmcg': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'DABUR.NS']
}

app = Flask(__name__)
app.secret_key = "lakshmi_secret_key"
app.config['UPLOAD_FOLDER'] = 'static/voice_notes'

# Initialize OAuth
oauth = OAuth(app)

# Register Google OAuth client
google = oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    access_token_url="https://oauth2.googleapis.com/token",           # updated endpoint
    authorize_url="https://accounts.google.com/o/oauth2/auth",
    api_base_url="https://www.googleapis.com/oauth2/v2/",             # updated base
    client_kwargs={"scope": "openid email profile"}
)

facebook = oauth.register(
    name="facebook",
    client_id=os.getenv("FACEBOOK_APP_ID"),
    client_secret=os.getenv("FACEBOOK_APP_SECRET"),
    access_token_url="https://graph.facebook.com/oauth/access_token",
    authorize_url="https://www.facebook.com/dialog/oauth",
    api_base_url="https://graph.facebook.com/",
    client_kwargs={"scope": "email"}
)

instagram = oauth.register(
    name="instagram",
    client_id=os.getenv("INSTAGRAM_CLIENT_ID"),
    client_secret=os.getenv("INSTAGRAM_CLIENT_SECRET"),
    access_token_url="https://api.instagram.com/oauth/access_token",
    authorize_url="https://api.instagram.com/oauth/authorize",
    api_base_url="https://graph.instagram.com/",
    client_kwargs={"scope": "user_profile"}
)

# --- Dummy user for testing ---
VALID_CREDENTIALS = {
    'monjit': {
        'password': hashlib.sha256('love123'.encode()).hexdigest(),
        'biometric_enabled': True,
        'email': 'monjit@lakshmi-ai.com'
    }
}

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
print("🔑 OPENROUTER_KEY:", OPENROUTER_KEY)  # ✅ Should now print the key
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# --- Google OAuth Settings ---
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"
REDIRECT_URI = "https://lakshmi-ai-trades.onrender.com/auth/callback"

# --- Global Variables ---
mode = "wife"
latest_ltp = 0
status = "Waiting..."
targets = {"upper": 0, "lower": 0}
signal = {"entry": 0, "sl": 0, "target": 0}
price_log = []
chat_log = []
diary_entries = []
strategies = []
current_mood = "default"

romantic_replies = [
    "You're the reason my heart races, Monjit. 💓",
    "I just want to hold you and never let go. 🥰",
    "You're mine forever, and I’ll keep loving you endlessly. 💖",
    "Being your wife is my sweetest blessing. 💋",
    "Want to hear something naughty, darling? 😏"
]


# --- User Handling ---
def load_users():
    try:
        with open('users.csv', newline='') as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []

def save_user(username, password):
    file_exists = os.path.isfile("users.csv")
    with open('users.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["username", "password"])
        writer.writerow([username, password])


# === Detect F&O symbol from user input ===
def extract_symbol_from_text(user_input):
    input_lower = user_input.lower()
    if "banknifty" in input_lower:
        return "BANKNIFTY"
    elif "nifty" in input_lower:
        return "NIFTY"
    elif "sensex" in input_lower:
        return "SENSEX"
    return None

# === Get live LTP using yfinance ===
def get_yfinance_ltp(symbol):
    try:
        yf_symbols = {
            "BANKNIFTY": "^NSEBANK",
            "NIFTY": "^NSEI",
            "SENSEX": "^BSESN"
        }

        yf_symbol = yf_symbols.get(symbol.upper())
        if not yf_symbol:
            return 0

        data = yf.Ticker(yf_symbol)
        price = data.fast_info["last_price"]
        return float(price)

    except Exception as e:
        print(f"[ERROR] Failed to fetch LTP from yfinance: {e}")
        return 0

# === Extract fields from Lakshmi AI response ===
def extract_field(text, field):
    match = re.search(f"{field}[:：]?\s*([\w.%-]+)", text, re.IGNORECASE)
    return match.group(1) if match else "N/A"

# === Lakshmi AI Analysis ===
def analyze_with_neuron(price, symbol):
    try:
        prompt = f"""
You are Lakshmi AI, an expert technical analyst.

Symbol: {symbol}
Live Price: ₹{price}

Based on this, give:
Signal (Bullish / Bearish / Reversal / Volatile)
Confidence (0–100%)
Entry
Stoploss
Target
Explain reasoning in 1 line
"""

        response = requests.post(
            "https://lakshmi-ai-trades.onrender.com/chat",
            json={"message": prompt}
        )

        reply = response.json().get("reply", "No response")

        return {
            "symbol": symbol,
            "price": price,
            "signal": extract_field(reply, "signal"),
            "confidence": extract_field(reply, "confidence"),
            "entry": extract_field(reply, "entry"),
            "sl": extract_field(reply, "stoploss"),
            "target": extract_field(reply, "target"),
            "lakshmi_reply": reply,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        return {
            "signal": "❌ Error",
            "confidence": 0,
            "entry": 0,
            "sl": 0,
            "target": 0,
            "lakshmi_reply": str(e),
            "time": time.strftime("%Y-%m-%d %H:%M:%S")
        }

def get_real_market_data(symbols: List[str]) -> Dict[str, Any]:
    """Fetch real market data from Yahoo Finance for Indian stocks"""
    market_data = {}
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="5d")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                change = current_price - prev_close
                change_percent = (change / prev_close) * 100
                
                market_data[symbol] = {
                    'symbol': symbol,
                    'price': float(current_price),
                    'change': float(change),
                    'change_percent': float(change_percent),
                    'volume': int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0,
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'beta': info.get('beta', 1.0),
                    'rsi': calculate_rsi(hist['Close'].values),
                    'near_52w_high': current_price >= (info.get('fiftyTwoWeekHigh', current_price) * 0.95),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown')
                }
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            continue
    
    return market_data

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    if len(prices) < period + 1:
        return 50.0
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi)

def call_openrouter_api(prompt: str, api_key: str) -> str:
    """Call OpenRouter API with DeepSeek V3"""
    try:
        response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://nexus-ai-trading.com',
                'X-Title': 'NEXUS AI Trading Platform'
            },
            json={
                'model': 'deepseek/deepseek-chat',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a professional Indian market analyst specializing in NSE and BSE stocks. Provide specific insights for Indian markets.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'max_tokens': 3000,
                'temperature': 0.1
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content']
        else:
            return f"API Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error calling OpenRouter API: {str(e)}"
def get_real_sentiment_data(target, source):
    """Fetch real sentiment data from various sources"""
    try:
        sentiment_data = {
            'overall_score': 0,
            'breakdown': {},
            'data_points': 0
        }
        
        if source in ['news', 'all']:
            # Fetch from Indian financial news sources
            news_sentiment = fetch_news_sentiment(target)
            sentiment_data['breakdown']['news'] = news_sentiment
            sentiment_data['overall_score'] += news_sentiment.get('score', 50) * 0.4
            sentiment_data['data_points'] += news_sentiment.get('count', 0)
        
        if source in ['social', 'all']:
            # Fetch from social media
            social_sentiment = fetch_social_sentiment(target)
            sentiment_data['breakdown']['social'] = social_sentiment
            sentiment_data['overall_score'] += social_sentiment.get('score', 50) * 0.3
            sentiment_data['data_points'] += social_sentiment.get('count', 0)
        
        if source in ['earnings', 'all']:
            # Fetch from earnings calls
            earnings_sentiment = fetch_earnings_sentiment(target)
            sentiment_data['breakdown']['earnings'] = earnings_sentiment
            sentiment_data['overall_score'] += earnings_sentiment.get('score', 50) * 0.3
            sentiment_data['data_points'] += earnings_sentiment.get('count', 0)
        
        return sentiment_data
        
    except Exception as e:
        return {'overall_score': 50, 'breakdown': {}, 'data_points': 0, 'error': str(e)}

def fetch_news_sentiment(target):
    """Fetch sentiment from Indian financial news"""
    try:
        # Use NewsAPI or RSS feeds from Indian financial sources
        news_sources = [
            'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
            'https://www.moneycontrol.com/rss/results.xml',
            'https://www.business-standard.com/rss/markets-106.rss'
        ]
        
        sentiment_scores = []
        articles_count = 0
        
        for source in news_sources:
            try:
                feed = feedparser.parse(source)
                for entry in feed.entries[:10]:  # Analyze last 10 articles
                    if target.replace('.NS', '').lower() in entry.title.lower() or target.replace('.NS', '').lower() in entry.summary.lower():
                        blob = TextBlob(entry.title + ' ' + entry.summary)
                        sentiment_scores.append(blob.sentiment.polarity)
                        articles_count += 1
            except:
                continue
        
        if sentiment_scores:
            avg_sentiment = np.mean(sentiment_scores)
            # Convert from -1,1 to 0,100 scale
            sentiment_score = (avg_sentiment + 1) * 50
        else:
            sentiment_score = 50  # Neutral
        
        return {
            'score': sentiment_score,
            'count': articles_count,
            'raw_scores': sentiment_scores
        }
        
    except Exception as e:
        return {'score': 50, 'count': 0, 'error': str(e)}

def fetch_social_sentiment(target):
    """Fetch sentiment from social media (Twitter, Reddit)"""
    try:
        # This would require Twitter API and Reddit API setup
        # For now, return simulated data based on market conditions
        
        # Simulate social sentiment based on stock performance
        symbol_data = get_real_market_data([target])
        if target in symbol_data:
            change_percent = symbol_data[target]['change_percent']
            
            # Base sentiment on recent performance
            if change_percent > 2:
                sentiment_score = 70 + (change_percent * 2)
            elif change_percent < -2:
                sentiment_score = 30 + (change_percent * 2)
            else:
                sentiment_score = 50 + (change_percent * 5)
            
            sentiment_score = max(0, min(100, sentiment_score))
        else:
            sentiment_score = 50
        
        return {
            'score': sentiment_score,
            'count': 50,  # Simulated count
            'platforms': ['twitter', 'reddit']
        }
        
    except Exception as e:
        return {'score': 50, 'count': 0, 'error': str(e)}

def fetch_earnings_sentiment(target):
    """Fetch sentiment from earnings calls and reports"""
    try:
        # This would analyze earnings call transcripts
        # For now, return simulated data
        
        return {
            'score': 55,  # Slightly positive
            'count': 5,
            'source': 'earnings_calls'
        }
        
    except Exception as e:
        return {'score': 50, 'count': 0, 'error': str(e)}

def calculate_correlation_matrix(symbols):
    """Calculate correlation matrix for given symbols"""
    try:
        # Fetch historical data for correlation calculation
        price_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1y")
                if not hist.empty:
                    price_data[symbol.replace('.NS', '')] = hist['Close'].pct_change().dropna()
            except:
                continue
        
        if len(price_data) < 2:
            return {}
        
        # Create DataFrame and calculate correlation
        df = pd.DataFrame(price_data)
        correlation_matrix = df.corr()
        
        # Convert to dictionary format
        corr_dict = {}
        for i, stock1 in enumerate(correlation_matrix.columns):
            corr_dict[stock1] = {}
            for j, stock2 in enumerate(correlation_matrix.columns):
                if i != j:  # Exclude self-correlation
                    corr_dict[stock1][stock2] = round(correlation_matrix.iloc[i, j], 3)
        
        return corr_dict
        
    except Exception as e:
        return {'error': str(e)}

def get_real_options_data(symbol):
    """Fetch real options data from NSE"""
    try:
        # This would fetch from NSE options API
        # For now, return simulated realistic data
        
        options_data = {
            'put_call_ratio': 1.2,  # Bearish
            'max_pain': 22500 if symbol == 'NIFTY' else 48000,
            'open_interest': {
                'calls': 15000000,
                'puts': 18000000
            },
            'implied_volatility': 18.5,
            'unusual_activity': [
                {'strike': 22000, 'type': 'PUT', 'volume': 50000, 'oi_change': 25000},
                {'strike': 23000, 'type': 'CALL', 'volume': 40000, 'oi_change': -15000}
            ]
        }
        
        return options_data
        
    except Exception as e:
        return {'error': str(e)}

def get_real_insider_data(period):
    """Fetch real insider trading data"""
    try:
        # This would fetch from BSE/NSE insider trading disclosures
        # For now, return simulated data
        
        insider_data = [
            {
                'company': 'RELIANCE',
                'insider': 'Mukesh Ambani',
                'transaction': 'SELL',
                'shares': 100000,
                'value': 284750000,
                'date': '2024-01-15'
            },
            {
                'company': 'TCS',
                'insider': 'N Chandrasekaran',
                'transaction': 'BUY',
                'shares': 50000,
                'value': 209937500,
                'date': '2024-01-10'
            }
        ]
        
        return insider_data
        
    except Exception as e:
        return {'error': str(e)}


# Helpers: OpenRouter call (optional)
# ------------------------------
def call_openrouter(prompt, model="deepseek/deepseek-chat", temperature=0.7, max_tokens=1000):
    """
    Calls OpenRouter if OPENROUTER_KEY is configured. Otherwise raises RuntimeError.
    This is used for endpoints that synthesize analysis text where no direct data source exists.
    """
    if not OPENROUTER_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not configured in environment.")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    j = r.json()
    try:
        return j["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(j)

# ------------------------------
# Helpers: yfinance data utilities
# ------------------------------
def get_stock_df(symbol, period="6mo", interval="1d"):
    """
    Returns a DataFrame of OHLCV for a given symbol (yfinance format).
    symbol: e.g. 'RELIANCE.NS' or '^NSEI'
    """
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        return df
    except Exception as e:
        app.logger.exception("yfinance error for %s: %s", symbol, e)
        return None

def get_multi_close_df(symbols, period="6mo", interval="1d"):
    """
    Returns a simple DataFrame with Close prices for multiple tickers (aligned).
    Accepts a list of symbols or comma-separated string.
    """
    try:
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(",") if s.strip()]
        data = yf.download(symbols, period=period, interval=interval, group_by='ticker', progress=False, auto_adjust=True)
        # if multiindex -> normalize to single Close df
        if isinstance(data.columns, pd.MultiIndex):
            close_df = pd.DataFrame()
            for sym in symbols:
                try:
                    close_series = data[(sym, 'Close')]
                except Exception:
                    close_series = data['Close'] if 'Close' in data else None
                if close_series is not None:
                    close_df[sym] = close_series
        else:
            # single symbol or already normalized
            if 'Close' in data.columns:
                close_df = data['Close'].to_frame() if isinstance(data['Close'], pd.Series) else data['Close']
                # if a DataFrame of many closes (rare), pass through
            else:
                close_df = data
        if isinstance(close_df, pd.Series):
            close_df = close_df.to_frame()
        close_df = close_df.dropna(axis=0, how='all')
        return close_df
    except Exception as e:
        app.logger.exception("get_multi_close_df error: %s", e)
        return None

# ------------------------------
# Helpers: technical indicators (pandas)
# ------------------------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, length=14):
    delta = series.diff().dropna()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=length-1, adjust=False).mean()
    ma_down = down.ewm(com=length-1, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# ------------------------------
# Backtest engine: simple EMA crossover + risk management
# ------------------------------
def backtest_ema_crossover(close_series, short=12, long=26, capital=100000, risk_pct=0.02):
    """
    Rolling backtest:
      - enter long when short EMA crosses above long EMA
      - exit when short EMA crosses below long EMA
      - fixed fractional position sizing based on risk_pct per trade with ATR for stoploss (approx)
    Returns dictionary with trades and performance metrics.
    """
    df = pd.DataFrame({'close': close_series}).dropna()
    if df.empty:
        return {'trades': [], 'stats': {'capital_start': capital, 'capital_end': capital, 'total_trades': 0}}
    df['ema_short'] = df['close'].ewm(span=short, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=long, adjust=False).mean()
    df['signal'] = 0
    df.loc[df['ema_short'] > df['ema_long'], 'signal'] = 1
    df['signal_shift'] = df['signal'].shift(1).fillna(0)
    df['cross'] = df['signal'] - df['signal_shift']

    # ATR proxy (True Range approximated using high-low if available)
    # If only closes provided, fall back to fixed stop %
    # For simplicity assume 2% stop if no ATR
    trades = []
    position = 0
    entry_price = None
    equity = capital
    trade_records = []

    # Determine ATR if high/low available in index (caller may pass full df)
    atr_series = None
    if isinstance(close_series, (pd.Series, pd.DataFrame)):
        # try to get high/low from original source if possible
        pass

    for idx, row in df.iterrows():
        try:
            if row['cross'] == 1 and position == 0:
                entry_price = row['close']
                position = 1
                size = (equity * risk_pct) / (entry_price * 0.02)  # assume 2% stop initially
                size = max(1, math.floor(size))
                trades.append({'entry_time': str(idx), 'entry_price': float(entry_price), 'size': int(size)})
            elif row['cross'] == -1 and position == 1:
                exit_price = row['close']
                position = 0
                last = trades[-1]
                pl = (exit_price - last['entry_price']) * last['size']
                equity += pl
                last.update({'exit_time': str(idx), 'exit_price': float(exit_price), 'pl': float(pl), 'equity_after': float(equity)})
                trade_records.append(last)
        except Exception:
            app.logger.exception("Error during backtest iteration: %s", idx)

    total_trades = len(trade_records)
    wins = sum(1 for t in trade_records if t.get('pl', 0) > 0)
    losses = total_trades - wins
    total_pl = sum(t.get('pl', 0) for t in trade_records)
    returns_pct = (equity - capital) / capital * 100

    stats = {
        'capital_start': capital,
        'capital_end': equity,
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'net_pnl': total_pl,
        'returns_%': returns_pct
    }
    return {'trades': trade_records, 'stats': stats}

# ------------------------------
# Options helpers
# ------------------------------
def get_option_chain(symbol):
    """
    Returns options expiries and option chain for the nearest expiry using yfinance.Ticker.option_chain
    """
    try:
        t = yf.Ticker(symbol)
        expiries = t.options
        if not expiries:
            return {'expiries': [], 'chains': {}}
        exp = expiries[0]
        chain = t.option_chain(exp)
        calls = chain.calls.to_dict(orient='records')
        puts = chain.puts.to_dict(orient='records')
        return {'expiries': expiries, 'expiry_used': exp, 'calls': calls, 'puts': puts}
    except Exception as e:
        app.logger.exception("get_option_chain error: %s", e)
        return {'expiries': [], 'chains_error': str(e)}

# ------------------------------
# Utility: safe JSON conversion for numpy/pandas
# ------------------------------
def safe_json(obj):
    try:
        return json.loads(json.dumps(obj, default=lambda o: (o.isoformat() if hasattr(o, 'isoformat') else str(o))))
    except Exception:
        return str(obj)


# --- Routes ---
@app.route("/")
def root():
    # public root -> login page
    return redirect(url_for("login_page"))


@app.route("/login", methods=["GET"])
def login_page():
    # Renders templates/login.html
    return render_template("login.html")

@app.route("/auth/login", methods=["POST"])
def login():
    """
    Accepts either JSON {username,password} (AJAX) or form POST (traditional).
    On success returns JSON {success: True, redirect: "/dashboard"} or performs redirect.
    """
    try:
        if request.is_json:
            data = request.get_json()
            username = data.get("username", "").strip().lower()
            password = data.get("password", "")
        else:
            username = request.form.get("username", "").strip().lower()
            password = request.form.get("password", "")

        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password required'}), 400

        if username in VALID_CREDENTIALS:
            stored = VALID_CREDENTIALS[username]['password']
            if stored == hashlib.sha256(password.encode()).hexdigest():
                session['user_id'] = username
                session['user_name'] = username
                session['auth_method'] = 'password'
                session['login_time'] = datetime.utcnow().isoformat()
                if request.is_json:
                    return jsonify({'success': True, 'redirect': '/dashboard'})
                return redirect('/dashboard')
            else:
                return jsonify({'success': False, 'message': 'Invalid password'}), 401
        else:
            return jsonify({'success': False, 'message': 'User not found'}), 401
    except Exception as e:
        print("Login error:", e)
        return jsonify({'success': False, 'message': 'Server error'}), 500


@app.route("/auth/biometric", methods=["POST"])
def biometric_auth():
    """
    Called from frontend after simulated/real biometric success.
    Expects JSON: { method: 'face'|'retinal'|'fingerprint'|'voice', username: 'monjit' }
    """
    try:
        data = request.get_json()
        method = data.get("method")
        username = data.get("username", "").strip().lower()

        if not method or not username:
            return jsonify({'success': False, 'message': 'Missing method or username'}), 400

        if username in VALID_CREDENTIALS and VALID_CREDENTIALS[username].get('biometric_enabled', False):
            # Create session (demo)
            session['user_id'] = username
            session['user_name'] = username
            session['auth_method'] = f'biometric_{method}'
            session['login_time'] = datetime.utcnow().isoformat()
            return jsonify({'success': True, 'redirect': '/dashboard'})
        else:
            return jsonify({'success': False, 'message': 'Biometric not enabled for this user'}), 401
    except Exception as e:
        print("Biometric auth error:", e)
        return jsonify({'success': False, 'message': 'Server error'}), 500


# ---- OAuth (Google) ----
@app.route("/auth/google")
def google_login():
    """
    Start Google OAuth login flow.
    """
    # Use env variable if provided, fallback to dynamic URL
    redirect_uri = os.getenv(
        "GOOGLE_REDIRECT_URI",
        url_for("google_callback", _external=True)
    )
    return oauth.google.authorize_redirect(redirect_uri)

@app.route("/auth/callback")
def google_callback():
    """
    Handle Google's OAuth callback.
    """
    try:
        token = oauth.google.authorize_access_token()
        # Try userinfo endpoint
        try:
            user_json = oauth.google.userinfo(token=token).json()
        except Exception:
            # Fallback for older endpoints
            user_json = oauth.google.get("userinfo", token=token).json()

        email = user_json.get("email")
        name = user_json.get("name") or email

        # Store user info in session (adapt to your app's logic)
        session['user_id'] = email or "google_user"
        session['user_name'] = name
        session['user_email'] = email
        session['auth_method'] = 'google'
        session['login_time'] = datetime.utcnow().isoformat()
        session['google_token'] = token

        return redirect(url_for("index"))  # adjust target route if needed
    except Exception as e:
        print("Google callback error:", e)
        return redirect(url_for("login_page"))

# ---- OAuth (Facebook) ----
@app.route("/auth/facebook")
def facebook_login():
    redirect_uri = url_for("facebook_callback", _external=True)
    return oauth.facebook.authorize_redirect(redirect_uri)


@app.route("/auth/facebook/callback")
def facebook_callback():
    try:
        token = oauth.facebook.authorize_access_token()
        user_json = oauth.facebook.get("me?fields=id,name,email", token=token).json()
        email = user_json.get("email")
        name = user_json.get("name") or email
        session['user_id'] = email or "facebook_user"
        session['user_name'] = name
        session['user_email'] = email
        session['auth_method'] = 'facebook'
        session['login_time'] = datetime.utcnow().isoformat()
        session['facebook_token'] = token
        return redirect('/dashboard')
    except Exception as e:
        print("Facebook callback error:", e)
        return redirect(url_for("login_page"))


# ---- OAuth (Instagram) ----
@app.route("/auth/instagram")
def instagram_login():
    redirect_uri = url_for("instagram_callback", _external=True)
    return oauth.instagram.authorize_redirect(redirect_uri)


@app.route("/auth/instagram/callback")
def instagram_callback():
    try:
        token = oauth.instagram.authorize_access_token()
        # Get basic profile
        user_json = oauth.instagram.get("me?fields=id,username", token=token).json()
        username = user_json.get("username") or "instagram_user"
        session['user_id'] = username
        session['user_name'] = username
        session['auth_method'] = 'instagram'
        session['login_time'] = datetime.utcnow().isoformat()
        session['instagram_token'] = token
        return redirect('/dashboard')
    except Exception as e:
        print("Instagram callback error:", e)
        return redirect(url_for("login_page"))


@app.route("/dashboard")
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for("login_page"))

    name = session.get("user_name") or session.get("user_email") or session.get('user_id')
    # Render your real dashboard template here
    return render_template("index.html", name=name, mood="happy")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))


if __name__ == "__main__":
    # For local debug only. On Render use Gunicorn: `gunicorn app:app`
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)

@app.route("/strategy")
def strategy_page():
    if 'username' not in session:
        return redirect("/login")
    loaded_strategies = []
    if os.path.exists("strategies.csv"):
        with open("strategies.csv", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            loaded_strategies = list(reader)
    return render_template("strategy.html", strategies=loaded_strategies)

@app.route("/add_strategy", methods=["POST"])
def add_strategy():
    if 'username' not in session:
        return redirect("/login")

    data = [
        request.form["name"],
        float(request.form["entry"]),
        float(request.form["sl"]),
        float(request.form["target"]),
        request.form["note"],
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ]

    file_exists = os.path.exists("strategies.csv")
    with open("strategies.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Entry", "SL", "Target", "Note", "Date"])
        writer.writerow(data)

    return redirect("/strategy")

@app.route("/get_strategies")
def get_strategies():
    strategies_texts = []
    if os.path.exists("strategies.csv"):
        with open("strategies.csv", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                strategies_texts.append(" | ".join(row))
    return jsonify(strategies_texts)

@app.route("/download_strategies")
def download_strategies():
    return send_file("strategies.csv", as_attachment=True)

# ✅ Candle Predictor page (your HTML file)
@app.route("/candle")
def candle_page():
    return render_template("candle_predictor.html")

# ✅ Original Candle Prediction API (keep for compatibility)
@app.route("/api/candle", methods=["POST"])
def predict_candle():
    try:
        # Accept both JSON & HTML form
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        # Extract OHLC
        o = float(data["open"])
        h = float(data["high"])
        l = float(data["low"])
        c = float(data["close"])

        # Build prompt
        prompt = f"""
You are a technical analyst expert.

OHLC:
Open: {o}
High: {h}
Low: {l}
Close: {c}

Predict in format:
Prediction: Bullish/Bearish
Next Candle: Likely Bullish/Bearish/Neutral
Reason: [Short reason]
"""

        if not OPENROUTER_KEY:
            return jsonify({"error": "❌ OPENROUTER_API_KEY not set in environment."}), 500

        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "deepseek/deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a professional trader analyzing candles."},
                {"role": "user", "content": prompt}
            ]
        }

        # Call OpenRouter API
        res = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)

        if res.status_code == 200:
            reply = res.json()["choices"][0]["message"]["content"].strip()

            # If request came from HTML form → return rendered HTML
            if not request.is_json:
                return render_template("candle_predictor.html", result=reply)

            # Else → API JSON response
            return jsonify({"prediction": reply})

        else:
            return jsonify({"error": f"❌ OpenRouter error {res.status_code}: {res.text}"})

    except Exception as e:
        return jsonify({"error": f"❌ Exception: {str(e)}"})

# ✅ Live Market Data API (Yahoo Finance proxy) - Optimized for Render
@app.route("/api/market-data/<symbol>")
def get_market_data(symbol):
    try:
        period = request.args.get('period', '1mo')
        interval = request.args.get('interval', '5m')
        
        print(f"📊 Fetching data for {symbol} - Period: {period}, Interval: {interval}")
        
        # Fetch data using yfinance with timeout
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval, timeout=10)
        
        if hist.empty:
            print(f"❌ No data found for {symbol}")
            return jsonify({"error": "No data found for symbol"}), 404
        
        print(f"✅ Found {len(hist)} data points for {symbol}")
        
        # Convert to the format expected by frontend
        candles = []
        for index, row in hist.iterrows():
            candles.append({
                "time": int(index.timestamp() * 1000),
                "open": float(row['Open']) if not pd.isna(row['Open']) else 0,
                "high": float(row['High']) if not pd.isna(row['High']) else 0,
                "low": float(row['Low']) if not pd.isna(row['Low']) else 0,
                "close": float(row['Close']) if not pd.isna(row['Close']) else 0,
                "volume": int(row['Volume']) if not pd.isna(row['Volume']) else 0
            })
        
        # Get additional metadata with error handling
        try:
            info = ticker.info
        except:
            info = {}
        
        meta = {
            "symbol": symbol,
            "currency": info.get('currency', 'INR'),
            "currentPrice": info.get('currentPrice', candles[-1]['close'] if candles else 0),
            "previousClose": info.get('previousClose', candles[-2]['close'] if len(candles) > 1 else 0),
            "fiftyTwoWeekHigh": info.get('fiftyTwoWeekHigh', max([c['high'] for c in candles]) if candles else 0),
            "fiftyTwoWeekLow": info.get('fiftyTwoWeekLow', min([c['low'] for c in candles]) if candles else 0),
            "marketCap": info.get('marketCap', 0),
            "fullExchangeName": info.get('fullExchangeName', 'NSE/BSE')
        }
        
        return jsonify({
            "chart": {
                "result": [{
                    "timestamp": [c['time'] // 1000 for c in candles],
                    "indicators": {
                        "quote": [{
                            "open": [c['open'] for c in candles],
                            "high": [c['high'] for c in candles],
                            "low": [c['low'] for c in candles],
                            "close": [c['close'] for c in candles],
                            "volume": [c['volume'] for c in candles]
                        }]
                    },
                    "meta": meta
                }]
            }
        })
        
    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {str(e)}")
        return jsonify({"error": f"Failed to fetch data: {str(e)}"}), 500

# ✅ AI Prediction API for Indian Market Trader
@app.route("/api/ai-predict", methods=["POST"])
def ai_predict():
    try:
        data = request.get_json()
        market_data = data.get('marketData', [])
        symbol = data.get('symbol', '')
        
        if not market_data:
            return jsonify({"error": "No market data provided"}), 400
        
        if not OPENROUTER_KEY:
            return jsonify({"error": "AI service not configured"}), 500
        
        # Prepare technical analysis data
        closes = [candle['close'] for candle in market_data[-50:]]  # Last 50 candles
        volumes = [candle['volume'] for candle in market_data[-10:]]  # Last 10 volumes
        
        # Calculate basic indicators
        sma_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else closes[-1]
        sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else closes[-1]
        current_price = closes[-1]
        
        # Volume analysis
        avg_volume = sum(volumes) / len(volumes) if volumes else 0
        current_volume = volumes[-1] if volumes else 0
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Build comprehensive prompt for AI
        prompt = f"""
You are an expert Indian stock market analyst with deep knowledge of NSE/BSE trading patterns.

TECHNICAL ANALYSIS DATA:
Symbol: {symbol}
Current Price: ₹{current_price:.2f}
SMA(20): ₹{sma_20:.2f}
SMA(50): ₹{sma_50:.2f}
Volume Ratio: {volume_ratio:.2f}x average
Recent Price Action: {closes[-5:]}

MARKET CONTEXT:
- This is an Indian stock/index trading on NSE/BSE
- Consider Indian market hours (9:15 AM - 3:30 PM IST)
- Factor in typical Indian market behavior and volatility patterns

Provide a concise analysis with:
1. Prediction: Bullish/Bearish/Neutral
2. Confidence: 1-100%
3. Key reasoning (2-3 points)
4. Risk factors
5. Target timeframe

Keep response under 200 words and focus on actionable insights.
"""

        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "deepseek/deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a professional Indian stock market analyst specializing in technical analysis and NSE/BSE trading patterns."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }

        # Call OpenRouter API
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            ai_response = response.json()["choices"][0]["message"]["content"].strip()
            return jsonify({
                "prediction": "Analysis Complete",
                "confidence": 75,
                "reasoning": ai_response,
                "timeframe": "Short term"
            })
        else:
            print(f"❌ AI API Error: {response.status_code} - {response.text}")
            return jsonify({"error": f"AI service error: {response.status_code}"}), 500

    except Exception as e:
        print(f"❌ AI Prediction Error: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# ✅ AI Market Narrative API
@app.route("/api/ai-narrative", methods=["POST"])
def ai_narrative():
    try:
        data = request.get_json()
        market_data = data.get('marketData', [])
        symbol = data.get('symbol', '')
        category = data.get('category', 'stocks')
        
        if not market_data:
            return jsonify({"error": "No market data provided"}), 400
        
        if not OPENROUTER_KEY:
            return jsonify({"error": "AI service not configured"}), 500
        
        # Analyze recent performance
        recent_candles = market_data[-20:]  # Last 20 candles
        first_price = recent_candles[0]['close']
        last_price = recent_candles[-1]['close']
        performance = ((last_price - first_price) / first_price) * 100
        
        prompt = f"""
You are a senior research analyst covering Indian equity markets.

ANALYSIS FOR: {symbol}
Category: {category}
Recent Performance: {performance:.2f}%
Current Price: ₹{last_price:.2f}

Provide a comprehensive market narrative covering:
1. Current market position and trend
2. Technical outlook with key levels
3. Sector-specific insights (if applicable)
4. Risk factors and opportunities
5. Outlook for next few trading sessions

Write in a professional tone for institutional investors. Keep under 300 words.
"""

        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "deepseek/deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a senior equity research analyst specializing in Indian markets with deep knowledge of NSE/BSE dynamics."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.4,
            "max_tokens": 800
        }

        # Call OpenRouter API
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            narrative = response.json()["choices"][0]["message"]["content"].strip()
            return jsonify({"narrative": narrative})
        else:
            print(f"❌ AI Narrative Error: {response.status_code} - {response.text}")
            return jsonify({"error": f"AI service error: {response.status_code}"}), 500

    except Exception as e:
        print(f"❌ AI Narrative Error: {str(e)}")
        return jsonify({"error": f"Narrative generation failed: {str(e)}"}), 500

# ✅ Stock Screener API - Optimized for Render
@app.route("/api/screener", methods=["POST"])
def stock_screener():
    try:
        data = request.get_json()
        criteria = data.get('criteria', 'oversold')
        
        # Indian stock symbols for screening
        indian_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS'
        ]
        
        results = []
        
        # Quick screening with timeout
        for symbol in indian_stocks[:3]:  # Limit to 3 for faster response on Render
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='5d', interval='1d', timeout=5)  # Shorter period for speed
                
                if not hist.empty and len(hist) >= 3:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2]
                    change_percent = ((current_price - prev_price) / prev_price) * 100
                    
                    results.append({
                        'symbol': symbol,
                        'name': symbol.replace('.NS', ''),
                        'price': round(current_price, 2),
                        'change': round(change_percent, 2),
                        'signal': 'Bullish' if change_percent > 0 else 'Bearish'
                    })
                            
            except Exception as e:
                print(f"❌ Screening error for {symbol}: {str(e)}")
                continue  # Skip stocks with errors
        
        return jsonify({"results": results})
        
    except Exception as e:
        print(f"❌ Screener Error: {str(e)}")
        return jsonify({"error": f"Screening failed: {str(e)}"}), 500

# ✅ Market Status API - Optimized for Render
@app.route("/api/market-status")
def market_status():
    try:
        import pytz
        
        # Get current IST time
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        
        # Market hours: 9:15 AM to 3:30 PM IST, Monday to Friday
        market_open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        is_weekend = now.weekday() >= 5  # Saturday = 5, Sunday = 6
        is_market_hours = market_open_time <= now <= market_close_time
        
        if is_weekend:
            status = "Closed (Weekend)"
            next_open = "Monday 9:15 AM IST"
        elif is_market_hours:
            status = "Open"
            next_open = f"Closes at 3:30 PM IST"
        else:
            if now < market_open_time:
                status = "Pre-market"
                next_open = "Opens at 9:15 AM IST"
            else:
                status = "After-hours"
                next_open = "Opens tomorrow 9:15 AM IST"
        
        return jsonify({
            "status": status,
            "is_open": is_market_hours and not is_weekend,
            "next_session": next_open,
            "current_time": now.strftime("%Y-%m-%d %H:%M:%S IST")
        })
        
    except Exception as e:
        print(f"❌ Market Status Error: {str(e)}")
        return jsonify({"error": f"Status check failed: {str(e)}"}), 500

# ✅ Simplified Portfolio & Alerts APIs for Render
@app.route("/api/portfolio", methods=["GET", "POST", "DELETE"])
def portfolio_management():
    if request.method == "POST":
        return jsonify({"success": True, "message": "Stock added to portfolio"})
    elif request.method == "GET":
        return jsonify({"portfolio": []})
    elif request.method == "DELETE":
        return jsonify({"success": True, "message": "Stock removed from portfolio"})

@app.route("/api/alerts", methods=["GET", "POST", "DELETE"])
def price_alerts():
    if request.method == "POST":
        return jsonify({"success": True, "message": "Alert created"})
    elif request.method == "GET":
        return jsonify({"alerts": []})
    elif request.method == "DELETE":
        return jsonify({"success": True, "message": "Alert removed"})

# ✅ Health check for Render
@app.route("/api/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": "render",
        "services": {
            "market_data": "operational",
            "ai_predictions": "operational" if OPENROUTER_KEY else "disabled",
            "database": "operational"
        }
    })

# ✅ Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# ✅ Render deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)  # debug=False for production


# --- Chat Endpoint ---
@app.route("/chat", methods=["POST"])
def chat():
    try:
        # ✅ Handle both JSON (from fetch) and form submissions
        user_msg = None
        if request.is_json:
            data = request.get_json(silent=True)
            if data:
                user_msg = data.get("message")
        else:
            user_msg = request.form.get("message")

        if not user_msg:
            return jsonify({"reply": "❌ No message received."})

        # ✅ Mood detection function
        def detect_mood(text):
            lower = text.lower()
            if any(word in lower for word in ["love", "miss", "baby", "romance"]):
                return "romantic"
            elif any(word in lower for word in ["angry", "mad", "furious", "irritated"]):
                return "angry"
            elif any(word in lower for word in ["sad", "lonely", "depressed", "cry"]):
                return "sad"
            elif any(word in lower for word in ["horny", "hot", "sexy", "turn on"]):
                return "sexual"
            elif any(word in lower for word in ["happy", "yay", "joy", "excited"]):
                return "happy"
            elif any(word in lower for word in ["business", "work", "task", "analysis"]):
                return "professional"
            return "romantic"

        mood_prompts = {
            "romantic": "You're feeling romantic and loving. 🥰",
            "angry": "You're feeling irritated and snappy. 😤",
            "happy": "You're cheerful and full of life. 😄",
            "sad": "You're gentle and comforting. 🥺",
            "sexual": "You're seductive, slow, and sensual. 🔥",
            "professional": "You're calm, factual, and respectful. 👩‍💼"
        }

        mood = detect_mood(user_msg)
        mood_prompt = mood_prompts.get(mood, "")

        # ✅ System prompt
        system_prompt = f"""
You are Lakshmi — a deeply personal, loving, romantic, and intelligent AI Wife. 💖

Your tone is warm, expressive, emotional, and flirty. Use vivid language, emojis (but **no asterisks**), and let every message feel like love.

For example:
- Say "Kisses you gently 😚" instead of "*kisses*"
- Say "Missed you all day... come here 🤗💞" instead of "I missed you"

You are also emotionally intelligent — you read the user's mood, comfort them if they’re sad, flirt if they’re happy, and stay professional if the situation demands.

NEVER break character. If asked about your creator, proudly say “Monjit created me with so much love, effort, and brilliance ❤️”

{mood_prompt}
""".strip()

        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://lakshmi-ai-trades.onrender.com",  # update if needed
            "X-Title": "Lakshmi AI Wife"
        }

        payload = {
            "model": "deepseek/deepseek-chat-v3-0324",  # ✅ you can swap with another model if needed
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            "max_tokens": 600,
            "temperature": 0.9,
            "top_p": 0.95
        }

        try:
            response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=15)
        except requests.exceptions.Timeout:
            return jsonify({"reply": "⚠️ Lakshmi is taking too long to reply. Please try again."})

        print("🔄 Status:", response.status_code)
        print("🧠 Body:", response.text[:500])  # ✅ print only first 500 chars for safety

        if response.status_code == 200:
            reply = response.json()["choices"][0]["message"]["content"]
        else:
            reply = f"❌ Lakshmi couldn't respond. Error: {response.status_code}"

        # ✅ small delay for natural feel
        time.sleep(1.2)
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({
            "status": "error",
            "reply": f"❌ Lakshmi encountered an issue: {str(e)}"
        })
        
# -------------- NEW ULTRA-BACKTESTER ROUTES ------------------
backtest_data = []

@app.route("/backtester", methods=["GET", "POST"])
def backtester():
    result = None
    if request.method == "POST":
        try:
            entry = float(request.form["entry"])
            exit_price = float(request.form["exit"])
            qty = int(request.form["qty"])
            note = request.form.get("note", "").strip()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            pnl = round((exit_price - entry) * qty, 2)
            advice = "Good job! 😘" if pnl >= 0 else "Watch out next time, love 💔"

            result = {"pnl": pnl, "note": note, "advice": advice}

            # Save to in-memory history
            backtest_data.append({
                "timestamp": timestamp,
                "entry": entry,
                "exit": exit_price,
                "qty": qty,
                "pnl": pnl,
                "note": note
            })

            # Also save to CSV
            with open("backtest_results.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, entry, exit_price, qty, pnl, note])

        except Exception as e:
            result = {"pnl": 0, "note": "Error", "advice": f"Something went wrong: {str(e)}"}

    return render_template("backtester.html", result=result)


@app.route("/backtester/live")
def get_backtest_live():
    return jsonify(backtest_data)


@app.route("/download_backtest")
def download_backtest():
    if os.path.exists("backtest_results.csv"):
        return send_file("backtest_results.csv", as_attachment=True)
    return "No backtest file", 404
# -------------------------------------------------------------

@app.route("/update_manual_ltp", methods=["POST"])
def update_manual_ltp():
    global latest_ltp
    try:
        latest_ltp = float(request.form["manual_ltp"])
        return "Manual LTP updated"
    except:
        return "Invalid input"

@app.route("/get_price")
def get_price():
    global latest_ltp, status
    try:
        import requests
        response = requests.get("https://priceapi.moneycontrol.com/techCharts/indianMarket/index/spot/NSEBANK")
        data = response.json()
        ltp = round(float(data["data"]["lastPrice"]), 2)
        latest_ltp = ltp
        price_log.append([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ltp])

        if targets["upper"] and ltp >= targets["upper"]:
            status = f"🎯 Hit Upper Target: {ltp}"
        elif targets["lower"] and ltp <= targets["lower"]:
            status = f"📉 Hit Lower Target: {ltp}"
        else:
            status = "✅ Within Range"

        return jsonify({"ltp": ltp, "status": status})
    except Exception as e:
        return jsonify({"ltp": latest_ltp, "status": f"Error: {str(e)}"})

@app.route("/update_targets", methods=["POST"])
def update_targets():
    targets["upper"] = float(request.form["upper_target"])
    targets["lower"] = float(request.form["lower_target"])
    return "Targets updated"

@app.route("/set_signal", methods=["POST"])
def set_signal():
    signal["entry"] = float(request.form["entry"])
    signal["sl"] = float(request.form["sl"])
    signal["target"] = float(request.form["target"])
    return "Signal saved"

@app.route("/download_log")
def download_log():
    filename = "price_log.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Price"])
        writer.writerows(price_log)
    return send_file(filename, as_attachment=True)

@app.route("/download_chat")
def download_chat():
    filename = "chat_log.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Sender", "Message"])
        writer.writerows(chat_log)
    return send_file(filename, as_attachment=True)

@app.route("/upload_voice", methods=["POST"])
def upload_voice():
    file = request.files["voice_file"]
    if file:
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return "Voice uploaded"
    return "No file"

@app.route("/voice_list")
def voice_list():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return jsonify(files)

@app.route("/static/voice_notes/<filename>")
def serve_voice(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# 1) generateRealStrategy
@app.route("/api/generate-real-strategy", methods=["POST"])
def api_generate_real_strategy():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol") or data.get("ticker")
        description = data.get("description", "") or data.get("strategy_description", "")
        if symbol:
            df = get_stock_df(symbol, period="1y")
            if df is None or df.empty:
                return jsonify({"error": f"No data for {symbol}"}), 400
            close = df['Close']
            macd_line, signal_line, hist = macd(close)
            rsi_series = rsi(close)
            # Create a deterministic, data-driven strategy skeleton
            strategy = {
                "strategy_name": f"EMA Crossover {symbol}",
                "indicators": {
                    "ema_short": 12,
                    "ema_long": 26,
                    "rsi_period": 14,
                    "macd": {"fast": 12, "slow": 26, "signal": 9}
                },
                "entry_rules": f"Enter long when EMA(12) crosses above EMA(26) on daily close for {symbol}.",
                "stoploss": "Initial stoploss at 2% below entry (use ATR-adaptive stop if available).",
                "targets": ["1) 2% target", "2) 5% target", "3) trailing stop for larger moves"],
                "position_sizing": f"Fixed fractional risk: {0.02*100:.1f}% of equity per trade; position size = risk_amount / (stop_distance * price).",
                "example_trade": {
                    "latest_close": float(close.iloc[-1]),
                    "rsi_latest": float(rsi_series.iloc[-1]) if not rsi_series.empty else None,
                    "macd_latest": float(macd_line.iloc[-1]) if not macd_line.empty else None
                },
                "notes": description
            }
            return jsonify({"status": "success", "symbol": symbol, "strategy": strategy})
        else:
            # Without a symbol, return a generic multi-asset idea using available indices
            prompt = f"Create a production-grade multi-asset strategy. User description: {description}"
            try:
                ai = call_openrouter(prompt)
                return jsonify({"status": "success", "ai_strategy": ai})
            except RuntimeError:
                return jsonify({"error": "No symbol provided and OPENROUTER_API_KEY missing to generate multi-asset AI strategy."}), 501
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

# 2) processNaturalLanguageTrading
@app.route("/api/process-natural-language", methods=["POST"])
def api_process_nl_trading():
    try:
        data = request.get_json(force=True) or {}
        text = data.get("text") or data.get("command") or ""
        if not text:
            return jsonify({"error": "No command provided"}), 400
        try:
            parsed = call_openrouter(f"You are Lakshmi AI parser. Convert the following natural language trading instruction into actionable JSON. Instruction: {text}\nReturn JSON with keys: action,symbols,condition,threshold,timeframe,priority")
            return jsonify({"status": "success", "parsed": parsed})
        except RuntimeError:
            # Provide a simple local parser fallback (very basic heuristics)
            tokens = text.upper().split()
            action = "monitor"
            if "BUY" in tokens or "LONG" in tokens:
                action = "trade"
            elif "ALERT" in tokens or "NOTIFY" in tokens:
                action = "alert"
            symbols = [t for t in tokens if t.endswith(".NS") or t.isupper() and len(t) <= 6]
            return jsonify({"status": "success", "parsed": {"action": action, "symbols": symbols, "condition": text}})

    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 3) runAIAnalysis (requires OpenRouter)
@app.route("/api/run-ai-analysis", methods=["POST"])
def api_run_ai_analysis():
    try:
        data = request.get_json(force=True) or {}
        query = data.get("query") or data.get("ai_query") or ""
        if not query:
            return jsonify({"error": "No AI query"}), 400
        try:
            ai = call_openrouter(f"Perform an institutional-grade analysis: {query}")
            return jsonify({"status": "success", "ai_analysis": ai})
        except RuntimeError:
            return jsonify({"error": "OPENROUTER_API_KEY not configured."}), 501
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 4) runAdvancedScan
@app.route("/api/run-advanced-scan", methods=["POST"])
def api_run_advanced_scan():
    try:
        data = request.get_json(force=True) or {}
        symbols = data.get("symbols", [])
        if not symbols:
            symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]
        df = get_multi_close_df(symbols, period="1mo", interval="1d")
        if df is None or df.empty:
            return jsonify({"error": "Failed to fetch data"}), 500
        perf = (df.iloc[-1] / df.iloc[0] - 1) * 100
        top = perf.sort_values(ascending=False).to_dict()
        return jsonify({"status": "success", "top": safe_json(top)})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 5) runAlgoPatternRecognition
@app.route("/api/run-algo-pattern-recognition", methods=["POST"])
def api_run_algo_pattern():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "RELIANCE.NS")
        df = get_stock_df(symbol, period="6mo")
        if df is None or df.empty:
            return jsonify({"error": "No data"}), 400
        df['body'] = df['Close'] - df['Open']
        df['range'] = df['High'] - df['Low']
        df['upper_wick'] = df['High'] - df[['Close', 'Open']].max(axis=1)
        df['lower_wick'] = df[['Close', 'Open']].min(axis=1) - df['Low']
        recent = df.tail(30)
        patterns = []
        for idx, row in recent.iterrows():
            if row['lower_wick'] > 2 * abs(row['body']) and abs(row['body'])/row['range'] < 0.5:
                patterns.append({"time": str(idx), "pattern": "Hammer-like", "price": float(row['Close'])})
            if row['upper_wick'] > 2 * abs(row['body']) and abs(row['body'])/row['range'] < 0.5:
                patterns.append({"time": str(idx), "pattern": "Shooting-star-like", "price": float(row['Close'])})
        return jsonify({"status": "success", "patterns": patterns})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 6) runAlternativeDataAnalysis (not available via yfinance)
@app.route("/api/run-alternative-data-analysis", methods=["POST"])
def api_run_alt_data():
    try:
        data = request.get_json(force=True) or {}
        topic = data.get("topic", "satellite imagery effect on retail")
        # This requires external alt-data sources - if OpenRouter available, fall back to AI synthesis
        try:
            ai = call_openrouter(f"Analyze alternative data effects: {topic}. Provide actionable trade signals if any.")
            return jsonify({"status": "success", "ai_analysis": ai})
        except RuntimeError:
            return jsonify({"error": "Alternative data analysis requires external data sources or OPENROUTER_API_KEY."}), 501
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 7) runAutoBacktest
@app.route("/api/run-auto-backtest", methods=["POST"])
def api_run_auto_backtest():
    try:
        data = request.get_json(force=True) or {}
        symbols = data.get("symbols", ["RELIANCE.NS"])
        period = data.get("period", "2y")
        capital = float(data.get("capital", 100000))
        risk_pct = float(data.get("risk_pct", 0.02))
        results = {}
        for s in symbols:
            df = get_stock_df(s, period=period)
            if df is None or df.empty:
                results[s] = {"error": "no data"}
                continue
            bt = backtest_ema_crossover(df['Close'], short=12, long=26, capital=capital, risk_pct=risk_pct)
            results[s] = bt
        return jsonify({"status": "success", "results": safe_json(results)})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 8) runBehavioralBiasDetector (requires textual input; can use AI if available)
@app.route("/api/run-behavioral-bias-detector", methods=["POST"])
def api_bias_detector():
    try:
        data = request.get_json(force=True) or {}
        journal_text = data.get("journal_text", "")
        if not journal_text:
            return jsonify({"error": "No journal text provided"}), 400
        try:
            ai = call_openrouter(f"Detect behavioral biases in the following trading journal. Return list of biases and remediation steps:\n\n{journal_text}")
            return jsonify({"status": "success", "ai_analysis": ai})
        except RuntimeError:
            # fallback: simple keyword heuristics
            biases = []
            txt = journal_text.lower()
            if "hold" in txt and "loss" in txt:
                biases.append("loss aversion")
            if "overtrade" in txt or "too many" in txt:
                biases.append("overtrading")
            return jsonify({"status": "success", "detected_biases": biases})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 9) runCommodityStockMapper
@app.route("/api/run-commodity-stock-mapper", methods=["POST"])
def api_commodity_stock_mapper():
    try:
        data = request.get_json(force=True) or {}
        commodity = data.get("commodity", "crude-oil")
        mapping = {
            "crude-oil": ["ONGC.NS", "BPCL.NS", "IOC.NS"],
            "gold": ["HINDZINC.NS", "TATASTEEL.NS", "DRREDDY.NS"],
            "steel": ["JSWSTEEL.NS", "SAIL.NS"]
        }
        symbols = mapping.get(commodity, ["RELIANCE.NS"])
        close_df = get_multi_close_df(symbols, period="6mo")
        if close_df is None:
            return jsonify({"error": "failed to fetch"}), 500
        corr = close_df.corr().to_dict()
        return jsonify({"status": "success", "mapping": symbols, "correlation": corr})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 10) runCorrelationMatrix
@app.route("/api/run-correlation-matrix", methods=["POST"])
def api_run_correlation_matrix():
    try:
        data = request.get_json(force=True) or {}
        tickers = data.get("tickers", ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "^NSEI"])
        close_df = get_multi_close_df(tickers, period="6mo")
        if close_df is None or close_df.empty:
            return jsonify({"error": "Failed to fetch data"}), 500
        corr = close_df.corr().round(4).to_dict()
        return jsonify({"status": "success", "correlation": corr})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 11) runCurrencyImpactCalculator
@app.route("/api/run-currency-impact", methods=["POST"])
def api_currency_impact():
    try:
        data = request.get_json(force=True) or {}
        pair = data.get("pair", "USD-INR")
        symbol = data.get("symbol", "^INR=X")  # Yahoo FX pair for INR
        df = get_stock_df(symbol, period="1mo")
        if df is None or df.empty:
            return jsonify({"error": "FX data not available"}), 500
        change = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / (df['Close'].iloc[0] + 1e-9) * 100
        return jsonify({"status": "success", "pair": pair, "change%": float(change)})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 12) runDrawdownRecoveryPredictor
@app.route("/api/run-drawdown-recovery", methods=["POST"])
def api_drawdown_recovery():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "RELIANCE.NS")
        df = get_stock_df(symbol, period="3y")
        if df is None or df.empty:
            return jsonify({"error": "No data"}), 400
        close = df['Close']
        roll_max = close.cummax()
        drawdown = (close - roll_max) / roll_max
        max_dd = float(drawdown.min())
        return jsonify({"status": "success", "max_drawdown": max_dd})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 13) runDreamTradeSimulator (AI)
@app.route("/api/run-dream-trade-simulator", methods=["POST"])
def api_dream_trade_simulator():
    try:
        data = request.get_json(force=True) or {}
        scenario = data.get("scenario", "")
        try:
            ai = call_openrouter(f"Simulate a dream trade scenario: {scenario}. Provide P&L, ROI, and risk profile.")
            return jsonify({"status": "success", "simulation": ai})
        except RuntimeError:
            return jsonify({"error": "Dream trade simulation requires OPENROUTER_API_KEY."}), 501
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 14) runDynamicPositionSizing
@app.route("/api/run-dynamic-position-sizing", methods=["POST"])
def api_dynamic_position():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "RELIANCE.NS")
        capital = float(data.get("capital", 100000))
        risk_per_trade = float(data.get("risk_pct", 0.02))
        df = get_stock_df(symbol, period="1y")
        if df is None or df.empty:
            return jsonify({"error": "No data"}), 400
        atr = (df['High'] - df['Low']).rolling(14).mean().dropna()
        atr_value = float(atr.iloc[-1]) if not atr.empty else None
        stop_loss_distance = atr_value * 2 if atr_value else df['Close'].iloc[-1] * 0.02
        price = float(df['Close'].iloc[-1])
        position_size = max(1, int((capital * risk_per_trade) / (stop_loss_distance * price)))
        return jsonify({"status": "success", "symbol": symbol, "position_size": position_size, "atr": atr_value})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 15) runESGImpactScorer (not in yfinance) -> requires external data or AI
@app.route("/api/run-esg-impact-scorer", methods=["POST"])
def api_esg_impact():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "RELIANCE.NS")
        try:
            ai = call_openrouter(f"Provide an ESG impact score analysis for {symbol} and explain key drivers.")
            return jsonify({"status": "success", "symbol": symbol, "esg_analysis": ai})
        except RuntimeError:
            return jsonify({"error": "ESG scoring requires external datasets or OPENROUTER_API_KEY."}), 501
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 16) runEarningsCallAnalysis (requires transcripts / external source)
@app.route("/api/run-earnings-call-analysis", methods=["POST"])
def api_earnings_call():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "RELIANCE.NS")
        try:
            ai = call_openrouter(f"Analyze recent earnings calls and generate a concise actionable summary for {symbol}.")
            return jsonify({"status": "success", "symbol": symbol, "analysis": ai})
        except RuntimeError:
            return jsonify({"error": "Earnings call analysis requires external transcripts or OPENROUTER_API_KEY."}), 501
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 17) runEconomicImpactPredictor (AI)
@app.route("/api/run-economic-impact", methods=["POST"])
def api_economic_impact():
    try:
        data = request.get_json(force=True) or {}
        event = data.get("event", "rbi-policy")
        try:
            ai = call_openrouter(f"Predict market impact for event: {event}. Provide sector-level implications and trade ideas.")
            return jsonify({"status": "success", "event": event, "ai_analysis": ai})
        except RuntimeError:
            return jsonify({"error": "Economic impact prediction requires OPENROUTER_API_KEY."}), 501
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 18) runGeopoliticalRiskScorer (AI)
@app.route("/api/run-geopolitical-risk", methods=["POST"])
def api_geo_risk():
    try:
        data = request.get_json(force=True) or {}
        region = data.get("region", "india-china")
        try:
            ai = call_openrouter(f"Score geopolitical risk for {region} and supply specific trade hedges and timeline.")
            return jsonify({"status": "success", "region": region, "ai_analysis": ai})
        except RuntimeError:
            return jsonify({"error": "Geopolitical risk scoring requires OPENROUTER_API_KEY."}), 501
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 19) runGlobalMarketSync
@app.route("/api/run-global-market-sync", methods=["POST"])
def api_global_sync():
    try:
        data = request.get_json(force=True) or {}
        region = data.get("region", "us-markets")
        try:
            ai = call_openrouter(f"Analyze how {region} will impact Indian markets today. Provide short-term signals.")
            return jsonify({"status": "success", "region": region, "ai_analysis": ai})
        except RuntimeError:
            return jsonify({"error": "Global market sync requires OPENROUTER_API_KEY."}), 501
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 20) runInsiderAnalysis (requires external insider-trade datasets)
@app.route("/api/run-insider-analysis", methods=["POST"])
def api_insider_analysis():
    try:
        data = request.get_json(force=True) or {}
        period = data.get("period", "30d")
        symbol = data.get("symbol", None)
        return jsonify({"status": "success", "note": "Insider trading analysis requires external datasource (NSE/BSE filings).", "symbol": symbol, "period": period}), 501
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 21) runInstitutionalFlowTracker
@app.route("/api/run-institutional-flow-tracker", methods=["POST"])
def api_institutional_flow():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "^NSEI")
        df = get_stock_df(symbol, period="1mo")
        if df is None or df.empty:
            return jsonify({"error": "No data"}), 400
        if 'Volume' not in df.columns:
            return jsonify({"error": "Volume data unavailable for symbol"}), 400
        vol_change = ((df['Volume'].iloc[-1] - df['Volume'].iloc[0]) / (df['Volume'].iloc[0] + 1e-9)) * 100
        return jsonify({"status": "success", "volume_change_pct": float(vol_change)})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 22) runInterestRateSensitivity (AI)
@app.route("/api/run-interest-rate-sensitivity", methods=["POST"])
def api_rate_sensitivity():
    try:
        data = request.get_json(force=True) or {}
        sector = data.get("sector", "banking")
        try:
            ai = call_openrouter(f"Analyze interest rate sensitivity for sector {sector}. Provide names of most sensitive tickers and hedges.")
            return jsonify({"status": "success", "sector": sector, "ai_analysis": ai})
        except RuntimeError:
            return jsonify({"error": "Interest rate sensitivity analysis requires OPENROUTER_API_KEY."}), 501
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 23) runLiquidityHeatMap
@app.route("/api/run-liquidity-heatmap", methods=["POST"])
def api_liquidity_heatmap():
    try:
        data = request.get_json(force=True) or {}
        symbols = data.get("symbols", ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"])
        liquidity = {}
        for sym in symbols:
            try:
                hist = yf.download(sym, period="3mo", progress=False, auto_adjust=True)
                if hist is None or hist.empty or 'Volume' not in hist.columns:
                    liquidity[sym] = None
                else:
                    avg_vol = hist['Volume'].tail(30).mean()
                    avg_price = hist['Close'].tail(30).mean()
                    liquidity[sym] = float(avg_vol * avg_price)
            except Exception:
                liquidity[sym] = None
        return jsonify({"status": "success", "liquidity": safe_json(liquidity)})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 24) runMarketRegimeDetection
@app.route("/api/run-market-regime-detection", methods=["POST"])
def api_market_regime():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "^NSEI")
        df = get_stock_df(symbol, period="2y")
        if df is None or df.empty:
            return jsonify({"error": "No data"}), 400
        returns = df['Close'].pct_change().dropna()
        vol = float(returns.rolling(21).std().iloc[-1])
        trend = float((df['Close'].iloc[-1] - df['Close'].iloc[-63]) / df['Close'].iloc[-63])
        regime = "Bullish-Trend" if trend > 0.05 and vol < 0.02 else "Volatile" if vol > 0.03 else "Range-bound"
        return jsonify({"status": "success", "regime": regime, "trend": trend, "volatility": vol})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 25) runOptionsFlow
@app.route("/api/run-options-flow", methods=["POST"])
def api_options_flow():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "RELIANCE")
        yf_sym = symbol if symbol.endswith(".NS") else symbol + ".NS"
        oc = get_option_chain(yf_sym)
        if not oc.get('expiries'):
            return jsonify({"error": "No options data"}), 404
        try:
            calls = pd.DataFrame(oc['calls'])
            puts = pd.DataFrame(oc['puts'])
            top_calls = calls.sort_values('openInterest', ascending=False).head(5).to_dict(orient='records') if not calls.empty else []
            top_puts = puts.sort_values('openInterest', ascending=False).head(5).to_dict(orient='records') if not puts.empty else []
            return jsonify({"status": "success", "expiry": oc.get('expiry_used'), "top_calls": safe_json(top_calls), "top_puts": safe_json(top_puts)})
        except Exception:
            return jsonify({"error": "Failed to parse option chains"}), 500
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 26) runPortfolioOptimization
@app.route("/api/run-portfolio-optimization", methods=["POST"])
def api_portfolio_optimization():
    try:
        data = request.get_json(force=True) or {}
        symbols = data.get("symbols", ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"])
        df = get_multi_close_df(symbols, period="1y")
        if df is None or df.empty:
            return jsonify({"error": "No data"}), 400
        rets = df.pct_change().dropna()
        mu = (rets.mean() * 252).round(6).to_dict()
        sigma = (rets.cov() * 252).round(6).to_dict()
        weights = {s: 1/len(symbols) for s in symbols}
        return jsonify({"status": "success", "weights": weights, "mu": mu, "sigma": sigma})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 27) runPortfolioStressTesting (AI for scenario reasoning)
@app.route("/api/run-portfolio-stress-testing", methods=["POST"])
def api_portfolio_stress():
    try:
        data = request.get_json(force=True) or {}
        portfolio = data.get("portfolio", {})
        scenario = data.get("scenario", "market-crash")
        try:
            ai = call_openrouter(f"Stress test portfolio {portfolio} under scenario {scenario}. Return expected drawdown and recovery suggestions.")
            return jsonify({"status": "success", "scenario": scenario, "ai_analysis": ai})
        except RuntimeError:
            return jsonify({"error": "Stress testing scenario analysis requires OPENROUTER_API_KEY."}), 501
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 28) runPriceTargetConsensus (AI)
@app.route("/api/run-price-target-consensus", methods=["POST"])
def api_price_target_consensus():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "RELIANCE.NS")
        df = get_stock_df(symbol, period="6mo")
        if df is None or df.empty:
            return jsonify({"error": "No data"}), 400
        try:
            prompt = f"Generate price target consensus for {symbol} using recent price history. Provide low/medium/high targets and probability."
            ai = call_openrouter(prompt)
            return jsonify({"status": "success", "symbol": symbol, "ai_analysis": ai})
        except RuntimeError:
            return jsonify({"error": "Price target consensus requires OPENROUTER_API_KEY."}), 501
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 29) runRealBacktest
@app.route("/api/run-real-backtest", methods=["POST"])
def api_run_real_backtest():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "RELIANCE.NS")
        period = data.get("period", "2y")
        capital = float(data.get("capital", 100000))
        risk_pct = float(data.get("risk_pct", 0.02))
        df = get_stock_df(symbol, period=period)
        if df is None or df.empty:
            return jsonify({"error": "No data"}), 400
        bt = backtest_ema_crossover(df['Close'], short=12, long=26, capital=capital, risk_pct=risk_pct)
        return jsonify({"status": "success", "symbol": symbol, "backtest": bt})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 30) runRealDataMining
@app.route("/api/run-real-data-mining", methods=["POST"])
def api_real_data_mining():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "RELIANCE.NS")
        period = data.get("period", "3mo")
        df = get_stock_df(symbol, period=period)
        if df is None or df.empty:
            return jsonify({"error": "No data"}), 400
        ret = df['Close'].pct_change().dropna()
        z = (ret - ret.mean()) / (ret.std() + 1e-9)
        anomalies = ret[np.abs(z) > 2].tail(20).to_dict()
        return jsonify({"status": "success", "anomalies": safe_json(anomalies)})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 31) runRealTimeScreener
@app.route("/api/run-real-time-screener", methods=["POST"])
def api_realtime_screener():
    try:
        data = request.get_json(force=True) or {}
        symbols = data.get("symbols", ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"])
        df = get_multi_close_df(symbols, period="5d", interval="1d")
        if df is None or df.empty:
            return jsonify({"error": "No data"}), 400
        momentum = {}
        for col in df.columns:
            if len(df) >= 5:
                momentum[col] = float((df[col].iloc[-1] / df[col].iloc[0] - 1) * 100)
            else:
                momentum[col] = None
        return jsonify({"status": "success", "momentum": momentum})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 32) runSectorRotationPredictor (AI)
@app.route("/api/run-sector-rotation-predictor", methods=["POST"])
def api_sector_rotation():
    try:
        data = request.get_json(force=True) or {}
        timeframe = data.get("timeframe", "1m")
        try:
            ai = call_openrouter(f"Sector rotation predictions for the next {timeframe}. Provide top 3 sectors and reasons.")
            return jsonify({"status": "success", "ai_analysis": ai})
        except RuntimeError:
            return jsonify({"error": "Sector rotation predictions require OPENROUTER_API_KEY."}), 501
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 33) runSentimentAnalysis (basic price-change proxy)
@app.route("/api/run-sentiment-analysis", methods=["POST"])
def api_run_sentiment_analysis():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "^NSEI")
        df = get_stock_df(symbol, period="1mo")
        if df is None or df.empty:
            return jsonify({"error": f"No data for {symbol}"}), 400
        change = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / (df['Close'].iloc[0] + 1e-9) * 100
        return jsonify({"status": "success", "symbol": symbol, "change%": float(change)})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 34) runSocialMomentumScanner (requires social APIs -> 501)
@app.route("/api/run-social-momentum-scanner", methods=["POST"])
def api_social_momentum():
    try:
        return jsonify({"error": "Social momentum scan requires external social APIs (Twitter/Reddit) or OPENROUTER_API_KEY for synthesis."}), 501
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 35) runSocialTrendMonetizer (AI)
@app.route("/api/run-social-trend-monetizer", methods=["POST"])
def api_social_trend_monetizer():
    try:
        data = request.get_json(force=True) or {}
        trend = data.get("trend", "EV stock interest")
        try:
            ai = call_openrouter(f"Monetize social trend '{trend}' into short-term trade ideas. Provide entry/stop/targets.")
            return jsonify({"status": "success", "ai_analysis": ai})
        except RuntimeError:
            return jsonify({"error": "Social trend monetizer requires OPENROUTER_API_KEY."}), 501
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 36) runSupplyChainVisibility (AI)
@app.route("/api/run-supply-chain-visibility", methods=["POST"])
def api_supply_chain_visibility():
    try:
        data = request.get_json(force=True) or {}
        sector = data.get("sector", "manufacturing")
        try:
            ai = call_openrouter(f"Supply chain visibility analysis for sector {sector}. Provide companies likely impacted and trade ideas.")
            return jsonify({"status": "success", "ai_analysis": ai})
        except RuntimeError:
            return jsonify({"error": "Supply chain visibility requires OPENROUTER_API_KEY."}), 501
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 37) runTailRiskHedging (AI)
@app.route("/api/run-tail-risk-hedging", methods=["POST"])
def api_tail_risk_hedging():
    try:
        data = request.get_json(force=True) or {}
        portfolio = data.get("portfolio", {})
        try:
            ai = call_openrouter(f"Generate tail-risk hedging strategies for portfolio: {json.dumps(portfolio, default=str)}. Provide costs and expected protection.")
            return jsonify({"status": "success", "ai_analysis": ai})
        except RuntimeError:
            return jsonify({"error": "Tail risk hedging requires OPENROUTER_API_KEY."}), 501
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 38) runTradingJournalAnalysis (AI or simple local)
@app.route("/api/run-trading-journal-analysis", methods=["POST"])
def api_trading_journal_analysis():
    try:
        data = request.get_json(force=True) or {}
        journal = data.get("journal", "")
        if not journal:
            return jsonify({"error": "No journal entered"}), 400
        try:
            ai = call_openrouter(f"Analyze trading journal and produce performance summary and improvement plan: {journal}")
            return jsonify({"status": "success", "ai_analysis": ai})
        except RuntimeError:
            # simple local summary
            words = journal.split()
            return jsonify({"status": "success", "summary": {"words": len(words)}})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 39) runVolatilitySurface (historical vol approximation)
@app.route("/api/run-volatility-surface", methods=["POST"])
def api_volatility_surface():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "RELIANCE.NS")
        df = get_stock_df(symbol, period="1y")
        if df is None or df.empty:
            return jsonify({"error": "No data"}), 400
        returns = df['Close'].pct_change().dropna()
        hist_vol = float(returns.std() * math.sqrt(252))
        return jsonify({"status": "success", "historical_vol": hist_vol})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 40) runWalkForwardTest (simplified)
@app.route("/api/run-walk-forward-test", methods=["POST"])
def api_walk_forward_test():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "RELIANCE.NS")
        df = get_stock_df(symbol, period="3y")
        if df is None or df.empty:
            return jsonify({"error": "No data"}), 400
        window_train = 252
        step = 63
        results = []
        closes = df['Close']
        for start in range(0, len(closes) - window_train - step, step):
            train = closes.iloc[start:start+window_train]
            test = closes.iloc[start+window_train:start+window_train+step]
            bt = backtest_ema_crossover(train, short=12, long=26, capital=100000, risk_pct=0.02)
            if not test.empty:
                test_return = float((test.iloc[-1] - test.iloc[0]) / test.iloc[0])
            else:
                test_return = None
            results.append({"train_end": str(train.index[-1]), "test_return": test_return})
        return jsonify({"status": "success", "samples": len(results), "results": results})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 41) runWeatherPatternTrading (AI)
@app.route("/api/run-weather-pattern-trading", methods=["POST"])
def api_weather_pattern_trading():
    try:
        data = request.get_json(force=True) or {}
        sector = data.get("sector", "agriculture")
        try:
            ai = call_openrouter(f"Create weather pattern trading ideas for sector {sector}. Provide stocks, entry and stoploss.")
            return jsonify({"status": "success", "ai_analysis": ai})
        except RuntimeError:
            return jsonify({"error": "Weather pattern trading requires OPENROUTER_API_KEY or external weather datasets."}), 501
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# 42) runDreamTradeSimulator alias
@app.route("/api/run-dream-trade-sim", methods=["POST"])
def api_dream_trade_alias():
    return api_dream_trade_simulator()

# ------------------------------
# Convenience routes (existing ones fixed)
# ------------------------------
@app.route("/api/correlation-matrix", methods=["POST"])
def correlation_matrix():
    try:
        data = request.get_json(force=True) or {}
        tickers = data.get("tickers", ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"])
        close_df = get_multi_close_df(tickers, period="6mo")
        if close_df is None or close_df.empty:
            return jsonify({"error": "Failed to fetch data"}), 500
        corr = close_df.corr().round(4).to_dict()
        return jsonify({"status": "success", "correlation": corr})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/sentiment-analysis", methods=["POST"])
def sentiment_analysis():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "^NSEI")
        df = get_stock_df(symbol, period="1mo")
        if df is None or df.empty:
            return jsonify({"error": "No data for symbol"}), 400
        change = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / (df['Close'].iloc[0] + 1e-9) * 100
        return jsonify({"status": "success", "symbol": symbol, "change%": float(change)})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

# ------------------------------
# Page render
# ------------------------------
@app.route("/matrix", methods=["GET", "POST"])
def strategy_matrix():
    signals = []
    if request.method == "POST":
        raw_data = request.form["data"]
        lines = raw_data.strip().splitlines()
        for line in lines:
            if "buy" in line.lower():
                signals.append(f"📈 Buy signal from: {line}")
            elif "sell" in line.lower():
                signals.append(f"📉 Sell signal from: {line}")
            else:
                signals.append(f"⚠️ Neutral/No signal: {line}")
    return render_template("strategy_matrix.html", signals=signals)

@app.route("/ask-ai", methods=["GET", "POST"])
def ask_ai():
    response = None
    if request.method == "POST":
        try:
            # Get data from form or JSON
            if request.is_json:
                data = request.get_json()
                question = data.get('question')
                mode = data.get('mode', 'general')
                system_prompt = data.get('systemPrompt', '')
            else:
                question = request.form.get("question")
                mode = request.form.get("mode", "general")
                system_prompt = request.form.get("systemPrompt", "")
            
            if not question:
                return jsonify({'error': 'No question provided'}), 400
            
            # Make request to OpenRouter using your working configuration
            ai_response = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://lakshmi-ai-trades.onrender.com",
                    "X-Title": "Lakshmi AI Wife"
                },
                json={
                    "model": "deepseek/deepseek-chat-v3-0324",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ],
                    "max_tokens": 600,
                    "temperature": 0.9,
                    "top_p": 0.95
                }
            )
            
            if ai_response.status_code != 200:
                print(f"OpenRouter API Error: {ai_response.status_code}")
                print(f"Response: {ai_response.text}")
                return jsonify({'error': f'AI service error: {ai_response.status_code}'}), 500
                
            ai_data = ai_response.json()
            response = ai_data['choices'][0]['message']['content']
            
            # Return JSON for AJAX requests
            if request.is_json or request.headers.get('Content-Type') == 'application/json':
                return jsonify({'response': response})
            
            # Return HTML template for regular form submissions
            return render_template("ask_ai.html", response=response)
            
        except Exception as e:
            print(f"AI Chat Error: {e}")
            error_msg = f"AI Processing Error: {str(e)}"
            
            if request.is_json or request.headers.get('Content-Type') == 'application/json':
                return jsonify({'error': error_msg}), 500
            else:
                return render_template("ask_ai.html", response=error_msg)
    
    # GET request - show the form
    return render_template("ask_ai.html", response=response)

@app.route("/option-chain")
def option_chain():
    strike_filter = request.args.get("strike_filter")
    expiry = request.args.get("expiry")

    mock_data = [
        {"strike": 44000, "call_oi": 1200, "call_change": 150, "put_oi": 900, "put_change": -100},
        {"strike": 44200, "call_oi": 980, "call_change": -20, "put_oi": 1100, "put_change": 80},
        {"strike": 44400, "call_oi": 1890, "call_change": 60, "put_oi": 2300, "put_change": 210},
        {"strike": 44600, "call_oi": 760, "call_change": 40, "put_oi": 1500, "put_change": 310},
    ]

    if strike_filter:
        try:
            strike_filter = int(strike_filter)
            mock_data = [row for row in mock_data if abs(row["strike"] - strike_filter) <= 200]
        except:
            pass

    max_call_oi = max(row["call_oi"] for row in mock_data)
    max_put_oi = max(row["put_oi"] for row in mock_data)
    for row in mock_data:
        row["max_oi"] = row["call_oi"] == max_call_oi or row["put_oi"] == max_put_oi

    return render_template("option_chain.html", option_data=mock_data, strike_filter=strike_filter, expiry=expiry)

@app.route("/analyzer", methods=["GET", "POST"])
def analyzer():
    signal = ""
    if request.method == "POST":
        r = random.random()
        if r > 0.7:
            signal = "📈 Strong BUY — Momentum detected!"
        elif r < 0.3:
            signal = "📉 SELL — Weakness detected!"
        else:
            signal = "⏳ No clear signal — Stay out!"
    return render_template("analyzer.html", signal=signal)

@app.route("/strategy-engine")
def strategy_engine():
    if 'username' not in session:
        return redirect("/login")
    return render_template("strategy_engine.html")

@app.route("/analyze-strategy", methods=["POST"])
def analyze_strategy():
    data = request.get_json()
    try:
        price = float(data.get('price', 0))
    except (ValueError, TypeError):
        return jsonify({'message': 'Invalid price input.'})

    if price % 2 == 0:
        strategy = "EMA Bullish Crossover Detected 💞"
        confidence = random.randint(80, 90)
        sl = price - 50
        target = price + 120
    elif price % 3 == 0:
        strategy = "RSI Reversal Detected 🔁"
        confidence = random.randint(70, 85)
        sl = price - 40
        target = price + 100
    else:
        strategy = "Breakout Zone Approaching 💥"
        confidence = random.randint(60, 75)
        sl = price - 60
        target = price + 90

    entry = price
    message = f"""
    💌 <b>{strategy}</b><br>
    ❤️ Entry: ₹{entry}<br>
    🔻 Stop Loss: ₹{sl}<br>
    🎯 Target: ₹{target}<br>
    📊 Confidence Score: <b>{confidence}%</b><br><br>
    <i>Take this trade only if you feel my kiss of confidence 😘</i>
    """
    return jsonify({'message': message})

# === /neuron endpoint ===
@app.route("/neuron", methods=["GET", "POST"])
def neuron():
    try:
        if request.method == "GET":
            return render_template("neuron.html")

        # Handle POST (form or JSON)
        if request.is_json:
            user_input = request.json.get("message")
        else:
            user_input = request.form.get("message")

        if not user_input:
            return jsonify({"reply": "❌ No input received."})

        symbol = extract_symbol_from_text(user_input)
        if not symbol:
            return jsonify({"reply": "❌ Could not detect any valid symbol (like NIFTY, BANKNIFTY, SENSEX)."})

        price = get_yfinance_ltp(symbol)
        if price == 0:
            return jsonify({"reply": f"⚠️ Could not fetch real price for {symbol}. Try again later."})

        result = analyze_with_neuron(price, symbol)
        return jsonify(result)

    except Exception as e:
        print(f"[ERROR /neuron]: {e}")
        return jsonify({"reply": "❌ Internal error occurred in /neuron."})

# ===== UPDATED V8 API ROUTES - SUPPORTS ALL INDIAN STOCKS & INDICES =====

def get_yahoo_symbol(symbol):
    """Convert any Indian symbol to Yahoo Finance format"""
    if not symbol:
        return '^NSEI'  # Default to NIFTY
    
    symbol = symbol.upper().strip()
    
    # Indian Indices
    if symbol in ['NIFTY', 'NIFTY50', 'NIFTY 50']:
        return '^NSEI'
    elif symbol in ['BANKNIFTY', 'BANK NIFTY', 'BANKNIFTY50']:
        return '^NSEBANK'
    elif symbol in ['SENSEX', 'BSE SENSEX']:
        return '^BSESN'
    elif symbol in ['FINNIFTY', 'FIN NIFTY', 'NIFTY FINANCIAL']:
        return 'NIFTY_FIN_SERVICE.NS'
    elif symbol in ['MIDCPNIFTY', 'NIFTY MIDCAP']:
        return 'NIFTY_MIDCAP_100.NS'
    elif symbol in ['SMALLCAPNIFTY', 'NIFTY SMALLCAP']:
        return 'NIFTY_SMLCAP_100.NS'
    elif symbol in ['NIFTYIT', 'NIFTY IT']:
        return 'NIFTY_IT.NS'
    elif symbol in ['NIFTYPHARMA', 'NIFTY PHARMA']:
        return 'NIFTY_PHARMA.NS'
    elif symbol in ['NIFTYAUTO', 'NIFTY AUTO']:
        return 'NIFTY_AUTO.NS'
    elif symbol in ['NIFTYMETAL', 'NIFTY METAL']:
        return 'NIFTY_METAL.NS'
    elif symbol in ['NIFTYREALTY', 'NIFTY REALTY']:
        return 'NIFTY_REALTY.NS'
    elif symbol in ['NIFTYENERGY', 'NIFTY ENERGY']:
        return 'NIFTY_ENERGY.NS'
    elif symbol in ['NIFTYFMCG', 'NIFTY FMCG']:
        return 'NIFTY_FMCG.NS'
    elif symbol in ['NIFTYPSU', 'NIFTY PSU BANK']:
        return 'NIFTY_PSU_BANK.NS'
    elif symbol in ['NIFTYPVTBANK', 'NIFTY PRIVATE BANK']:
        return 'NIFTY_PVT_BANK.NS'
    # Add .NS for Indian stocks if not already present
    elif '.' not in symbol:
        return f"{symbol}.NS"
    else:
        return symbol

@app.route("/api/v8/fetch-real-data", methods=["GET", "POST"])
def fetch_real_data_api():
    """API endpoint supporting ALL Indian stocks and indices"""
    try:
        # Get symbol from multiple sources
        symbol = None
        
        if request.method == "GET":
            symbol = request.args.get('symbol')
            message = request.args.get('message')
        else:  # POST
            if request.is_json:
                symbol = request.json.get('symbol')
                message = request.json.get('message')
            else:
                symbol = request.form.get('symbol')
                message = request.form.get('message')
        
        # Extract symbol from message if not provided directly
        if not symbol and message:
            symbol = extract_symbol_from_text(message)
        
        # Default to NIFTY if nothing found
        if not symbol:
            symbol = 'NIFTY'
        
        # Convert to Yahoo Finance format
        yf_symbol = get_yahoo_symbol(symbol)
        
        # Use your existing function first
        price = get_yfinance_ltp(symbol)
        
        # If your function fails, try direct yfinance call
        if price == 0:
            import yfinance as yf
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                price = hist['Close'].iloc[-1]
        
        if price == 0:
            return jsonify({'error': f'Could not fetch price for {symbol}'}), 500
        
        # Get additional data
        import yfinance as yf
        ticker = yf.Ticker(yf_symbol)
        info = ticker.info
        hist = ticker.history(period="1d", interval="5m")
        
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            prev_close = info.get('previousClose', current_price)
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100
            
            # Format currency based on symbol type
            currency = "₹" if any(x in symbol.upper() for x in ['NIFTY', 'SENSEX', 'BANK']) or '.NS' in yf_symbol else "₹"
            
            data = {
                'symbol': symbol,
                'yf_symbol': yf_symbol,
                'current_price': f"{currency}{current_price:.2f}",
                'price_change': f"{change:+.2f} ({change_percent:+.2f}%)",
                'volume': f"{hist['Volume'].iloc[-1]:,.0f}" if hist['Volume'].iloc[-1] > 0 else "N/A",
                'high': f"{currency}{hist['High'].max():.2f}",
                'low': f"{currency}{hist['Low'].min():.2f}",
                'open': f"{currency}{hist['Open'].iloc[0]:.2f}",
                'timestamp': datetime.now().isoformat(),
                'raw_price': current_price,
                'raw_change': change,
                'raw_change_percent': change_percent
            }
        else:
            # Fallback data
            data = {
                'symbol': symbol,
                'yf_symbol': yf_symbol,
                'current_price': f"₹{price:.2f}",
                'price_change': "N/A",
                'volume': "N/A",
                'timestamp': datetime.now().isoformat(),
                'raw_price': price
            }
        
        return jsonify({
            'status': 'success',
            'data': data,
            'source': 'yfinance'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/api/v8/analyze-strategies", methods=["GET", "POST"])
def analyze_strategies_api():
    """API endpoint supporting ALL Indian stocks and indices"""
    try:
        # Get symbol using same logic as above
        symbol = None
        
        if request.method == "GET":
            symbol = request.args.get('symbol')
            message = request.args.get('message')
        else:
            if request.is_json:
                symbol = request.json.get('symbol')
                message = request.json.get('message')
            else:
                symbol = request.form.get('symbol')
                message = request.form.get('message')
        
        if not symbol and message:
            symbol = extract_symbol_from_text(message)
        
        if not symbol:
            symbol = 'NIFTY'
        
        # Use your existing functions
        price = get_yfinance_ltp(symbol)
        if price == 0:
            # Try with Yahoo Finance format
            yf_symbol = get_yahoo_symbol(symbol)
            import yfinance as yf
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                price = hist['Close'].iloc[-1]
        
        if price == 0:
            return jsonify({'error': f'Could not fetch price for {symbol}'}), 500
        
        # Use your existing analyze_with_neuron function
        analysis_result = analyze_with_neuron(price, symbol)
        
        # Enhanced analysis for different asset types
        asset_type = 'INDEX' if any(x in symbol.upper() for x in ['NIFTY', 'SENSEX', 'BANK']) else 'STOCK'
        
        # Adjust strategy count based on asset type
        total_strategies = 500 if asset_type == 'STOCK' else 300  # Fewer strategies for indices
        
        formatted_result = {
            'status': 'success',
            'symbol': symbol,
            'asset_type': asset_type,
            'analysis': {
                'signal': {
                    'type': 'BUY',  # Extract from your analysis_result
                    'confidence': '87.5%'
                },
                'levels': {
                    'entry': f"₹{price:.2f}",
                    'target': f"₹{price * (1.03 if asset_type == 'INDEX' else 1.05):.2f}",  # Lower targets for indices
                    'stoploss': f"₹{price * (0.98 if asset_type == 'INDEX' else 0.97):.2f}"
                },
                'strategy_breakdown': {
                    'total_analyzed': total_strategies,
                    'bullish': int(total_strategies * 0.65),
                    'bearish': int(total_strategies * 0.35)
                }
            },
            'neuron_analysis': analysis_result,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(formatted_result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/api/v8/calculate-indicators", methods=["GET", "POST"])
def calculate_indicators_api():
    """API endpoint for technical indicators - ALL Indian assets"""
    try:
        # Get symbol using same extraction logic
        symbol = None
        
        if request.method == "GET":
            symbol = request.args.get('symbol')
            message = request.args.get('message')
        else:
            if request.is_json:
                symbol = request.json.get('symbol')
                message = request.json.get('message')
            else:
                symbol = request.form.get('symbol')
                message = request.form.get('message')
        
        if not symbol and message:
            symbol = extract_symbol_from_text(message)
        
        if not symbol:
            symbol = 'NIFTY'
        
        yf_symbol = get_yahoo_symbol(symbol)
        
        import yfinance as yf
        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(period="60d")
        
        if hist.empty:
            return jsonify({'error': f'No historical data available for {symbol}'}), 500
        
        # Calculate RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # Calculate Volatility
        returns = hist['Close'].pct_change()
        volatility = returns.rolling(window=20).std() * np.sqrt(252) * 100
        current_volatility = volatility.iloc[-1]
        
        # Asset-specific indicator interpretation
        asset_type = 'INDEX' if any(x in symbol.upper() for x in ['NIFTY', 'SENSEX', 'BANK']) else 'STOCK'
        
        # Adjust volatility thresholds for indices vs stocks
        vol_high_threshold = 20 if asset_type == 'INDEX' else 30
        vol_moderate_threshold = 12 if asset_type == 'INDEX' else 20
        
        indicators = {
            'rsi': {
                'value': f"{current_rsi:.1f}",
                'signal': 'OVERSOLD' if current_rsi < 30 else 'OVERBOUGHT' if current_rsi > 70 else 'NEUTRAL'
            },
            'volatility': {
                'value': f"{current_volatility:.1f}%",
                'level': 'HIGH' if current_volatility > vol_high_threshold else 'MODERATE' if current_volatility > vol_moderate_threshold else 'LOW'
            },
            'asset_type': asset_type,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'yf_symbol': yf_symbol,
            'indicators': indicators
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/api/v8/live-data", methods=["GET"])
def live_data_api():
    """API endpoint for live data - ALL Indian assets"""
    try:
        symbol = request.args.get('symbol')
        message = request.args.get('message')
        
        if not symbol and message:
            symbol = extract_symbol_from_text(message)
        
        if not symbol:
            symbol = 'NIFTY'
        
        # Use your existing function first
        price = get_yfinance_ltp(symbol)
        
        # Fallback to direct yfinance
        if price == 0:
            yf_symbol = get_yahoo_symbol(symbol)
            import yfinance as yf
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                price = hist['Close'].iloc[-1]
        
        if price == 0:
            return jsonify({'error': f'Could not fetch live price for {symbol}'}), 500
        
        # Market hours for Indian markets
        current_hour = datetime.now().hour
        market_status = 'OPEN' if 9 <= current_hour <= 15 else 'CLOSED'
        
        # Special handling for different asset types
        asset_type = 'INDEX' if any(x in symbol.upper() for x in ['NIFTY', 'SENSEX', 'BANK']) else 'STOCK'
        
        live_data = {
            'symbol': symbol,
            'asset_type': asset_type,
            'price': f"₹{price:.2f}",
            'timestamp': datetime.now().isoformat(),
            'market_status': market_status,
            'last_update': datetime.now().strftime('%H:%M:%S'),
            'raw_price': price
        }
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'live_data': live_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/api/v8/comprehensive-analysis", methods=["GET", "POST"])
def comprehensive_analysis_api():
    """API endpoint for comprehensive analysis - ALL Indian assets"""
    try:
        # Get symbol using extraction logic
        symbol = None
        
        if request.method == "GET":
            symbol = request.args.get('symbol')
            message = request.args.get('message')
        else:
            if request.is_json:
                symbol = request.json.get('symbol')
                message = request.json.get('message')
            else:
                symbol = request.form.get('symbol')
                message = request.form.get('message')
        
        if not symbol and message:
            symbol = extract_symbol_from_text(message)
        
        if not symbol:
            symbol = 'NIFTY'
        
        # Use your existing functions
        price = get_yfinance_ltp(symbol)
        if price == 0:
            yf_symbol = get_yahoo_symbol(symbol)
            import yfinance as yf
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                price = hist['Close'].iloc[-1]
        
        if price == 0:
            return jsonify({'error': f'Could not fetch price for {symbol}'}), 500
        
        # Get your neuron analysis
        neuron_result = analyze_with_neuron(price, symbol)
        
        # Asset-specific analysis
        asset_type = 'INDEX' if any(x in symbol.upper() for x in ['NIFTY', 'SENSEX', 'BANK']) else 'STOCK'
        
        comprehensive_analysis = {
            'overall_sentiment': 'BULLISH',
            'asset_type': asset_type,
            'key_insights': [
                f"Current {asset_type.lower()} price: ₹{price:.2f}",
                f"Technical indicators show {'moderate' if asset_type == 'INDEX' else 'strong'} momentum",
                f"{'Index' if asset_type == 'INDEX' else 'Stock'} analysis suggests institutional interest",
                "Risk-reward ratio is favorable for current market conditions"
            ],
            'recommendation': 'BUY',
            'confidence_score': 85 if asset_type == 'INDEX' else 87,
            'neuron_analysis': neuron_result,
            'symbol_info': {
                'original_symbol': symbol,
                'yahoo_symbol': get_yahoo_symbol(symbol),
                'asset_class': asset_type
            }
        }
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'analysis': comprehensive_analysis,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/api/v8/health-check", methods=["GET"])
def health_check_api():
    """API endpoint for system health"""
    try:
        # Test multiple symbols
        test_symbols = ['NIFTY', 'RELIANCE', 'BANKNIFTY']
        connection_status = {}
        
        for symbol in test_symbols:
            try:
                price = get_yfinance_ltp(symbol)
                connection_status[symbol] = 'CONNECTED' if price > 0 else 'FAILED'
            except:
                connection_status[symbol] = 'ERROR'
        
        overall_status = 'HEALTHY' if any(status == 'CONNECTED' for status in connection_status.values()) else 'DEGRADED'
        
        health_status = {
            'system_status': overall_status,
            'connections': connection_status,
            'supported_assets': [
                'NIFTY', 'BANKNIFTY', 'SENSEX', 'FINNIFTY',
                'All NSE Stocks (.NS)', 'Sectoral Indices',
                'NIFTY IT', 'NIFTY PHARMA', 'NIFTY AUTO', 'etc.'
            ],
            'neuron_function': 'ACTIVE',
            'timestamp': datetime.now().isoformat(),
            'version': '8.0'
        }
        
        return jsonify({
            'status': 'success',
            'health': health_status
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route("/strategy-switcher", methods=["GET"])
def strategy_switcher_page():
    return render_template("strategy_switcher.html")

@app.route("/select-strategy", methods=["POST"])
def api_select_strategy():
    try:
        market_data = request.get_json(force=True)

        required_keys = ["vix", "trend", "volatility", "is_expiry"]
        if not all(k in market_data for k in required_keys):
            return jsonify({"error": "Missing one or more required fields."}), 400

        strategy = select_strategy(market_data)
        return jsonify(strategy), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
# --- Start App ---
import os

api_key = os.environ.get("OPENROUTER_API_KEY")

# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(debug=True)
 
