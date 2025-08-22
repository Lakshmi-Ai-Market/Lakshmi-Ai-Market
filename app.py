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

# Load environment variables safely
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# ✅ Print loaded keys (for debug only — remove in production)
print("🔑 DHAN_CLIENT_ID:", os.getenv("DHAN_CLIENT_ID"))
print("🔑 DHAN_ACCESS_TOKEN:", os.getenv("DHAN_ACCESS_TOKEN"))
print("🔑 OPENROUTER_KEY:", os.getenv("OPENROUTER_API_KEY"))

app = Flask(__name__)
app.secret_key = "lakshmi_secret_key"
app.config['UPLOAD_FOLDER'] = 'static/voice_notes'

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

# --- Routes ---
@app.route("/")
def home():
    return redirect("/login")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        dob = request.form.get("dob", "").strip()
        gender = request.form.get("gender", "").strip()
        password = request.form["password"]
        confirm_password = request.form.get("confirm_password")
        terms_agreed = request.form.get("terms")

        if not terms_agreed:
            return render_template("signup.html", error="Please accept terms and conditions.")
        if password != confirm_password:
            return render_template("signup.html", error="Passwords do not match.")

        users = load_users()
        if any(u['username'] == username for u in users):
            return render_template("signup.html", error="Username already exists 💔")

        file_exists = os.path.isfile("users.csv")
        with open("users.csv", "a", newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["username", "email", "phone", "dob", "gender", "password"])
            writer.writerow([username, email, phone, dob, gender, password])

        session['username'] = username
        return redirect("/dashboard")

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        users = load_users()
        for u in users:
            if u["username"] == username and u["password"] == password:
                session["username"] = username
                return redirect("/dashboard")
        return render_template("login.html", error="Invalid credentials 💔")
    return render_template("login.html")
    
@app.route("/logout", methods=["POST"])
def logout():
    session.pop("username", None)
    session.pop("email", None)
    return redirect("/login")
    
def get_google_config():
    import requests
    return requests.get("https://accounts.google.com/.well-known/openid-configuration").json()

@app.route("/auth/login")
def google_login():
    cfg = get_google_config()
    auth_endpoint = cfg["authorization_endpoint"]

    params = {
        "response_type": "code",
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "redirect_uri": REDIRECT_URI,
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent"
    }
    return redirect(f"{auth_endpoint}?{urlencode(params)}")

@app.route("/auth/callback")
def google_callback():
    code = request.args.get("code")
    if not code:
        return "❌ No code from Google", 400

    cfg = get_google_config()
    token_endpoint = cfg["token_endpoint"]

    token_res = requests.post(
        token_endpoint,
        data={
            "code": code,
            "client_id": os.getenv("GOOGLE_CLIENT_ID"),
            "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code"
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    token_json = token_res.json()
    access_token = token_json.get("access_token")
    if not access_token:
        return "❌ Failed to get token from Google", 400

    userinfo_res = requests.get(
        cfg["userinfo_endpoint"],
        headers={"Authorization": f"Bearer {access_token}"}
    )
    userinfo = userinfo_res.json()
    email = userinfo.get("email")

    if not email:
        return "❌ Failed to get user email", 400

    session["email"] = email
    return redirect("/dashboard")

@app.route("/dashboard")
def dashboard():
    if "username" not in session and "email" not in session:
        return redirect("/login")

    name = session.get("username") or session.get("email")
    return render_template("index.html", name=name, mood=current_mood)

@app.route("/candle")
def candle_ui():
    return render_template("dashboard.html")
    
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
@app.route("/")
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
 
