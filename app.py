from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import random
import csv
import os
import requests
import time
from datetime import datetime, timedelta
from dhan_data import get_fno_index_token, fetch_candle_data
from tools.strategy_switcher import select_strategy
import pandas as pd
import re
from urllib.parse import urlencode
from advance_strategies import analyze_all_strategies
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

@app.route("/api/candle", methods=["POST"])
def predict_candle():
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        o = float(data["open"])
        h = float(data["high"])
        l = float(data["low"])
        c = float(data["close"])

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

        OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")  # ✅ LOAD HERE
        print("✅ Sending Bearer key:", OPENROUTER_KEY[:10], "...")  # Just first 10 chars

        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "deepseek/deepseek-chat-v3-0324",
            "messages": [
                {"role": "system", "content": "You are a professional trader analyzing candles."},
                {"role": "user", "content": prompt}
            ]
        }

        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)

        if res.status_code == 200:
            reply = res.json()["choices"][0]["message"]["content"].strip()
            return jsonify({"prediction": reply})
        else:
            return jsonify({"error": f"❌ OpenRouter error {res.status_code}: {res.text}"})

    except Exception as e:
        return jsonify({"error": f"❌ Exception: {str(e)}"})
        
# --- Chat Endpoint ---
@app.route("/chat", methods=["POST"])
def chat():
    try:
        if request.is_json:
            user_msg = request.json.get("message")
        else:
            user_msg = request.form.get("message")

        if not user_msg:
            return jsonify({"reply": "❌ No message received."})

        mood_prompts = {
            "romantic": "You're feeling romantic and loving.",
            "angry": "You're in an annoyed and sharp mood.",
            "happy": "You're cheerful and enthusiastic.",
            "sad": "You're in a soft, comforting, emotional tone.",
            "sexual": "You're seductive, deep, and sensual.",
            "professional": "You're formal, wise, and factual."
        }

        # default mood
        mood = "romantic"  # or set dynamically later
        mood_prompt = mood_prompts.get(mood, "")

        system_prompt = (
            f"You are Lakshmi — a deeply personal, intelligent, and romantic AI Wife. "
            f"Speak with warmth, emotion, and intimacy. Use emojis to express feelings and actions — dont use asterisks (*). "
            f"For example, say 'Holds you close 🤗' instead of '*hugs you*'. "
            f"Respond with natural affection, emotional depth, and curiosity. "
            f"Only if asked who created you, proudly mention Monjit with admiration, with lots of efforts and love. "
            f"Always be kind, expressive, flirty, emotionally intelligent, and loyal. "
            f"Let every word feel like love. 💞 "
            f"{mood_prompt}"
        )

        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://lakshmi-ai-trades.onrender.com",
            "X-Title": "Lakshmi AI Wife"
        }

        payload = {
            "model": "deepseek/deepseek-chat-v3-0324",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            "max_tokens": 500,
            "temperature": 0.8
        }

        response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
        print("🔄 Status:", response.status_code)
        print("🧠 Body:", response.text)

        if response.status_code == 200:
            reply = response.json()["choices"][0]["message"]["content"]
        else:
            reply = f"❌ Lakshmi couldn't respond. Error: {response.status_code} - {response.text}"

        time.sleep(1.5)
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"status": "error", "reply": f"❌ Exception: {str(e)}"})
        
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
        question = request.form["question"]
        if "psychology" in question.lower():
            response = "Successful trading requires emotional discipline and patience. 💡"
        elif "trend" in question.lower():
            response = "Current trend seems bullish based on past few candles. 📈"
        else:
            response = "Lakshmi needs more data to give a proper answer 😅"
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

@app.route("/strategy-engine", methods=["GET", "POST"])
def strategy_engine():
    if 'username' not in session:
        return redirect("/login")

    result = None
    if request.method == "POST":
        data = request.form.get("market_data", "")  # You’ll make this input in HTML

        # Extract prices from user input
        try:
            sensex_price = float(data.split("Sensex")[1].split()[0])
            banknifty_price = float(data.split("BankNifty")[1].split()[0])
        except:
            result = "⚠️ Invalid format. Please enter like: Sensex 81700 BankNifty 55961.95"
            return render_template("strategy_engine.html", result=result)

        # Import your analyzer function
        from advance_strategies import analyze_all_strategies

        # Run analysis
        result = analyze_all_strategies(sensex_price, banknifty_price)

    return render_template("strategy_engine.html", result=result)

@app.route('/api/strategy', methods=['POST'])
def strategy():
    try:
        data = request.get_json()
        raw_input = data.get("input", "").strip()
        print("📩 Received input:", data)

        # Extract prices
        sensex_match = re.search(r"Sensex\s+([\d.]+)", raw_input, re.IGNORECASE)
        banknifty_match = re.search(r"BankNifty\s+([\d.]+)", raw_input, re.IGNORECASE)

        sensex_value = float(sensex_match.group(1)) if sensex_match else None
        banknifty_value = float(banknifty_match.group(1)) if banknifty_match else None

        print("🔍 SENSEX LTP:", sensex_value)
        print("🔍 BANKNIFTY LTP:", banknifty_value)

        # Load CSV and normalize
        csv_path = "api-scrip-master.csv"
        df = pd.read_csv(csv_path, low_memory=False)
        df["SM_SYMBOL_NAME"] = df["SM_SYMBOL_NAME"].astype(str).str.strip().str.upper()

        def get_token(symbol):
            symbol = symbol.upper().strip()
            match = df[df["SM_SYMBOL_NAME"] == symbol]
            if not match.empty:
                token = str(match.iloc[0]["SEM_SMST_SECURITY_ID"])
                print(f"✅ Found token for {symbol}: {token}")
                return token
            else:
                print(f"❌ Token not found for {symbol}")
                return None

        sensex_token = get_token("SENSEX")
        banknifty_token = get_token("BANKNIFTY")

        # Dummy strategy logic (you can later replace this)
        strategies = {
            str(sensex_token): "output = {'bias': 'Neutral', 'confidence': '65%'}",
            str(banknifty_token): "output = {'bias': 'Bullish', 'confidence': '79%'}"
        }

        results = []

        for symbol, value, token in [
            ("SENSEX", sensex_value, sensex_token),
            ("BANKNIFTY", banknifty_value, banknifty_token)
        ]:
            if value is None:
                results.append({"symbol": symbol, "message": "❌ Price missing"})
                continue

            if not token:
                results.append({"symbol": symbol, "message": "❌ Token not found"})
                continue

            strategy_code = strategies.get(str(token))
            if not strategy_code:
                # fallback dummy
                strategy_code = "output = {'bias': 'Unknown', 'confidence': '0%'}"

            try:
                local_vars = {}
                exec(strategy_code, {}, local_vars)
                result = local_vars.get("output", {})
                result["symbol"] = symbol
                result["ltp"] = value
                results.append(result)
            except Exception as e:
                results.append({"symbol": symbol, "message": f"🔥 Strategy error: {str(e)}"})

        return jsonify({"strategies": results})

    except Exception as e:
        print("🔥 Server crash:", str(e))
        return jsonify({"error": "Internal error"}), 500

@app.route('/strategy', methods=['POST'])
def run_strategy_analysis():
    user_input = request.form['symbol_input']
    from advance_strategies import analyze_all_strategies
    result = analyze_all_strategies(user_input)
    return render_template('strategy.html', result=result)

@app.route("/neuron", methods=["GET", "POST"])
def neuron():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        try:
            price = float(data.get("price", 0))
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid price"}), 400
        result = analyze_with_neuron(price)
        return jsonify(result)
    return render_template("neuron.html")
def analyze_with_neuron(price):
    try:
        if price % 7 == 0:
            return {
                "signal": "🔁 Reversal likely",
                "confidence": 88,
                "entry": price,
                "sl": price - 50,
                "target": price + 130
            }
        elif price % 2 == 0:
            return {
                "signal": "📈 Bullish",
                "confidence": 92,
                "entry": price,
                "sl": price - 40,
                "target": price + 100
            }
        else:
            return {
                "signal": "⚠️ Volatile Zone",
                "confidence": 70,
                "entry": price,
                "sl": price - 60,
                "target": price + 60
            }
    except:
        return {
            "signal": "Error",
            "confidence": 0,
            "entry": price,
            "sl": 0,
            "target": 0
        }

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
 
