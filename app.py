# app.py
"""
Prototype Forex Signal App (single-file)
- FastAPI web app with username/password login (username: "kc trades", password hashed)
- Persistent session token in-memory while the process runs (no re-login until restart)
- Background monitoring task (Mon-Fri) that polls price data (simulated by default)
- Indicator calculations (SMA, RSI, MACD, ATR, Bollinger)
- Simple ML model placeholder (RandomForest) producing probability
- Smart stop-loss calculation (ATR + recent swing)
- Telegram notifications optional (requires TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID env vars)
- Plotly candlestick charts per pair (protected behind login)
Run: uvicorn app:app --reload
"""

import os
import time
import math
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List

import uvicorn
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
import bcrypt
import jwt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import requests
import plotly.graph_objs as go

try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except Exception:
    TELEGRAM_AVAILABLE = False

# ---------------------------
# CONFIG - edit these values
# ---------------------------
USERNAME = "kc trades"
PLAIN_PASSWORD = "376148"   # Provided by you; hashed below
JWT_SECRET = "replace_with_a_random_secret_in_production"  # change for prod
JWT_ALGORITHM = "HS256"

# Telegram settings - set these as environment variables in Replit or your system if you want
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")  # leave empty for testing without Telegram
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")      # leave empty for testing without Telegram

# Data provider (optional). If empty, script uses simulated prices for demo.
DATA_API_KEY = os.getenv("DATA_API_KEY", "")

# Pairs to monitor (majors + some minors) - you can expand
FX_PAIRS = [
    "EUR/USD","GBP/USD","USD/JPY","USD/CHF","AUD/USD","USD/CAD","NZD/USD",
    "EUR/GBP","EUR/JPY","GBP/JPY","AUD/JPY","EUR/AUD","GBP/CHF"
]

POLL_INTERVAL_SECONDS = 30  # fetch frequency (seconds). Increase if using free API.

# ---------------------------
# INTERNAL STORAGE
# ---------------------------
# Hash the password with bcrypt
hashed_password = bcrypt.hashpw(PLAIN_PASSWORD.encode(), bcrypt.gensalt())

# in-memory storage for OHLC per pair
latest_bars: Dict[str, pd.DataFrame] = {pair: pd.DataFrame(columns=["datetime","open","high","low","close","volume"]) for pair in FX_PAIRS}

# avoid spamming identical signals
last_signal_cache: Dict[str, Dict[str, Any]] = {}

# simple in-memory active sessions
active_sessions: Dict[str, float] = {}  # token -> last_active_time

# initialize telegram bot object if token provided and library available
telegram_bot = None
if TELEGRAM_BOT_TOKEN and TELEGRAM_AVAILABLE:
    try:
        telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)
    except Exception:
        telegram_bot = None

app = FastAPI()

# ---------------------------
# AUTH helpers
# ---------------------------
def verify_credentials(username: str, password: str) -> bool:
    if username != USERNAME:
        return False
    try:
        return bcrypt.checkpw(password.encode(), hashed_password)
    except Exception:
        return False

def create_session_token(username: str) -> str:
    payload = {"sub": username, "iat": int(time.time())}
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    active_sessions[token] = time.time()
    return token

def verify_token(token: str) -> bool:
    if not token:
        return False
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("sub") != USERNAME:
            return False
        if token not in active_sessions:
            return False
        active_sessions[token] = time.time()
        return True
    except Exception:
        return False

# ---------------------------
# Indicators
# ---------------------------
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def macd(df: pd.DataFrame, fast=12, slow=26, signal=9):
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, macd_signal

def bollinger_bands(series: pd.Series, window: int = 20, n_std: float = 2):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + (std * n_std)
    lower = ma - (std * n_std)
    return upper, lower

# ---------------------------
# ML placeholder
# ---------------------------
MODEL_PATH = "rf_model.joblib"

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f['close'] = df['close']
    f['ret1'] = df['close'].pct_change(1).fillna(0)
    f['ret3'] = df['close'].pct_change(3).fillna(0)
    f['sma8'] = sma(df['close'], 8)
    f['sma21'] = sma(df['close'], 21)
    f['rsi'] = rsi(df['close']).fillna(50)
    f['atr'] = atr(df).fillna(method='bfill').fillna(0)
    f = f.fillna(0)
    return f

def train_demo_model(pair: str):
    if os.path.exists(f"{pair.replace('/','')}_history.csv"):
        df = pd.read_csv(f"{pair.replace('/','')}_history.csv", parse_dates=['datetime']).set_index('datetime')
    else:
        idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=500, freq='T')
        price = 1.0 + np.cumsum(np.random.normal(0, 0.0002, size=len(idx)))
        df = pd.DataFrame({"close": price, "open": price, "high": price, "low": price, "volume": 1}, index=idx)
    X = build_features(df)
    nxt = df['close'].shift(-1)
    y = (nxt > df['close']).astype(int).fillna(0)
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        m = train_demo_model(FX_PAIRS[0])
        return m

ml_model = load_model()

# ---------------------------
# Smart SL/TP
# ---------------------------
def calculate_smart_sl_tp(df: pd.DataFrame, entry_price: float, side: str, atr_mult: float = 1.5):
    last_atr = atr(df).iloc[-1] if len(df) >= 14 else df['close'].std() * 0.5
    sl_distance = max(0.00001, last_atr * atr_mult)
    recent_low = df['low'].rolling(10).min().iloc[-1]
    recent_high = df['high'].rolling(10).max().iloc[-1]
    if side == "BUY":
        sl = min(entry_price - sl_distance, recent_low - sl_distance * 0.5)
        tp = entry_price + sl_distance * 2.0
    else:
        sl = max(entry_price + sl_distance, recent_high + sl_distance * 0.5)
        tp = entry_price - sl_distance * 2.0
    return float(round(sl, 6)), float(round(tp, 6)), float(round(sl_distance, 6))

# ---------------------------
# Signal logic
# ---------------------------
def generate_signal_for_pair(pair: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if len(df) < 50:
        return None
    close = df['close']
    sma_fast = sma(close, 8).iloc[-1]
    sma_slow = sma(close, 21).iloc[-1]
    rsi_val = rsi(close).iloc[-1]
    macd_line, macd_sig = macd(df)
    macd_val = macd_line.iloc[-1] - macd_sig.iloc[-1]
    atr_val = atr(df).iloc[-1]
    X = build_features(df).iloc[[-1]]
    prob_up = float(ml_model.predict_proba(X)[0][1]) if hasattr(ml_model, "predict_proba") else 0.5

    indicators_used = []
    reasons = []

    rule_long = sma_fast > sma_slow and rsi_val > 45 and macd_val > 0
    rule_short = sma_fast < sma_slow and rsi_val < 55 and macd_val < 0

    wick = (df['high'].iloc[-1] - df['close'].iloc[-1]) if df['close'].iloc[-1] < df['open'].iloc[-1] else (df['close'].iloc[-1] - df['low'].iloc[-1])
    wick_ratio = 0.0
    hl = (df['high'].iloc[-1] - df['low'].iloc[-1])
    if hl > 0:
        wick_ratio = wick / hl
    low_momentum = abs(macd_val) < (atr_val * 0.1)

    if rule_long:
        indicators_used += ["SMA8>21","RSI","MACD"]
        reasons.append("Trend+momentum")
    if rule_short:
        indicators_used += ["SMA8<21","RSI","MACD"]
        reasons.append("Trend+momentum")

    final_side = None
    final_prob = prob_up
    prob_threshold = 0.6
    if rule_long and prob_up >= prob_threshold and not low_momentum and wick_ratio < 0.7:
        final_side = "BUY"
        reasons.append(f"ML prob {prob_up:.2f} >= {prob_threshold}")
    elif rule_short and (1 - prob_up) >= prob_threshold and not low_momentum and wick_ratio < 0.7:
        final_side = "SELL"
        final_prob = 1 - prob_up
        reasons.append(f"ML prob {1-prob_up:.2f} >= {prob_threshold}")
    else:
        return None

    entry = float(df['close'].iloc[-1])
    sl, tp, sl_dist = calculate_smart_sl_tp(df, entry, final_side, atr_mult=1.5)

    signal = {
        "pair": pair,
        "side": final_side,
        "entry": round(entry, 6),
        "tp": tp,
        "sl": sl,
        "probability": round(float(final_prob), 4),
        "indicators": indicators_used,
        "reasons": reasons,
        "timestamp": pd.Timestamp.utcnow().isoformat()
    }
    return signal

# ---------------------------
# Telegram helper
# ---------------------------
def send_telegram_message(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID or not TELEGRAM_AVAILABLE or not telegram_bot:
        print("Telegram not configured — would send:", text)
        return
    try:
        telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode="Markdown")
    except Exception as e:
        print("Telegram send error:", e)

def format_signal_message(sig: Dict[str, Any]) -> str:
    lines = [
        f"*SIGNAL* — `{sig['pair']}`",
        f"*Action:* {sig['side']}",
        f"*Entry:* {sig['entry']}",
        f"*TP:* {sig['tp']}",
        f"*SL:* {sig['sl']}",
        f"*Probability:* {sig['probability']:.2f}",
        f"*Indicators:* {', '.join(sig['indicators'])}",
        f"*Reasons:* {', '.join(sig['reasons'])}",
        f"*Time (UTC):* {sig['timestamp']}"
    ]
    return "\n".join(lines)

# ---------------------------
# Data fetch (simulated by default)
# ---------------------------
def fetch_latest_candle(pair: str) -> Optional[Dict]:
    if DATA_API_KEY:
        symbol = pair.replace('/', '')
        try:
            url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1min&outputsize=1&apikey={DATA_API_KEY}"
            r = requests.get(url, timeout=10)
            j = r.json()
            if "values" in j:
                v = j["values"][0]
                return {
                    "datetime": pd.to_datetime(v["datetime"]),
                    "open": float(v["open"]),
                    "high": float(v["high"]),
                    "low": float(v["low"]),
                    "close": float(v["close"]),
                    "volume": float(v.get("volume", 0))
                }
        except Exception as e:
            print("Data fetch error:", e)
            return None
    df = latest_bars[pair]
    if df.empty:
        base = 1.1000 + np.random.normal(0, 0.001)
    else:
        base = float(df['close'].iloc[-1])
    new_price = base + np.random.normal(0, 0.0002)
    return {"datetime": pd.Timestamp.utcnow(), "open": base, "high": max(base, new_price)+0.00005, "low": min(base, new_price)-0.00005, "close": new_price, "volume": 1.0}

# ---------------------------
# Background worker
# ---------------------------
async def background_monitor():
    print("Background monitor started.")
    while True:
        now = datetime.utcnow()
        weekday = now.weekday()
        if weekday >= 5:
            await asyncio.sleep(60*60)
            continue

        for pair in FX_PAIRS:
            candle = fetch_latest_candle(pair)
            if candle is None:
                continue
            df = latest_bars[pair]
            new_row = pd.DataFrame([candle])
            latest_bars[pair] = pd.concat([df, new_row], ignore_index=True)
            if len(latest_bars[pair]) > 5000:
                latest_bars[pair] = latest_bars[pair].iloc[-5000:].reset_index(drop=True)

            sig = generate_signal_for_pair(pair, latest_bars[pair])
            if sig:
                last = last_signal_cache.get(pair)
                send_it = False
                if not last:
                    send_it = True
                else:
                    try:
                        prev_t = pd.Timestamp(last.get("timestamp"))
                        cur_t = pd.Timestamp(sig["timestamp"])
                        if sig.get("side") != last.get("side") or (cur_t - prev_t).total_seconds() > 60*10:
                            send_it = True
                    except Exception:
                        send_it = True
                if send_it:
                    last_signal_cache[pair] = sig
                    msg = format_signal_message(sig)
                    # send (or print) message
                    loop = asyncio.get_event_loop()
                    loop.run_in_executor(None, send_telegram_message, msg)
                    print("Signal generated:", sig)
        await asyncio.sleep(POLL_INTERVAL_SECONDS)

@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_event_loop()
    loop.create_task(background_monitor())
    print("App startup complete. Background monitor scheduled.")

# ---------------------------
# Web UI (login & chart)
# ---------------------------
LOGIN_PAGE_HTML = """
<!doctype html>
<html>
<head><title>Login - Forex Signal App</title></head>
<body>
  <h2>Forex Signals — Login</h2>
  <form action="/login" method="post">
    <label>Username:</label><br/>
    <input type="text" name="username" value="kc trades" /><br/>
    <label>Password:</label><br/>
    <input type="password" name="password" /><br/><br/>
    <input type="submit" value="Login" />
  </form>
  <p>Close the app (stop the server) to require login again.</p>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(LOGIN_PAGE_HTML)

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    if verify_credentials(username, password):
        token = create_session_token(username)
        return RedirectResponse(url=f"/dashboard?token={token}", status_code=303)
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

def require_token(token: Optional[str]) -> None:
    if not token or not verify_token(token):
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(token: Optional[str] = None):
    require_token(token)
    pairs_html = "".join([f'<li><a href="/pair/{p.replace("/","")}?token={token}">{p}</a></li>' for p in FX_PAIRS])
    html = f"""
    <html><body>
      <h2>Forex Signals Dashboard</h2>
      <p>Logged in as: {USERNAME}</p>
      <ul>{pairs_html}</ul>
      <p>Telegram configured: {bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)}</p>
      <p><a href="/api/last_signals?token={token}">View last signals (JSON)</a></p>
      <p>Note: Close the app (stop the server) to require login again.</p>
    </body></html>
    """
    return HTMLResponse(html)

@app.get("/pair/{pair}", response_class=HTMLResponse)
async def pair_chart(pair: str, token: Optional[str] = None):
    require_token(token)
    pair_slash = pair[:3] + "/" + pair[3:]
    df = latest_bars.get(pair_slash)
    if df is None or df.empty:
        return HTMLResponse(f"<html><body><p>No data for {pair_slash} yet (wait about a minute).</p><p><a href='/dashboard?token={token}'>Back</a></p></body></html>")
    fig = go.Figure(data=[go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close']
    )])
    if len(df) >= 21:
        fig.add_trace(go.Scatter(x=df['datetime'], y=sma(df['close'],8), name='SMA8'))
        fig.add_trace(go.Scatter(x=df['datetime'], y=sma(df['close'],21), name='SMA21'))
    fig.update_layout(title=f"Live chart - {pair_slash}", xaxis_rangeslider_visible=False, height=700)
    plot_html = fig.to_html(full_html=False)
    page = f"<html><body><a href='/dashboard?token={token}'>Back</a><br/>{plot_html}</body></html>"
    return HTMLResponse(page)

@app.get("/api/last_signals")
async def api_last_signals(token: Optional[str] = None):
    require_token(token)
    return last_signal_cache

@app.get("/logout")
async def logout(token: Optional[str] = None):
    if token in active_sessions:
        del active_sessions[token]
    return RedirectResponse(url="/", status_code=303)

# ---------------------------
# Run uvicorn directly
# ---------------------------
if __name__ == "__main__":
    print("Run with: uvicorn app:app --reload")
if __name__ == "__main__":
    print("Run with: uvicorn app:app --reload")
