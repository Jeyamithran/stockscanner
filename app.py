import os
import json
import re
import logging
import ast
import sqlite3
import time
import math
from collections import deque
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from json import JSONDecodeError
from datetime import datetime, date, timedelta, timezone
from typing import List, Dict, Any, Tuple, Iterable, Optional, Set

import requests
from flask import Flask, jsonify, render_template, request
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import HTTPException, InternalServerError
from dotenv import load_dotenv
from openai import OpenAI
import xml.etree.ElementTree as ET
try:
    import yfinance as yf
except ImportError:
    yf = None
try:
    import pandas as pd
except ImportError:
    pd = None
from prompts import (
    HIGH_GROWTH_ANALYSIS_PROMPT,
    HISTORY_DEEP_DIVE_PROMPT,
    NEWS_SYSTEM_PROMPT,
    SUPER_TREND_PROMPT,
    ZIGZAG_PROMPT,
    TRADINGVIEW_SIGNAL_PROMPT,
    select_scanner_prompt
)

# -------------------------------------------------------------
#  Setup
# -------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
load_dotenv()

app = Flask(__name__)
application = app
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change_me_locally_only")

# --- AUTHENTICATION ---
auth = HTTPBasicAuth()

USER_DATA = {
    "username": os.environ.get("SCANNER_USERNAME", "admin"),
    "password_hash": generate_password_hash(
        os.environ.get("SCANNER_PASSWORD", "default_password"),
        method="pbkdf2:sha256:260000"
    )
}


@auth.verify_password
def verify_password(username, password):
    if username == USER_DATA["username"]:
        if check_password_hash(USER_DATA["password_hash"], password):
            return username
    return None


# --- Perplexity Client ---
PERPLEXITY_API_KEY = os.environ.get("PPLX_API_KEY")
if not PERPLEXITY_API_KEY:
    logging.error("PPLX_API_KEY not found. Perplexity features will be disabled.")
    pplx_client = None
else:
    pplx_client = OpenAI(
        api_key=PERPLEXITY_API_KEY,
        base_url="https://api.perplexity.ai"
    )
DEFAULT_PPLX_MODEL = os.environ.get("PPLX_DEFAULT_MODEL", "sonar-reasoning-pro")
PPLX_MODEL_ALIASES = {
    "reasoning": "sonar-reasoning-pro",
    "pro": "sonar-pro",
    "classic": "sonar-pro"
}

# --- External API keys ---
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
BENZINGA_API_KEY = os.environ.get("BENZINGA_API_KEY")
FMP_API_KEY = os.environ.get("FMP_API_KEY") or os.environ.get("FMP_API_TOKEN")
PRICE_ENRICH_TIMEOUT = float(os.environ.get("SCANNER_PRICE_ENRICH_TIMEOUT", "18"))
FINNHUB_CACHE_TTL = float(os.environ.get("FINNHUB_CACHE_TTL_SECONDS", "45"))
FINNHUB_RATE_LIMIT_SECONDS = float(os.environ.get("FINNHUB_RATE_LIMIT_SECONDS", "0.25"))
SCANNER_ALLOW_FALLBACK = os.environ.get("SCANNER_ALLOW_FALLBACK", "false").strip().lower() == "true"

PROFILE_LABELS = {
    "hedge_fund": "Hedge Fund PM",
    "pro_trader": "Pro Momentum",
    "catalyst": "News / SEC",
    "bio_analyst": "Biotech Catalyst",
    "immediate_breakout": "Breakout Radar",
    "high_growth": "High Growth"
}


def format_profile_label(profile_key: str) -> str:
    if not profile_key:
        return ""
    return PROFILE_LABELS.get(profile_key, profile_key.replace("_", " ").title())


def resolve_pplx_model(requested: str) -> str:
    if not requested:
        return DEFAULT_PPLX_MODEL
    key = requested.strip().lower()
    if key in PPLX_MODEL_ALIASES:
        return PPLX_MODEL_ALIASES[key]
    # Allow passing exact model id
    if key in ("sonar-pro", "sonar-reasoning-pro"):
        return key
    return DEFAULT_PPLX_MODEL


def normalize_model_variant(model_name: str) -> str:
    """Map raw model ids to UI-friendly variants (reasoning/pro/etc)."""
    if not model_name:
        return ""
    name = model_name.strip().lower()
    if "reasoning" in name:
        return "reasoning"
    if name in {"sonar-pro", "pro", "classic"}:
        return "pro"
    return name

os.makedirs(app.instance_path, exist_ok=True)
ALERT_HISTORY_DB = os.environ.get(
    "ALERT_HISTORY_DB_PATH",
    os.path.join(app.instance_path, "alert_history.db")
)


def init_alert_history_db() -> None:
    """Ensure the SQLite database and table exist for persisted alert history."""
    conn = sqlite3.connect(ALERT_HISTORY_DB)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alert_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_iso TEXT NOT NULL,
                date TEXT,
                time TEXT,
                profile TEXT,
                tier TEXT,
                ticker TEXT,
                entry REAL,
                target REAL,
                ai_target_price REAL,
                stop REAL,
                scan_price REAL,
                potential_gain_pct REAL,
                ai_potential_gain_pct REAL,
                primary_catalyst TEXT,
                detailed_analysis TEXT,
                model_used TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_alert_history_ticker ON alert_history(ticker)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_alert_history_timestamp ON alert_history(timestamp_iso)"
        )
        conn.commit()
    finally:
        conn.close()


def ensure_alert_history_columns() -> None:
    """Add newly required columns to alert_history if they are missing."""
    conn = sqlite3.connect(ALERT_HISTORY_DB)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute("PRAGMA table_info(alert_history)")
        existing = {row["name"] for row in cur.fetchall()}
        alters = []
        if "ai_target_price" not in existing:
            alters.append("ALTER TABLE alert_history ADD COLUMN ai_target_price REAL")
        if "ai_potential_gain_pct" not in existing:
            alters.append("ALTER TABLE alert_history ADD COLUMN ai_potential_gain_pct REAL")
        if "model_used" not in existing:
            alters.append("ALTER TABLE alert_history ADD COLUMN model_used TEXT")

        for stmt in alters:
            conn.execute(stmt)
        if alters:
            conn.commit()
    finally:
        conn.close()


def insert_alert_rows(rows: Iterable[Tuple[Any, ...]]) -> None:
    rows = list(rows)
    if not rows:
        return
    conn = sqlite3.connect(ALERT_HISTORY_DB)
    try:
        conn.executemany(
            """
            INSERT INTO alert_history (
                timestamp_iso,
                date,
                time,
                profile,
                tier,
                ticker,
                entry,
                target,
                ai_target_price,
                stop,
                scan_price,
                potential_gain_pct,
                ai_potential_gain_pct,
                primary_catalyst,
                detailed_analysis,
                model_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows
        )
        conn.commit()
    finally:
        conn.close()


ALERT_HISTORY_CACHE: List[Dict[str, Any]] = []
FINNHUB_QUOTE_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
LAST_FINNHUB_REQUEST = 0.0
ECON_CAL_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
ECON_CAL_CACHE_TTL = float(os.environ.get("ECON_CAL_CACHE_TTL_SECONDS", "300"))
FOREXFACTORY_SOURCES = [
    ("thisweek", "https://nfs.faireconomy.media/ff_calendar_thisweek.json"),
    ("nextweek", "https://nfs.faireconomy.media/ff_calendar_nextweek.json"),
]
FOREXFACTORY_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
FOREXFACTORY_CACHE_TTL = float(os.environ.get("FOREX_FACTORY_CACHE_TTL_SECONDS", "600"))
SUPER_TREND_TICKERS = ["SPY", "QQQ", "NVDA", "TSLA", "MSFT"]
MAX_CUSTOM_TICKERS = int(os.environ.get("RADAR_MAX_TICKERS", "10"))
TRADINGVIEW_WEBHOOK_TOKEN = os.environ.get("TRADINGVIEW_WEBHOOK_TOKEN")
TRADINGVIEW_MAX_BARS = max(60, int(os.environ.get("TRADINGVIEW_MAX_BARS", "320")))
TRADINGVIEW_MIN_BARS_FOR_SIGNAL = max(5, int(os.environ.get("TRADINGVIEW_MIN_BARS", "40")))
TRADINGVIEW_PROMPT_BAR_LIMIT = max(
    20,
    min(int(os.environ.get("TRADINGVIEW_PROMPT_BAR_LIMIT", "120")), TRADINGVIEW_MAX_BARS)
)
TRADINGVIEW_SIGNAL_MODEL = resolve_pplx_model(os.environ.get("TRADINGVIEW_SIGNAL_MODEL", DEFAULT_PPLX_MODEL))
TRADINGVIEW_SIGNAL_MAX_WORKERS = max(1, int(os.environ.get("TRADINGVIEW_SIGNAL_MAX_WORKERS", "2")))
TRADINGVIEW_SIGNAL_COOLDOWN_SECONDS = float(os.environ.get("TRADINGVIEW_SIGNAL_COOLDOWN_SECONDS", "60"))


def _tv_coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _tv_safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _tv_epoch_seconds(raw: Any) -> Optional[int]:
    try:
        ts = int(float(raw))
    except (TypeError, ValueError):
        return None
    if ts > 1_000_000_000_000:  # TradingView sends ms timestamps
        ts = int(ts / 1000)
    return ts


def _tv_iso(ts: int) -> str:
    return datetime.utcfromtimestamp(ts).replace(microsecond=0).isoformat() + "Z"


class TradingViewRelay:
    """In-memory store for TradingView OHLCV relays and latest AI signals."""

    def __init__(self, max_bars: int = 320):
        self.max_bars = max(40, int(max_bars))
        self._bars: Dict[str, deque] = {}
        self._signals: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def stream_key(self, symbol: str, timeframe: str) -> str:
        symbol_clean = (symbol or "").strip().upper()
        timeframe_clean = str(timeframe or "").strip()
        return f"{symbol_clean}::{timeframe_clean}"

    def add_bar(self, symbol: str, timeframe: str, bar: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        key = self.stream_key(symbol, timeframe)
        if key.endswith("::"):
            raise ValueError("Symbol and timeframe are required for TradingView ingestion.")
        epoch = _tv_epoch_seconds(bar.get("time"))
        if epoch is None:
            raise ValueError("Invalid or missing bar timestamp.")
        symbol_val, timeframe_val = key.split("::", 1)
        normalized = {
            "symbol": symbol_val,
            "timeframe": timeframe_val,
            "time": epoch,
            "time_iso": _tv_iso(epoch),
            "open": _tv_safe_float(bar.get("open")),
            "high": _tv_safe_float(bar.get("high")),
            "low": _tv_safe_float(bar.get("low")),
            "close": _tv_safe_float(bar.get("close")),
            "volume": _tv_safe_float(bar.get("volume")) or 0.0,
        }
        if normalized["close"] is None:
            raise ValueError("Close price is required for TradingView ingestion.")
        with self._lock:
            dq = self._bars.setdefault(key, deque(maxlen=self.max_bars))
            dq.append(normalized)
            count = len(dq)
        return count, normalized

    def get_bars(self, symbol: str, timeframe: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        key = self.stream_key(symbol, timeframe)
        with self._lock:
            dq = self._bars.get(key)
            if not dq:
                return []
            data = list(dq)
        if limit:
            return data[-int(limit):]
        return data

    def set_signal(self, symbol: str, timeframe: str, signal: Dict[str, Any]) -> Dict[str, Any]:
        key = self.stream_key(symbol, timeframe)
        payload = dict(signal or {})
        stream_symbol, stream_tf = key.split("::", 1)
        payload.setdefault("symbol", stream_symbol)
        payload.setdefault("timeframe", stream_tf)
        with self._lock:
            self._signals[key] = payload
        return payload

    def get_signal(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        key = self.stream_key(symbol, timeframe)
        with self._lock:
            signal = self._signals.get(key)
            return dict(signal) if signal else None

    def describe_streams(self) -> List[Dict[str, Any]]:
        snapshots: List[Dict[str, Any]] = []
        with self._lock:
            for key, dq in self._bars.items():
                symbol, timeframe = key.split("::", 1)
                last_bar = dq[-1] if dq else None
                signal = self._signals.get(key)
                snapshots.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "bar_count": len(dq),
                    "last_bar_time": last_bar.get("time_iso") if last_bar else None,
                    "last_price": last_bar.get("close") if last_bar else None,
                    "latest_signal": signal.get("signal") if signal else None,
                    "latest_signal_time": signal.get("generated_at") if signal else None,
                })
        return snapshots


TRADINGVIEW_SIGNAL_LOCK = Lock()
TRADINGVIEW_SIGNAL_INFLIGHT: Set[str] = set()
TRADINGVIEW_SIGNAL_LAST_RUN: Dict[str, float] = {}
tradingview_relay = TradingViewRelay(max_bars=TRADINGVIEW_MAX_BARS)
tradingview_executor = ThreadPoolExecutor(max_workers=TRADINGVIEW_SIGNAL_MAX_WORKERS)


def summarize_tradingview_bars(bars: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not bars:
        return {}
    closes = [float(bar.get("close") or 0.0) for bar in bars]
    highs = [float(bar.get("high") or bar.get("close") or 0.0) for bar in bars]
    lows = [float(bar.get("low") or bar.get("close") or 0.0) for bar in bars]
    volumes = [float(bar.get("volume") or 0.0) for bar in bars]
    lookback = len(bars)
    last_close = closes[-1]
    prev_close = closes[-2] if lookback > 1 else last_close
    first_close = closes[0] or last_close
    change_from_prev = ((last_close - prev_close) / prev_close * 100.0) if prev_close else 0.0
    change_from_start = ((last_close - first_close) / first_close * 100.0) if first_close else 0.0
    intraday_range = (max(highs) - min(lows)) if highs and lows else 0.0
    range_pct = (intraday_range / last_close * 100.0) if last_close else 0.0
    volume_window = min(20, lookback)
    recent_volumes = volumes[-volume_window:]
    avg_volume = sum(recent_volumes) / volume_window if volume_window else 0.0
    volume_vs_avg = (volumes[-1] / avg_volume) if avg_volume else None
    window_seconds = bars[-1].get("time", 0) - bars[0].get("time", 0)
    window_minutes = window_seconds / 60 if window_seconds else 0
    return {
        "bars": lookback,
        "window_minutes": window_minutes,
        "last_close": last_close,
        "prev_close": prev_close,
        "change_pct": round(change_from_prev, 2),
        "change_pct_total": round(change_from_start, 2),
        "range_pct": round(range_pct, 2),
        "avg_volume": avg_volume,
        "last_volume": volumes[-1],
        "volume_vs_avg": round(volume_vs_avg, 2) if volume_vs_avg is not None else None,
    }


def build_tradingview_prompt_payload(symbol: str, timeframe: str, bars: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = summarize_tradingview_bars(bars)
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "bar_count": len(bars),
        "summary": summary,
        "bars": bars,
    }


def maybe_queue_tradingview_signal(symbol: str, timeframe: str, force: bool = False) -> Tuple[bool, Optional[str]]:
    if not pplx_client:
        return False, "Perplexity client is not configured."
    bars = tradingview_relay.get_bars(symbol, timeframe)
    if len(bars) < TRADINGVIEW_MIN_BARS_FOR_SIGNAL:
        return False, f"Need {TRADINGVIEW_MIN_BARS_FOR_SIGNAL} bars; only {len(bars)} available."

    snippet = bars[-TRADINGVIEW_PROMPT_BAR_LIMIT:]
    key = tradingview_relay.stream_key(symbol, timeframe)
    now = time.monotonic()
    with TRADINGVIEW_SIGNAL_LOCK:
        if key in TRADINGVIEW_SIGNAL_INFLIGHT:
            return False, "Signal generation already running for this stream."
        last_run = TRADINGVIEW_SIGNAL_LAST_RUN.get(key, 0.0)
        if not force and TRADINGVIEW_SIGNAL_COOLDOWN_SECONDS > 0:
            wait = TRADINGVIEW_SIGNAL_COOLDOWN_SECONDS - (now - last_run)
            if wait > 0:
                return False, f"Cooldown active; retry in {math.ceil(wait)}s."
        TRADINGVIEW_SIGNAL_INFLIGHT.add(key)

    tradingview_executor.submit(_run_tradingview_signal_job, symbol, timeframe, snippet, key)
    return True, None


def _run_tradingview_signal_job(symbol: str, timeframe: str, bars: List[Dict[str, Any]], key: str) -> None:
    try:
        generate_tradingview_signal(symbol, timeframe, bars)
    finally:
        with TRADINGVIEW_SIGNAL_LOCK:
            TRADINGVIEW_SIGNAL_INFLIGHT.discard(key)
            TRADINGVIEW_SIGNAL_LAST_RUN[key] = time.monotonic()


def generate_tradingview_signal(symbol: str, timeframe: str, bars: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not pplx_client:
        logging.warning("Skipped TradingView signal for %s %s (Perplexity disabled).", symbol, timeframe)
        return None
    payload = build_tradingview_prompt_payload(symbol, timeframe, bars)
    user_prompt = (
        "OHLCV bars forwarded from TradingView (oldest to newest). Prices already match what the trader sees; do "
        "NOT invent new bars.\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n\n"
        "Return ONLY the JSON object described in the system prompt."
    )
    try:
        response = pplx_client.chat.completions.create(
            model=TRADINGVIEW_SIGNAL_MODEL,
            messages=[
                {"role": "system", "content": TRADINGVIEW_SIGNAL_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            extra_body={"search_recency_filter": "day"}
        )
        raw = response.choices[0].message.content or ""
        json_payload = extract_json_from_text(raw)
        data = json.loads(json_payload)
    except Exception as exc:
        logging.error("TradingView signal generation failed for %s %s: %s", symbol, timeframe, exc)
        return None

    data.setdefault("symbol", symbol.upper())
    data.setdefault("timeframe", timeframe)
    data.setdefault("generated_at", datetime.utcnow().replace(microsecond=0).isoformat() + "Z")
    data.setdefault("bars_used", len(bars))
    tradingview_relay.set_signal(symbol, timeframe, data)
    logging.info("Stored TradingView AI signal for %s %s: %s", symbol, timeframe, data.get("signal"))
    return data



def load_alert_history_cache() -> None:
    """Load all existing history rows from SQLite into the in-process cache for fast access."""
    conn = sqlite3.connect(ALERT_HISTORY_DB)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            """
            SELECT
                timestamp_iso,
                date,
                time,
                profile,
                tier,
                ticker,
                entry,
                target,
                ai_target_price,
                stop,
                scan_price,
                potential_gain_pct,
                ai_potential_gain_pct,
                primary_catalyst,
                detailed_analysis,
                model_used
            FROM alert_history
            ORDER BY timestamp_iso ASC
            """
        )
        ALERT_HISTORY_CACHE.extend(dict(r) for r in cur.fetchall())
    finally:
        conn.close()


def fetch_alert_rows_by_ticker(ticker: str, limit: int = 30) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(ALERT_HISTORY_DB)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            """
            SELECT
                timestamp_iso,
                date,
                time,
                profile,
                tier,
                ticker,
                entry,
                target,
                ai_target_price,
                stop,
                scan_price,
                potential_gain_pct,
                ai_potential_gain_pct,
                primary_catalyst,
                detailed_analysis,
                model_used
            FROM alert_history
            WHERE ticker = ?
            ORDER BY timestamp_iso DESC
            LIMIT ?
            """,
            (ticker, limit)
        )
        rows = [dict(r) for r in cur.fetchall()]
        if rows:
            return rows
    finally:
        conn.close()

    # Fallback to cache (useful if DB write failed or during first writes)
    cached = [
        dict(r) for r in ALERT_HISTORY_CACHE
        if (r.get("ticker") or "").upper() == ticker.upper()
    ]
    cached.sort(key=lambda r: r.get("timestamp_iso", ""), reverse=True)
    return cached[:limit]


def fetch_all_alert_rows(limit: int = None) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(ALERT_HISTORY_DB)
    conn.row_factory = sqlite3.Row
    try:
        query = """
            SELECT
                timestamp_iso,
                date,
                time,
                profile,
                tier,
                ticker,
                entry,
                target,
                ai_target_price,
                potential_gain_pct,
                ai_potential_gain_pct,
                scan_price,
                model_used
            FROM alert_history
            ORDER BY timestamp_iso DESC
        """
        if limit:
            query += f" LIMIT {int(limit)}"
        cur = conn.execute(query)
        rows = [dict(r) for r in cur.fetchall()]
        if rows:
            return rows
    finally:
        conn.close()

    # Fallback to cache
    return [
        {
            "timestamp_iso": r.get("timestamp_iso"),
            "date": r.get("date"),
            "time": r.get("time"),
            "profile": r.get("profile"),
            "tier": r.get("tier"),
            "ticker": r.get("ticker"),
            "entry": r.get("entry"),
            "target": r.get("target"),
            "ai_target_price": r.get("ai_target_price"),
            "potential_gain_pct": r.get("potential_gain_pct"),
            "ai_potential_gain_pct": r.get("ai_potential_gain_pct"),
            "scan_price": r.get("scan_price"),
            "model_used": r.get("model_used")
        }
        for r in ALERT_HISTORY_CACHE
    ]


init_alert_history_db()
ensure_alert_history_columns()
load_alert_history_cache()


# -------------------------------------------------------------
#  Helper: Extract JSON from AI output
# -------------------------------------------------------------
def extract_json_from_text(text: str) -> str:
    """Aggressively extracts the JSON object from noisy AI output."""
    if not isinstance(text, str):
        return "{}"

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```json\s*", "", text, flags=re.DOTALL)
    text = text.replace("```", "")

    match = re.search(r"\{.*\}", text.strip(), re.DOTALL)
    if match:
        return match.group(0)

    return text.strip()


# -------------------------------------------------------------
#  Technical indicators / Supertrend helpers
# -------------------------------------------------------------
def fetch_finnhub_candles(symbol: str, days: int = 120, resolution: str = "D") -> Optional[Dict[str, Any]]:
    """Fetch historical candles for a symbol from Finnhub."""
    if not FINNHUB_API_KEY:
        return _fetch_yfinance_candles(symbol, days, resolution)

    now = datetime.utcnow()
    start = int((now - timedelta(days=days)).timestamp())
    end = int(now.timestamp())
    url = "https://finnhub.io/api/v1/stock/candle"
    params = {
        "symbol": symbol,
        "resolution": resolution,
        "from": start,
        "to": end,
        "token": FINNHUB_API_KEY,
    }
    global LAST_FINNHUB_REQUEST
    elapsed = time.monotonic() - LAST_FINNHUB_REQUEST
    if elapsed < FINNHUB_RATE_LIMIT_SECONDS:
        time.sleep(FINNHUB_RATE_LIMIT_SECONDS - elapsed)
    try:
        resp = requests.get(url, params=params, timeout=10)
        LAST_FINNHUB_REQUEST = time.monotonic()
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logging.error(f"Finnhub candle fetch error for {symbol}: {exc}")
        return _fetch_yfinance_candles(symbol, days, resolution)

    if data.get("s") != "ok":
        logging.warning("Finnhub candle response not OK for %s: %s", symbol, data.get("s"))
        return _fetch_yfinance_candles(symbol, days, resolution)
    return data


def _fetch_yfinance_candles(symbol: str, days: int, resolution: str) -> Optional[Dict[str, Any]]:
    if resolution.upper() != "D":
        return None
    if yf is None:
        return None
    if pd is None:
        return None
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=days + 5)
        df = yf.download(
            symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            progress=False
        )
    except Exception as exc:
        logging.error(f"yfinance error for {symbol}: {exc}")
        return None
    if df is None or df.empty:
        logging.warning("yfinance returned empty data for %s", symbol)
        return None
    df = df.dropna()
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    opens = df["Open"].tolist()
    highs = df["High"].tolist()
    lows = df["Low"].tolist()
    closes = df["Close"].tolist()
    volumes = df["Volume"].fillna(0).tolist()
    timestamps = [int(ts.timestamp()) for ts in df.index.to_pydatetime()]
    return {
        "o": opens,
        "h": highs,
        "l": lows,
        "c": closes,
        "v": volumes,
        "t": timestamps,
        "s": "ok"
    }


def parse_ticker_list(raw: Any) -> List[str]:
    default = list(SUPER_TREND_TICKERS)
    if raw is None or raw == "":
        return default
    if isinstance(raw, str):
        tokens = re.split(r"[,\s]+", raw.strip())
    elif isinstance(raw, list):
        tokens = raw
    else:
        raise ValueError("tickers must be a comma-separated string or list.")

    normalized: List[str] = []
    for token in tokens:
        if not token:
            continue
        val = token.strip().upper()
        if not re.fullmatch(r"[A-Z0-9.\-]{1,12}", val):
            raise ValueError(f"Invalid ticker '{token}'.")
        if val not in normalized:
            normalized.append(val)
        if len(normalized) >= MAX_CUSTOM_TICKERS:
            break

    if not normalized:
        return default
    return normalized


def compute_anchored_vwap(highs, lows, closes, volumes, start_idx: int, end_idx: int) -> Optional[float]:
    if start_idx < 0 or end_idx >= len(closes) or start_idx > end_idx:
        return None
    cumulative_price = 0.0
    cumulative_volume = 0.0
    for idx in range(start_idx, end_idx + 1):
        vol = float(volumes[idx]) if idx < len(volumes) else 0.0
        price = (float(highs[idx]) + float(lows[idx]) + float(closes[idx])) / 3.0
        cumulative_price += price * (vol if vol > 0 else 1.0)
        cumulative_volume += (vol if vol > 0 else 1.0)
    if cumulative_volume <= 0:
        return None
    return cumulative_price / cumulative_volume


def fetch_intraday_candles_yf(symbol: str, interval: str = "15m") -> Optional[Dict[str, Any]]:
    if yf is None or pd is None:
        return None
    interval = interval.lower()
    if interval not in {"1m", "5m", "15m"}:
        interval = "15m"
    period_map = {
        "1m": "5d",
        "5m": "60d",
        "15m": "60d"
    }
    period = period_map.get(interval, "60d")
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False
        )
    except Exception as exc:
        logging.error(f"Intraday yfinance error for {symbol}: {exc}")
        return None
    if df is None or df.empty:
        return None
    df = df.dropna()
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    opens = df["Open"].tolist()
    highs = df["High"].tolist()
    lows = df["Low"].tolist()
    closes = df["Close"].tolist()
    volumes = df["Volume"].fillna(0).tolist()
    timestamps = [int(ts.timestamp()) for ts in df.index.to_pydatetime()]
    return {
        "o": opens,
        "h": highs,
        "l": lows,
        "c": closes,
        "v": volumes,
        "t": timestamps,
        "s": "ok"
    }


def compute_session_vwap(candles: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    timestamps = candles.get("t") or []
    closes = candles.get("c") or []
    highs = candles.get("h") or []
    lows = candles.get("l") or []
    volumes = candles.get("v") or []
    if not timestamps:
        return None, None, None
    last_ts = int(timestamps[-1])
    last_day = datetime.utcfromtimestamp(last_ts).date()
    start_idx = 0
    for idx in range(len(timestamps) - 1, -1, -1):
        ts = datetime.utcfromtimestamp(int(timestamps[idx]))
        if ts.date() != last_day:
            start_idx = idx + 1
            break
    else:
        start_idx = 0
    if start_idx >= len(timestamps):
        start_idx = 0
    vwap = compute_anchored_vwap(highs, lows, closes, volumes, start_idx, len(timestamps) - 1)
    if vwap is None:
        return None, None, None
    last_close = float(closes[-1])
    delta = ((last_close - vwap) / vwap * 100.0) if vwap else None
    session_start_iso = datetime.utcfromtimestamp(int(timestamps[start_idx])).replace(microsecond=0).isoformat() + "Z"
    return round(vwap, 4), (round(delta, 2) if delta is not None else None), session_start_iso


def _rma(values: List[float], period: int) -> List[Optional[float]]:
    if len(values) < period:
        return [None] * len(values)
    rmas: List[Optional[float]] = [None] * len(values)
    initial = sum(values[:period]) / period
    rmas[period - 1] = initial
    prev = initial
    for idx in range(period, len(values)):
        prev = (prev * (period - 1) + values[idx]) / period
        rmas[idx] = prev
    return rmas


def _compute_supertrend(symbol: str, candles: Dict[str, Any], period: int = 10, multiplier: float = 3.0) -> Optional[Dict[str, Any]]:
    highs = candles.get("h") or []
    lows = candles.get("l") or []
    closes = candles.get("c") or []
    volumes = candles.get("v") or []
    timestamps = candles.get("t") or []
    length = min(len(highs), len(lows), len(closes), len(volumes), len(timestamps))
    if length < period + 5:
        return None

    highs = highs[-length:]
    lows = lows[-length:]
    closes = closes[-length:]
    volumes = volumes[-length:]
    timestamps = timestamps[-length:]

    trs: List[float] = []
    for idx in range(length):
        if idx == 0:
            tr = float(highs[idx]) - float(lows[idx])
        else:
            prev_close = float(closes[idx - 1])
            tr = max(
                float(highs[idx]) - float(lows[idx]),
                abs(float(highs[idx]) - prev_close),
                abs(float(lows[idx]) - prev_close)
            )
        trs.append(tr)

    atr_values = _rma(trs, period)
    src = [(float(highs[i]) + float(lows[i])) / 2.0 for i in range(length)]
    final_up: List[Optional[float]] = [None] * length
    final_dn: List[Optional[float]] = [None] * length
    trend: List[int] = [1] * length

    for idx in range(length):
        atr_val = atr_values[idx]
        if atr_val is None:
            if idx > 0:
                trend[idx] = trend[idx - 1]
                final_up[idx] = final_up[idx - 1]
                final_dn[idx] = final_dn[idx - 1]
            continue

        basic_up = src[idx] - multiplier * atr_val
        basic_dn = src[idx] + multiplier * atr_val

        if idx == 0:
            final_up[idx] = basic_up
            final_dn[idx] = basic_dn
            trend[idx] = 1
            continue

        prev_close = float(closes[idx - 1])
        prev_up = final_up[idx - 1]
        prev_dn = final_dn[idx - 1]

        if prev_up is None or prev_close <= prev_up:
            final_up[idx] = basic_up
        else:
            final_up[idx] = max(basic_up, prev_up)

        if prev_dn is None or prev_close >= prev_dn:
            final_dn[idx] = basic_dn
        else:
            final_dn[idx] = min(basic_dn, prev_dn)

        prev_trend = trend[idx - 1]
        if prev_trend == -1 and prev_dn is not None and float(closes[idx]) > prev_dn:
            trend[idx] = 1
        elif prev_trend == 1 and prev_up is not None and float(closes[idx]) < prev_up:
            trend[idx] = -1
        else:
            trend[idx] = prev_trend

    last_idx = length - 1
    prev_idx = max(last_idx - 1, 0)
    final_trend = trend[last_idx]
    buy_signal = final_trend == 1 and trend[prev_idx] == -1
    flip_idx = None
    for idx in range(last_idx, 0, -1):
        if trend[idx] != trend[idx - 1]:
            flip_idx = idx
            break

    def _ts_to_iso(ts: int) -> str:
        return datetime.utcfromtimestamp(ts).replace(microsecond=0).isoformat() + "Z"

    last_close = float(closes[last_idx])
    prev_close = float(closes[prev_idx]) if prev_idx != last_idx else last_close
    change_pct = ((last_close - prev_close) / prev_close * 100.0) if prev_close else 0.0
    flip_time = _ts_to_iso(int(timestamps[flip_idx])) if flip_idx is not None else None
    anchor_idx = flip_idx if flip_idx is not None else last_idx
    anchor_idx = max(0, min(last_idx, anchor_idx))
    anchored_vwap = compute_anchored_vwap(highs, lows, closes, volumes, anchor_idx, last_idx)
    vwap_distance = None
    if anchored_vwap:
        vwap_distance = round(((last_close - anchored_vwap) / anchored_vwap) * 100.0, 2)
        anchored_vwap = round(anchored_vwap, 4)

    return {
        "ticker": symbol.upper(),
        "close": round(last_close, 2),
        "trend": "Uptrend" if final_trend == 1 else "Downtrend",
        "trend_value": final_trend,
        "buy_signal": buy_signal,
        "supertrend_support": round(final_up[last_idx], 4) if final_up[last_idx] is not None else None,
        "supertrend_resistance": round(final_dn[last_idx], 4) if final_dn[last_idx] is not None else None,
        "atr": round(atr_values[last_idx], 4) if atr_values[last_idx] is not None else None,
        "volume": float(volumes[last_idx]),
        "change_pct": round(change_pct, 2),
        "flip_time": flip_time,
        "last_updated": _ts_to_iso(int(timestamps[last_idx])),
        "anchored_vwap": anchored_vwap,
        "distance_from_vwap_pct": vwap_distance,
    }


def compute_supertrend_signals(tickers: List[str], period: int = 10, multiplier: float = 3.0) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Compute Supertrend stats for the requested tickers."""
    results: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for symbol in tickers:
        candles = fetch_finnhub_candles(symbol)
        if not candles:
            warnings.append(f"Unable to load price history for {symbol}.")
            continue
        computed = _compute_supertrend(symbol, candles, period=period, multiplier=multiplier)
        if not computed:
            warnings.append(f"Insufficient data to compute Supertrend for {symbol}.")
            continue
        results.append(computed)
    return results, warnings


def _compute_zigzag(symbol: str, candles: Dict[str, Any], deviation: float = 5.0, backstep: int = 5) -> Optional[Dict[str, Any]]:
    closes = candles.get("c") or []
    highs = candles.get("h") or []
    lows = candles.get("l") or []
    timestamps = candles.get("t") or []
    volumes = candles.get("v") or []
    length = min(len(closes), len(highs), len(lows), len(timestamps))
    if length < backstep + 5:
        return None

    closes = closes[-length:]
    highs = highs[-length:]
    lows = lows[-length:]
    timestamps = timestamps[-length:]

    pivots: List[Dict[str, Any]] = []
    last_pivot_price = float(closes[0])
    last_pivot_idx = 0
    last_direction = 0  # -1 low, 1 high
    threshold = max(0.1, float(deviation)) / 100.0

    for idx in range(1, length):
        price = float(closes[idx])
        change = price - last_pivot_price
        change_pct = (change / last_pivot_price) if last_pivot_price else 0.0
        bars_since = idx - last_pivot_idx

        if change_pct >= threshold and bars_since >= backstep:
            last_direction = 1
            last_pivot_price = price
            last_pivot_idx = idx
            pivots.append({
                "direction": "Sell",
                "price": price,
                "index": idx,
                "timestamp": int(timestamps[idx])
            })
        elif change_pct <= -threshold and bars_since >= backstep:
            last_direction = -1
            last_pivot_price = price
            last_pivot_idx = idx
            pivots.append({
                "direction": "Buy",
                "price": price,
                "index": idx,
                "timestamp": int(timestamps[idx])
            })

    if not pivots:
        return None

    latest = pivots[-1]
    prev = pivots[-2] if len(pivots) > 1 else latest

    close = float(closes[-1])
    change_from_pivot = ((close - latest["price"]) / latest["price"] * 100.0) if latest["price"] else 0.0
    anchor_idx = max(0, min(length - 1, latest["index"]))
    anchor_end_idx = min(length - 1, len(volumes) - 1 if volumes else length - 1)
    anchored_vwap = compute_anchored_vwap(highs, lows, closes, volumes, anchor_idx, anchor_end_idx)
    vwap_distance = None
    if anchored_vwap:
        vwap_distance = round(((close - anchored_vwap) / anchored_vwap) * 100.0, 2)
        anchored_vwap = round(anchored_vwap, 4)

    def _ts(ts_val: int) -> str:
        return datetime.utcfromtimestamp(ts_val).replace(microsecond=0).isoformat() + "Z"

    return {
        "ticker": symbol.upper(),
        "direction": latest["direction"],
        "pivot_price": round(latest["price"], 2),
        "pivot_time": _ts(latest["timestamp"]),
        "prior_pivot_price": round(prev["price"], 2) if prev else None,
        "prior_direction": prev["direction"] if prev else latest["direction"],
        "change_from_pivot_pct": round(change_from_pivot, 2),
        "bars_since_pivot": length - latest["index"],
        "close": round(close, 2),
        "deviation_percent": deviation,
        "backstep_bars": backstep,
        "anchored_vwap": anchored_vwap,
        "distance_from_vwap_pct": vwap_distance,
    }


def compute_zigzag_signals(tickers: List[str], deviation: float = 5.0, backstep: int = 5) -> Tuple[List[Dict[str, Any]], List[str]]:
    results: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for symbol in tickers:
        candles = fetch_finnhub_candles(symbol)
        if not candles:
            warnings.append(f"Unable to load price history for {symbol}.")
            continue
        computed = _compute_zigzag(symbol, candles, deviation=deviation, backstep=backstep)
        if not computed:
            warnings.append(f"Insufficient data to compute ZigZag for {symbol}.")
            continue
        results.append(computed)
    return results, warnings


# -------------------------------------------------------------
#  External news helpers (Finnhub + Benzinga)
# -------------------------------------------------------------
def fetch_finnhub_news(limit: int = 50) -> Tuple[List[Dict[str, Any]], str]:
    """Fetch general market news from Finnhub and normalize."""
    if not FINNHUB_API_KEY:
        return [], "finnhub_missing_key"

    url = "https://finnhub.io/api/v1/news"
    params = {
        "category": "general",
        "token": FINNHUB_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        raw_items = resp.json()
    except Exception as e:
        logging.error(f"Finnhub news error: {e}")
        return [], "finnhub_error"

    items: List[Dict[str, Any]] = []
    for it in raw_items[:limit]:
        ts = it.get("datetime") or 0
        related = it.get("related") or ""
        tickers = [x for x in related.split(",") if x]

        items.append({
            "id": str(it.get("id") or f"finnhub_{ts}_{it.get('headline', '')[:20]}"),
            "headline": it.get("headline") or "",
            "source": "Finnhub",
            "provider": it.get("source") or "",
            "summary": it.get("summary") or "",
            "url": it.get("url") or "",
            "tickers": tickers,
            "published_ts": int(ts) if ts else 0,
        })
    return items, ""


def parse_benzinga_created(created_str: str) -> int:
    """Parse Benzinga's 'created' date string to Unix timestamp."""
    if not created_str:
        return 0
    try:
        dt = datetime.strptime(created_str, "%a, %d %b %Y %H:%M:%S %z")
        return int(dt.timestamp())
    except Exception:
        return 0


def fetch_benzinga_news(limit: int = 50) -> Tuple[List[Dict[str, Any]], str]:
    """Fetch latest Benzinga news (XML) and normalize."""
    if not BENZINGA_API_KEY:
        return [], "benzinga_missing_key"

    url = "https://api.benzinga.com/api/v2/news"
    params = {
        "token": BENZINGA_API_KEY,
        "displayOutput": "compact",
        "pageSize": min(limit, 100),
        "date": date.today().strftime("%Y-%m-%d"),
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        xml_text = resp.text
    except Exception as e:
        logging.error(f"Benzinga HTTP error: {e}")
        return [], "benzinga_error"

    try:
        root = ET.fromstring(xml_text)
    except Exception as e:
        logging.error(f"Benzinga XML parse error: {e}")
        return [], "benzinga_xml_error"

    items: List[Dict[str, Any]] = []
    for item in root.findall("item")[:limit]:
        news_id = item.findtext("id", "").strip()
        title = item.findtext("title", "") or ""
        created_str = item.findtext("created", "") or ""
        teaser = item.findtext("teaser", "") or ""
        body = item.findtext("body", "") or ""
        url_item = item.findtext("url", "") or ""

        ts = parse_benzinga_created(created_str)

        tickers: List[str] = []
        stocks_el = item.find("stocks")
        if stocks_el is not None:
            for s in stocks_el.findall("item"):
                name = s.findtext("name", "").strip()
                if name:
                    tickers.append(name)

        summary = teaser or body or ""

        items.append({
            "id": news_id or f"benzinga_{ts}_{title[:20]}",
            "headline": title,
            "source": "Benzinga",
            "provider": "Benzinga",
            "summary": summary,
            "url": url_item,
            "tickers": tickers,
            "published_ts": ts,
        })

    return items, ""


def _parse_finnhub_calendar_time(value: str) -> Optional[datetime]:
    """Parse Finnhub economic calendar times to UTC-aware datetimes."""
    if not value:
        return None
    patterns = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for pattern in patterns:
        try:
            dt = datetime.strptime(value, pattern)
            if pattern.endswith("%z"):
                return dt.astimezone(timezone.utc)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _normalize_impact_value(impact: Optional[str], importance: Any) -> str:
    if isinstance(impact, str) and impact.strip():
        return impact.strip().lower()
    try:
        level = float(importance)
        if level >= 3:
            return "high"
        if level >= 2:
            return "medium"
        if level > 0:
            return "low"
    except Exception:
        pass
    return ""


def fetch_finnhub_economic_calendar(start_date: str, end_date: str) -> Tuple[List[Dict[str, Any]], str]:
    """Fetch macro events between start_date and end_date (inclusive)."""
    if not FINNHUB_API_KEY:
        return [], "finnhub_missing_key"

    cache_key = f"finnhub:{start_date}_{end_date}"
    now = time.monotonic()
    cached = ECON_CAL_CACHE.get(cache_key)
    if cached and now - cached[0] < ECON_CAL_CACHE_TTL:
        return cached[1], ""

    url = "https://finnhub.io/api/v1/calendar/economic"
    params = {
        "from": start_date,
        "to": end_date,
        "token": FINNHUB_API_KEY
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
    except Exception as exc:
        logging.error(f"Finnhub economic calendar error: {exc}")
        return [], "finnhub_error"
    if resp.status_code == 403:
        logging.warning("Finnhub economic calendar forbidden (plan limit?): %s", resp.text[:180])
        return [], "finnhub_forbidden"
    if resp.status_code == 401:
        logging.warning("Finnhub economic calendar unauthorized: %s", resp.text[:180])
        return [], "finnhub_unauthorized"
    try:
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        logging.error(f"Finnhub economic calendar HTTP error: {exc}")
        return [], "finnhub_error"

    raw_events = payload.get("economicCalendar") or payload.get("data") or []
    events: List[Dict[str, Any]] = []
    for item in raw_events:
        event_name = (item.get("event") or item.get("title") or "").strip()
        if not event_name:
            continue
        dt_utc = _parse_finnhub_calendar_time(item.get("time") or item.get("datetime"))
        iso_utc = dt_utc.isoformat().replace("+00:00", "Z") if dt_utc else None
        ts = int(dt_utc.timestamp()) if dt_utc else None
        impact = _normalize_impact_value(item.get("impact"), item.get("importance"))
        country = (item.get("country") or "").strip().upper()
        currency = (item.get("currency") or country or "").strip().upper()
        event_id = (
            item.get("eventId")
            or item.get("id")
            or f"{event_name}_{iso_utc or ''}_{currency}"
        )

        events.append({
            "id": str(event_id),
            "event": event_name,
            "country": country,
            "currency": currency,
            "impact": impact,
            "actual": item.get("actual"),
            "forecast": item.get("forecast") or item.get("estimate"),
            "previous": item.get("previous"),
            "unit": item.get("unit"),
            "time_utc": iso_utc,
            "timestamp": ts,
            "source": "finnhub",
        })

    ECON_CAL_CACHE[cache_key] = (time.monotonic(), events)
    return events, ""


def fetch_fmp_economic_calendar(start_date: str, end_date: str) -> Tuple[List[Dict[str, Any]], str]:
    """Fallback economic calendar via Financial Modeling Prep."""
    if not FMP_API_KEY:
        return [], "fmp_missing_key"

    cache_key = f"fmp:{start_date}_{end_date}"
    now = time.monotonic()
    cached = ECON_CAL_CACHE.get(cache_key)
    if cached and now - cached[0] < ECON_CAL_CACHE_TTL:
        return cached[1], ""

    url = "https://financialmodelingprep.com/api/v3/economic_calendar"
    params = {
        "from": start_date,
        "to": end_date,
        "apikey": FMP_API_KEY
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        raw_items = resp.json()
    except Exception as exc:
        logging.error(f"FMP economic calendar error: {exc}")
        return [], "fmp_error"

    currency_guess = {
        "UNITED STATES": "USD",
        "UNITED KINGDOM": "GBP",
        "EURO AREA": "EUR",
        "GERMANY": "EUR",
        "FRANCE": "EUR",
        "ITALY": "EUR",
        "JAPAN": "JPY",
        "CHINA": "CNY",
        "CANADA": "CAD",
        "AUSTRALIA": "AUD",
        "NEW ZEALAND": "NZD",
        "SWITZERLAND": "CHF",
        "SPAIN": "EUR"
    }

    events: List[Dict[str, Any]] = []
    for item in raw_items or []:
        name = (item.get("event") or "").strip()
        if not name:
            continue
        country = (item.get("country") or "").strip()
        country_key = country.upper()
        currency = (item.get("currency") or currency_guess.get(country_key) or country_key[:3]).upper()
        raw_time = item.get("date")
        dt_utc = _parse_finnhub_calendar_time(raw_time)
        if not dt_utc and raw_time:
            try:
                dt = datetime.strptime(raw_time, "%Y-%m-%d %H:%M:%S")
                dt_utc = dt.replace(tzinfo=timezone.utc)
            except Exception:
                dt_utc = None
        iso_utc = dt_utc.isoformat().replace("+00:00", "Z") if dt_utc else None
        ts = int(dt_utc.timestamp()) if dt_utc else None

        events.append({
            "id": str(item.get("id") or f"fmp_{raw_time}_{name}"),
            "event": name,
            "country": country,
            "currency": currency,
            "impact": (item.get("importance") or "medium").strip().lower(),
            "actual": item.get("actual"),
            "forecast": item.get("forecast"),
            "previous": item.get("previous"),
            "unit": item.get("unit"),
            "time_utc": iso_utc,
            "timestamp": ts,
            "source": "fmp",
        })

    ECON_CAL_CACHE[cache_key] = (time.monotonic(), events)
    return events, ""


def _load_forexfactory_feed(url: str) -> Optional[List[Dict[str, Any]]]:
    cached = FOREXFACTORY_CACHE.get(url)
    now = time.monotonic()
    if cached and now - cached[0] < FOREXFACTORY_CACHE_TTL:
        return cached[1]

    headers = {
        "User-Agent": "curl/8.4.0",
        "Accept": "application/json,text/plain;q=0.9,*/*;q=0.8"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            FOREXFACTORY_CACHE[url] = (time.monotonic(), data)
            return data
    except Exception as exc:
        logging.warning(f"Direct ForexFactory fetch failed for {url}: {exc}")

    proxy_url = f"https://r.jina.ai/{url}"
    try:
        resp = requests.get(proxy_url, headers=headers, timeout=10)
        resp.raise_for_status()
        text = resp.text
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            json_text = text[start:end + 1]
            data = json.loads(json_text)
            if isinstance(data, list):
                FOREXFACTORY_CACHE[url] = (time.monotonic(), data)
                return data
    except Exception as exc:
        logging.error(f"Proxy ForexFactory fetch failed for {url}: {exc}")
    return None


def fetch_forexfactory_calendar(start_date: str, end_date: str) -> Tuple[List[Dict[str, Any]], str]:
    """Fallback to public ForexFactory JSON feeds (this week + next week)."""
    cache_key = f"ff:{start_date}_{end_date}"
    now = time.monotonic()
    cached = ECON_CAL_CACHE.get(cache_key)
    if cached and now - cached[0] < ECON_CAL_CACHE_TTL:
        return cached[1], ""

    events: List[Dict[str, Any]] = []
    for label, url in FOREXFACTORY_SOURCES:
        data = _load_forexfactory_feed(url)
        if not data:
            continue

        for item in data or []:
            name = (item.get("title") or "").strip()
            if not name:
                continue
            country = (item.get("country") or "").strip().upper()
            currency = country[:3] if len(country) >= 3 else country
            impact_raw = (item.get("impact") or "").strip()
            impact = impact_raw.lower()
            raw_time = item.get("date")
            dt = None
            if raw_time:
                try:
                    dt = datetime.fromisoformat(raw_time)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
                except Exception:
                    dt = None
            iso_utc = dt.isoformat().replace("+00:00", "Z") if dt else None
            ts = int(dt.timestamp()) if dt else None

            events.append({
                "id": f"ff_{label}_{raw_time}_{name}",
                "event": name,
                "country": country,
                "currency": currency,
                "impact": impact,
                "actual": item.get("actual"),
                "forecast": item.get("forecast"),
                "previous": item.get("previous"),
                "unit": item.get("unit") or "",
                "time_utc": iso_utc,
                "timestamp": ts,
                "source": "forexfactory",
            })

    if not events:
        return [], "forexfactory_error"

    # Filter to requested date range if timestamps available
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_ts = int(start_dt.replace(tzinfo=timezone.utc).timestamp())
    end_ts = int(end_dt.replace(tzinfo=timezone.utc).timestamp())

    filtered = []
    for event in events:
        ts = event.get("timestamp")
        if ts is None or (start_ts <= ts <= end_ts + 86400):
            filtered.append(event)

    ECON_CAL_CACHE[cache_key] = (time.monotonic(), filtered)
    return filtered, ""


# -------------------------------------------------------------
#  Finnhub quote + price enrichment (NO FILTERING)
# -------------------------------------------------------------
def fetch_finnhub_quote_data(symbol: str) -> Dict[str, Any]:
    """Fetch the full Finnhub quote payload for a ticker."""
    if not FINNHUB_API_KEY or not symbol:
        return {}
    url = "https://finnhub.io/api/v1/quote"
    symbol = symbol.upper()
    cache_key = symbol
    now = time.monotonic()
    cached = FINNHUB_QUOTE_CACHE.get(cache_key)
    if cached and now - cached[0] < FINNHUB_CACHE_TTL:
        return cached[1]

    params = {
        "symbol": symbol,
        "token": FINNHUB_API_KEY
    }

    global LAST_FINNHUB_REQUEST
    elapsed = now - LAST_FINNHUB_REQUEST
    if elapsed < FINNHUB_RATE_LIMIT_SECONDS:
        time.sleep(FINNHUB_RATE_LIMIT_SECONDS - elapsed)
    LAST_FINNHUB_REQUEST = time.monotonic()

    try:
        resp = requests.get(url, params=params, timeout=3)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            FINNHUB_QUOTE_CACHE[cache_key] = (time.monotonic(), data)
            return data
        FINNHUB_QUOTE_CACHE[cache_key] = (time.monotonic(), {})
    except requests.HTTPError as http_error:
        status = getattr(http_error.response, "status_code", None)
        if status == 429:
            logging.warning(f"Finnhub rate limit hit for {symbol}. Using cached empty payload for {FINNHUB_CACHE_TTL}s.")
            FINNHUB_QUOTE_CACHE[cache_key] = (time.monotonic(), {})
        else:
            logging.error(f"Finnhub quote HTTP error for {symbol}: {http_error}")
    except Exception as e:
        logging.error(f"Finnhub quote error for {symbol}: {e}")

    return FINNHUB_QUOTE_CACHE.get(cache_key, (time.monotonic(), {}))[1]


def get_finnhub_quote(symbol: str) -> float:
    """Compatibility helper: return only the current price."""
    data = fetch_finnhub_quote_data(symbol)
    price = data.get("c")
    if isinstance(price, (int, float)) and price > 0:
        return float(price)
    return None


def get_finnhub_snapshot(symbol: str) -> Dict[str, Any]:
    """Return expanded Finnhub context for a ticker (price + volume metrics)."""
    if not FINNHUB_API_KEY or not symbol:
        return {}

    symbol = symbol.upper()
    quote = fetch_finnhub_quote_data(symbol)

    metrics = {}
    metric_url = "https://finnhub.io/api/v1/stock/metric"
    params = {"symbol": symbol, "metric": "all", "token": FINNHUB_API_KEY}
    try:
        resp = requests.get(metric_url, params=params, timeout=8)
        resp.raise_for_status()
        json_data = resp.json()
        metrics = json_data.get("metric") or {}
    except Exception as exc:
        logging.warning(f"Finnhub metric fetch failed for {symbol}: {exc}")

    volume_now = metrics.get("currentVolume") or metrics.get("volume")

    snapshot = {
        "symbol": symbol,
        "current": quote.get("c"),
        "open": quote.get("o"),
        "high": quote.get("h"),
        "low": quote.get("l"),
        "previous_close": quote.get("pc"),
        "change": quote.get("c") - quote.get("pc") if quote.get("c") and quote.get("pc") else None,
        "change_pct": ((quote.get("c") - quote.get("pc")) / quote.get("pc") * 100.0) if quote.get("c") and quote.get("pc") else None,
        "timestamp": quote.get("t"),
        "avg_volume_10d": metrics.get("10DayAverageTradingVolume"),
        "avg_volume_30d": metrics.get("30DayAverageTradingVolume"),
        "volume": volume_now,
        "market_cap": metrics.get("marketCapitalization"),
        "beta": metrics.get("beta"),
        "fifty_two_week_high": metrics.get("52WeekHigh"),
        "fifty_two_week_low": metrics.get("52WeekLow"),
        "atr": metrics.get("atr"),
    }
    return {k: v for k, v in snapshot.items() if v is not None}


def enrich_scanner_with_realtime_prices(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Take the AI scanner JSON (SmallCap/MidCap/LargeCap) and:
    - Try to fetch real-time prices from Finnhub for each ticker.
    - NEVER filter or drop any ticker/alert.
    - Attach RealTimePrice when available.
    - Preserve the AI's original EntryPrice in AIEntryPrice.
    - Override EntryPrice with the real-time price (if available).
    - Recompute PotentialGainPercent using corrected EntryPrice & TargetPrice.
    """
    if not FINNHUB_API_KEY:
        return data

    buckets = ["SmallCap", "MidCap", "LargeCap"]

    tickers = set()
    for b in buckets:
        for alert in data.get(b, []) or []:
            t = (alert.get("Ticker") or "").strip().upper()
            if t:
                alert["Ticker"] = t
                tickers.add(t)

    def _to_float(val):
        try:
            return float(val)
        except Exception:
            return None

    quotes: Dict[str, float] = {}
    start = time.monotonic()
    for idx, t in enumerate(tickers, start=1):
        if time.monotonic() - start > PRICE_ENRICH_TIMEOUT:
            logging.warning(
                "Price enrichment aborted after %.1fs (%d/%d tickers processed).",
                time.monotonic() - start,
                idx - 1,
                len(tickers)
            )
            break

        price = get_finnhub_quote(t)
        if price:
            quotes[t] = price

    for b in buckets:
        alerts = data.get(b, []) or []
        for alert in alerts:
            ticker = (alert.get("Ticker") or "").strip().upper()
            if not ticker:
                continue

            target = _to_float(alert.get("TargetPrice"))
            if target is not None and "AITargetPrice" not in alert:
                alert["AITargetPrice"] = target

            ai_gain = _to_float(alert.get("PotentialGainPercent"))
            if ai_gain is not None and "AIPotentialGainPercent" not in alert:
                alert["AIPotentialGainPercent"] = ai_gain

            rt_price = quotes.get(ticker)
            if rt_price is None:
                continue

            alert["RealTimePrice"] = round(rt_price, 2)

            ai_entry = _to_float(alert.get("EntryPrice"))
            if ai_entry is not None:
                alert["AIEntryPrice"] = ai_entry

            alert["EntryPrice"] = round(rt_price, 2)

            if target is not None and rt_price > 0:
                gain_pct = (target - rt_price) / rt_price * 100.0
                alert["PotentialGainPercent"] = round(gain_pct, 2)

        data[b] = alerts

    return data


# -------------------------------------------------------------
#  Alert history recording and retrieval
# -------------------------------------------------------------
def _safe_float(val):
    try:
        return float(val)
    except Exception:
        return None


def record_alert_history(profile: str, data: Dict[str, Any], model_used: str = "") -> None:
    """Record snapshot of all alerts from a scan into persistent SQLite history."""
    now = datetime.utcnow()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")
    timestamp_iso = now.isoformat() + "Z"
    rows_to_insert: List[Tuple[Any, ...]] = []

    for tier in ["SmallCap", "MidCap", "LargeCap"]:
        for alert in data.get(tier, []) or []:
            ticker = (alert.get("Ticker") or "").strip().upper()
            if not ticker:
                continue

            ai_entry = _safe_float(alert.get("AIEntryPrice") or alert.get("EntryPrice"))
            target = _safe_float(alert.get("TargetPrice"))
            ai_target = _safe_float(alert.get("AITargetPrice") or alert.get("TargetPrice"))
            stop = _safe_float(alert.get("StopPrice"))
            rt_price = _safe_float(alert.get("RealTimePrice"))
            entry = ai_entry
            pot_gain = _safe_float(alert.get("PotentialGainPercent"))
            ai_pot_gain = _safe_float(alert.get("AIPotentialGainPercent"))

            rec = {
                "timestamp_iso": timestamp_iso,
                "date": date_str,
                "time": time_str,
                "profile": profile,
                "tier": tier,
                "ticker": ticker,
                "entry": entry,
                "target": target,
                "ai_target_price": ai_target,
                "stop": stop,
                "scan_price": rt_price,
                "potential_gain_pct": pot_gain,
                "ai_potential_gain_pct": ai_pot_gain,
                "primary_catalyst": alert.get("PrimaryCatalyst") or "",
                "detailed_analysis": alert.get("DetailedAnalysis") or "",
                "model_used": model_used or ""
            }
            ALERT_HISTORY_CACHE.append(rec)
            rows_to_insert.append(
                (
                    rec["timestamp_iso"],
                    rec["date"],
                    rec["time"],
                    rec["profile"],
                    rec["tier"],
                    rec["ticker"],
                    rec["entry"],
                    rec["target"],
                    rec["ai_target_price"],
                    rec["stop"],
                    rec["scan_price"],
                    rec["potential_gain_pct"],
                    rec["ai_potential_gain_pct"],
                    rec["primary_catalyst"],
                    rec["detailed_analysis"],
                    rec["model_used"]
                )
            )

    insert_alert_rows(rows_to_insert)


@app.route("/api/tradingview/webhook", methods=["POST"])
def tradingview_webhook():
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"success": False, "error": "Invalid JSON payload."}), 400

    provided_token = (
        request.headers.get("X-Tradingview-Token")
        or request.headers.get("X-TV-Token")
        or payload.get("token")
        or payload.get("secret")
        or request.args.get("token")
    )
    if TRADINGVIEW_WEBHOOK_TOKEN and provided_token != TRADINGVIEW_WEBHOOK_TOKEN:
        return jsonify({"success": False, "error": "Unauthorized webhook call."}), 403

    symbol = (payload.get("symbol") or payload.get("ticker") or "").strip().upper()
    timeframe = str(payload.get("tf") or payload.get("timeframe") or "").strip()
    if not symbol or not timeframe:
        return jsonify({"success": False, "error": "symbol and timeframe are required."}), 400

    timestamp = payload.get("time") or payload.get("timestamp")
    open_val = _tv_safe_float(payload.get("open"))
    high_val = _tv_safe_float(payload.get("high"))
    low_val = _tv_safe_float(payload.get("low"))
    close_val = _tv_safe_float(payload.get("close"))
    volume_val = _tv_safe_float(payload.get("volume")) or 0.0
    if None in (open_val, high_val, low_val, close_val):
        return jsonify({"success": False, "error": "Incomplete OHLC data."}), 400
    epoch = _tv_epoch_seconds(timestamp)
    if epoch is None:
        return jsonify({"success": False, "error": "Invalid timestamp supplied."}), 400

    try:
        bar_count, normalized = tradingview_relay.add_bar(
            symbol,
            timeframe,
            {
                "time": epoch,
                "open": open_val,
                "high": high_val,
                "low": low_val,
                "close": close_val,
                "volume": volume_val,
            }
        )
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        logging.error("TradingView ingestion error: %s", exc)
        return jsonify({"success": False, "error": "Failed to record bar."}), 500

    logging.info(
        "TradingView webhook bar captured: %s %s close=%s bars_cached=%d",
        symbol,
        timeframe,
        normalized.get("close"),
        bar_count
    )

    auto_signal = _tv_coerce_bool(payload.get("auto_signal"), default=True)
    force_signal = _tv_coerce_bool(payload.get("force_signal"))
    queued = False
    queue_message = None
    if auto_signal:
        queued, queue_message = maybe_queue_tradingview_signal(symbol, timeframe, force=force_signal)
        logging.info(
            "TradingView signal queue status: %s %s queued=%s reason=%s",
            symbol,
            timeframe,
            queued,
            queue_message or "ready"
        )

    key = tradingview_relay.stream_key(symbol, timeframe)
    with TRADINGVIEW_SIGNAL_LOCK:
        inflight = key in TRADINGVIEW_SIGNAL_INFLIGHT
        last_run = TRADINGVIEW_SIGNAL_LAST_RUN.get(key, 0.0)
        cooldown_remaining = 0.0
        if last_run and TRADINGVIEW_SIGNAL_COOLDOWN_SECONDS > 0:
            elapsed = time.monotonic() - last_run
            cooldown_remaining = max(0.0, TRADINGVIEW_SIGNAL_COOLDOWN_SECONDS - elapsed)

    return jsonify({
        "success": True,
        "data": {
            "symbol": symbol,
            "timeframe": timeframe,
            "bars_cached": bar_count,
            "last_bar": normalized,
            "queued_signal": queued,
            "queue_message": queue_message,
            "signal_inflight": inflight,
            "cooldown_remaining": round(cooldown_remaining, 2),
        }
    })


@app.route("/api/tradingview/signals", methods=["GET"])
@auth.login_required
def tradingview_signals_api():
    symbol = (request.args.get("symbol") or request.args.get("ticker") or "").strip().upper()
    timeframe = (request.args.get("tf") or request.args.get("timeframe") or "").strip()
    limit_param = request.args.get("limit")
    try:
        limit = int(limit_param) if limit_param else TRADINGVIEW_PROMPT_BAR_LIMIT
    except ValueError:
        limit = TRADINGVIEW_PROMPT_BAR_LIMIT
    limit = max(1, min(limit, TRADINGVIEW_MAX_BARS))

    if symbol and timeframe:
        bars = tradingview_relay.get_bars(symbol, timeframe, limit=limit)
        latest_signal = tradingview_relay.get_signal(symbol, timeframe)
        key = tradingview_relay.stream_key(symbol, timeframe)
        with TRADINGVIEW_SIGNAL_LOCK:
            inflight = key in TRADINGVIEW_SIGNAL_INFLIGHT
            last_run = TRADINGVIEW_SIGNAL_LAST_RUN.get(key, 0.0)
            cooldown_remaining = 0.0
            if last_run and TRADINGVIEW_SIGNAL_COOLDOWN_SECONDS > 0:
                elapsed = time.monotonic() - last_run
                cooldown_remaining = max(0.0, TRADINGVIEW_SIGNAL_COOLDOWN_SECONDS - elapsed)

        return jsonify({
            "success": True,
            "data": {
                "symbol": symbol,
                "timeframe": timeframe,
                "bars": bars,
                "bar_count": len(bars),
                "latest_signal": latest_signal,
                "meta": {
                    "model": TRADINGVIEW_SIGNAL_MODEL,
                    "min_bars": TRADINGVIEW_MIN_BARS_FOR_SIGNAL,
                    "prompt_bar_limit": TRADINGVIEW_PROMPT_BAR_LIMIT,
                    "signal_inflight": inflight,
                    "cooldown_remaining": round(cooldown_remaining, 2),
                }
            }
        })

    streams = tradingview_relay.describe_streams()
    return jsonify({
        "success": True,
        "data": {
            "streams": streams,
            "meta": {
                "model": TRADINGVIEW_SIGNAL_MODEL,
                "min_bars": TRADINGVIEW_MIN_BARS_FOR_SIGNAL,
                "prompt_bar_limit": TRADINGVIEW_PROMPT_BAR_LIMIT,
                "stream_count": len(streams),
            }
        }
    })


@app.route("/api/history", methods=["GET"])
@auth.login_required
def api_history():
    """
    Get history for a given ticker.
    Query param: ?ticker=TSLA
    Returns past alerts + current price and % from original entry.
    """
    ticker = (request.args.get("ticker") or "").strip().upper()
    if not ticker:
        return jsonify({"success": True, "data": []})

    records = fetch_alert_rows_by_ticker(ticker)
    if not records:
        return jsonify({"success": True, "data": []})

    current_price = get_finnhub_quote(ticker)
    for r in records:
        r["profile_key"] = r.get("profile")
        r["profile_label"] = format_profile_label(r.get("profile"))
        ai_entry = _safe_float(r.get("entry"))
        scan_entry = _safe_float(r.get("scan_price"))
        entry = scan_entry if scan_entry not in (None, 0.0) else ai_entry
        if current_price is not None and entry not in (None, 0.0):
            pct = (current_price - entry) / entry * 100.0
            r["current_price"] = round(current_price, 2)
            r["current_change_pct"] = round(pct, 2)
        else:
            r["current_price"] = None
            r["current_change_pct"] = None

    return jsonify({"success": True, "data": records})


@app.route("/api/history/summary", methods=["GET"])
@auth.login_required
def api_history_summary():
    """
    Aggregate history by ticker for the History tab.
    Returns per-ticker stats and current performance from last entry.
    """
    live_quotes_raw = (request.args.get("live_quotes") or "").strip().lower()
    if live_quotes_raw == "":
        live_quotes_enabled = True
    else:
        live_quotes_enabled = live_quotes_raw not in {"0", "false", "off", "no"}

    alert_rows = fetch_all_alert_rows(limit=2000)
    if not alert_rows:
        return jsonify({
            "success": True,
            "data": [],
            "meta": {"live_quotes": live_quotes_enabled}
        })

    agg: Dict[str, Dict[str, Any]] = {}

    for r in alert_rows:
        ticker = r.get("ticker")
        if not ticker:
            continue

        ts = r.get("timestamp_iso", "")
        ai_entry = _safe_float(r.get("entry"))
        scan_price = _safe_float(r.get("scan_price"))
        recorded_entry = scan_price if scan_price not in (None, 0.0) else ai_entry
        model_used = r.get("model_used")
        model_variant = normalize_model_variant(model_used)

        if ticker not in agg:
            agg[ticker] = {
                "ticker": ticker,
                "alerts": 0,
                "first_timestamp": ts,
                "last_timestamp": ts,
                "first_date": r.get("date"),
                "last_date": r.get("date"),
                "last_profile": format_profile_label(r.get("profile")),
                "last_profile_key": r.get("profile"),
                "last_tier": r.get("tier"),
                "last_entry": recorded_entry,
                "last_ai_entry": ai_entry,
                "last_ai_target_price": r.get("ai_target_price"),
                "last_ai_potential_gain_pct": r.get("ai_potential_gain_pct"),
                "last_scan_price": scan_price,
                "last_model": model_used,
                "last_model_variant": model_variant
            }

        a = agg[ticker]
        a["alerts"] += 1

        if ts < a["first_timestamp"]:
            a["first_timestamp"] = ts
            a["first_date"] = r.get("date")

        if ts > a["last_timestamp"]:
            a["last_timestamp"] = ts
            a["last_date"] = r.get("date")
            a["last_profile"] = format_profile_label(r.get("profile"))
            a["last_profile_key"] = r.get("profile")
            a["last_tier"] = r.get("tier")
            a["last_entry"] = recorded_entry
            a["last_ai_entry"] = ai_entry
            a["last_ai_target_price"] = r.get("ai_target_price")
            a["last_ai_potential_gain_pct"] = r.get("ai_potential_gain_pct")
            a["last_scan_price"] = scan_price
            a["last_model"] = model_used
            a["last_model_variant"] = model_variant

    # Fetch current prices and compute performance from last entry
    for ticker, a in agg.items():
        entry = _safe_float(a.get("last_entry"))
        scan_price = _safe_float(a.get("last_scan_price"))
        ref_price = scan_price if scan_price not in (None, 0.0) else entry
        if entry not in (None, 0.0) and ref_price not in (None, 0.0):
            pct = (ref_price - entry) / entry * 100.0
            a["last_known_change_pct"] = round(pct, 2)
        else:
            a["last_known_change_pct"] = None

        a["current_price"] = None
        a["current_change_pct"] = None

        if "last_profile_key" in a:
            a["last_profile"] = format_profile_label(a.get("last_profile_key"))

        # clean internal timestamps
        del a["first_timestamp"]
        del a["last_timestamp"]
        # last_scan_price should only be exposed when useful
        if a.get("last_scan_price") is None:
            a.pop("last_scan_price", None)

    items = list(agg.values())
    # Sort: best performers (highest % from last entry) first,
    # those with no current_change_pct go to bottom.
    def _summary_sort_key(row: Dict[str, Any]):
        val = row.get("current_change_pct")
        if val is None:
            val = row.get("last_known_change_pct")
        if val is None:
            return (True, 0.0)
        return (False, -val)

    items.sort(key=_summary_sort_key)

    return jsonify({
        "success": True,
        "data": items,
        "meta": {"live_quotes": live_quotes_enabled}
    })


@app.route("/api/history/live-quotes", methods=["POST"])
@auth.login_required
def api_history_live_quotes():
    """
    Fetch live quotes for a limited batch of tickers.
    This endpoint is used by the History tab to avoid blasting Finnhub
    with requests for every recorded ticker at once.
    """
    if not FINNHUB_API_KEY:
        return jsonify({"success": False, "error": "Finnhub API key not configured."}), 503

    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"success": False, "error": "Invalid JSON body."}), 400

    tickers = payload.get("tickers") or []
    if not isinstance(tickers, list):
        return jsonify({"success": False, "error": "tickers must be a list."}), 400

    normalized: List[str] = []
    for t in tickers:
        ticker = (t or "").strip().upper()
        if ticker and ticker not in normalized:
            normalized.append(ticker)

    max_batch = int(os.environ.get("HISTORY_LIVE_QUOTE_BATCH", "20"))
    trimmed = normalized[:max_batch]
    if not trimmed:
        return jsonify({"success": True, "data": []})

    results = []
    for symbol in trimmed:
        price = get_finnhub_quote(symbol)
        if price is None:
            results.append({
                "ticker": symbol,
                "current_price": None,
                "fetched_at": datetime.utcnow().isoformat() + "Z"
            })
        else:
            results.append({
                "ticker": symbol,
                "current_price": round(price, 4),
                "fetched_at": datetime.utcnow().isoformat() + "Z"
            })

    return jsonify({"success": True, "data": results})


@app.route("/api/history/deep-dive", methods=["POST"])
@auth.login_required
def api_history_deep_dive():
    if not pplx_client:
        return jsonify({"success": False, "error": "Perplexity API key not configured."}), 503

    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"success": False, "error": "Invalid JSON body."}), 400

    ticker = (payload.get("ticker") or "").strip().upper()
    if not ticker:
        return jsonify({"success": False, "error": "Ticker is required."}), 400

    snapshot = get_finnhub_snapshot(ticker)
    recent_history = fetch_alert_rows_by_ticker(ticker, limit=8)

    history_lines = []
    for row in recent_history:
        history_lines.append(
            f"{row.get('date')} {row.get('time')} | {format_profile_label(row.get('profile'))} | "
            f"{row.get('tier')} | entry {row.get('entry')} | target {row.get('target')} | "
            f"potential {row.get('potential_gain_pct')}%"
        )

    context_blob = json.dumps(
        {
            "ticker": ticker,
            "snapshot": snapshot,
            "recent_alerts": history_lines,
        },
        ensure_ascii=False
    )

    user_prompt = (
        "Use the following structured context to analyze a stock:\n\n"
        f"{context_blob}\n\n"
        "Generate the requested JSON fields strictly per the system prompt."
    )

    try:
        resp = pplx_client.chat.completions.create(
            model="sonar-reasoning-pro",
            messages=[
                {"role": "system", "content": HISTORY_DEEP_DIVE_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            extra_body={
                "search_recency_filter": "day"
            }
        )
        raw = resp.choices[0].message.content
        json_str = extract_json_from_text(raw)
        ai_data = json.loads(json_str)
    except Exception as exc:
        logging.error(f"History deep dive error for {ticker}: {type(exc).__name__} - {exc}")
        return jsonify({"success": False, "error": "AI analysis failed. Check server logs."}), 500

    return jsonify({
        "success": True,
        "data": {
            "ticker": ticker,
            "snapshot": snapshot,
            "analysis": ai_data
        }
    })


# -------------------------------------------------------------
#  ROUTES
# -------------------------------------------------------------
@app.route("/api/news/headlines", methods=["GET"])
@auth.login_required
def api_news_headlines():
    limit_param = request.args.get("limit", "50")
    try:
        limit = max(1, min(int(limit_param), 200))
    except ValueError:
        limit = 50

    all_items: List[Dict[str, Any]] = []
    errors: List[str] = []

    finnhub_items, finnhub_err = fetch_finnhub_news(limit=limit)
    if finnhub_err:
        errors.append("finnhub")
    all_items.extend(finnhub_items)

    benzinga_items, benzinga_err = fetch_benzinga_news(limit=limit)
    if benzinga_err:
        errors.append("benzinga")
    all_items.extend(benzinga_items)

    all_items.sort(key=lambda x: x.get("published_ts", 0), reverse=True)
    if len(all_items) > limit:
        all_items = all_items[:limit]

    return jsonify({
        "success": True,
        "data": all_items,
        "errors": errors
    })


@app.route("/api/news/analysis", methods=["POST"])
@auth.login_required
def api_news_analysis():
    if not pplx_client:
        return jsonify({
            "success": False,
            "error": "Perplexity API key not configured."
        }), 503

    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"success": False, "error": "Invalid JSON body"}), 400

    article = payload.get("article") or {}
    symbol = payload.get("symbol")

    headline = article.get("headline", "")
    summary = article.get("summary", "")
    provider = article.get("provider", "") or article.get("source", "")
    src = article.get("source", "")
    url = article.get("url", "")
    tickers = article.get("tickers") or []
    tickers_str = ", ".join(tickers)

    user_prompt = (
        "Here is a single news item about one or more stocks.\n\n"
        f"Headline: {headline}\n"
        f"Source/Provider: {provider or src}\n"
        f"Tickers: {tickers_str}\n"
        f"Summary: {summary}\n"
        f"URL: {url}\n"
        f"Primary symbol of interest: {symbol or 'N/A'}\n\n"
        "Return ONLY the JSON object described in the system prompt."
    )

    try:
        resp = pplx_client.chat.completions.create(
            model="sonar-reasoning-pro",
            messages=[
                {"role": "system", "content": NEWS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            extra_body={
                "search_recency_filter": "day"
            }
        )
        raw = resp.choices[0].message.content
        json_str = extract_json_from_text(raw)
        data = json.loads(json_str)

        data.setdefault("stance", "neutral")
        data.setdefault("impact_level", "medium")
        data.setdefault("summary", summary or headline)
        data.setdefault(
            "rationale",
            []
        )
        data.setdefault(
            "risk_notes",
            "Trading around news involves volatility, gaps, and uncertainty. Position sizing and risk management are critical."
        )
        data.setdefault("disclaimer", "This is not personalized financial advice.")

        return jsonify({"success": True, "data": data})
    except Exception as e:
        logging.error(f"News analysis error: {type(e).__name__} - {e}")
        return jsonify({
            "success": False,
            "error": "AI analysis failed. Check server logs."
        }), 500


@app.route("/api/econ/calendar", methods=["GET"])
@auth.login_required
def api_econ_calendar():
    range_key = (request.args.get("range") or "week").strip().lower()
    days_param = request.args.get("days")
    lookback_param = request.args.get("lookback")
    tz_minutes_param = request.args.get("tz_offset")
    limit_param = request.args.get("limit")
    impact_param = request.args.get("impact") or request.args.get("impacts") or ""
    currency_param = request.args.get("currency") or request.args.get("currencies") or ""

    today = datetime.utcnow().date()
    start_date = today
    if lookback_param:
        try:
            lookback_days = max(0, min(3, int(lookback_param)))
            start_date = start_date - timedelta(days=lookback_days)
        except Exception:
            pass

    default_days = 7
    if range_key in {"today", "day"}:
        range_days = 1
    elif range_key in {"48h", "two_days", "2d"}:
        range_days = 2
    elif range_key in {"week", "this_week"}:
        range_days = 7
    else:
        try:
            range_days = int(days_param) if days_param else default_days
        except Exception:
            range_days = default_days
        range_days = max(1, min(range_days, 14))

    end_date = start_date + timedelta(days=range_days)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    warnings: List[str] = []
    error_messages = {
        "finnhub_missing_key": "Finnhub API key not configured.",
        "finnhub_forbidden": "Finnhub rejected the request (plan restriction).",
        "finnhub_unauthorized": "Finnhub API key was rejected. Double-check the token.",
        "finnhub_error": "Finnhub economic calendar request failed.",
        "fmp_missing_key": "Financial Modeling Prep API key not configured.",
        "fmp_error": "Financial Modeling Prep economic calendar request failed.",
        "forexfactory_error": "ForexFactory feed could not be loaded.",
    }

    events: List[Dict[str, Any]] = []
    source_used: Optional[str] = None

    providers = [
        ("finnhub", fetch_finnhub_economic_calendar, True),
        ("fmp", fetch_fmp_economic_calendar, bool(FMP_API_KEY)),
        ("forexfactory", fetch_forexfactory_calendar, True),
    ]

    for name, func, enabled in providers:
        if not enabled:
            if name == "fmp":
                msg = error_messages.get("fmp_missing_key")
                if msg:
                    warnings.append(msg)
            continue
        data, err = func(start_str, end_str)
        if not err and data:
            events = data
            source_used = name
            break
        elif err:
            msg = error_messages.get(err, err)
            if msg:
                warnings.append(msg)

    if not events:
        error_text = warnings[0] if warnings else "Unable to load economic calendar."
        if len(warnings) > 1:
            error_text = f"{error_text} Fallback also failed: {'; '.join(warnings[1:])}"
        return jsonify({"success": False, "error": error_text}), 502

    impacts_filter = {
        token.strip().lower()
        for token in re.split(r"[,\s]+", impact_param)
        if token.strip()
    }
    currencies_filter = {
        token.strip().upper()
        for token in re.split(r"[,\s]+", currency_param)
        if token.strip()
    }

    try:
        tz_offset_minutes = int(tz_minutes_param)
    except Exception:
        tz_offset_minutes = 0
    tz_offset_minutes = max(-12 * 60, min(14 * 60, tz_offset_minutes))
    tz = timezone(timedelta(minutes=tz_offset_minutes))

    try:
        limit = int(limit_param) if limit_param else 80
    except Exception:
        limit = 80
    limit = max(10, min(limit, 200))

    now_ts = int(time.time())
    filtered: List[Dict[str, Any]] = []
    for item in events:
        impact = (item.get("impact") or "").strip().lower()
        if impacts_filter and impact not in impacts_filter:
            continue
        currency = (item.get("currency") or item.get("country") or "").strip().upper()
        if currencies_filter and currency not in currencies_filter:
            continue

        ts = item.get("timestamp")
        entry = dict(item)
        if ts:
            utc_dt = datetime.fromtimestamp(ts, timezone.utc)
            local_dt = utc_dt.astimezone(tz)
            entry["time_local"] = local_dt.isoformat()
            entry["time_local_label"] = local_dt.strftime("%a %b %d • %H:%M")
            entry["time_utc_label"] = utc_dt.strftime("%a %b %d • %H:%M UTC")
            entry["time_epoch_ms"] = ts * 1000
            entry["status"] = "future" if ts >= now_ts else "past"
        else:
            entry["time_local"] = None
            entry["time_local_label"] = None
            entry["time_utc_label"] = entry.get("time_utc")
            entry["time_epoch_ms"] = None
            entry["status"] = "unknown"

        entry["currency"] = currency
        filtered.append(entry)

    filtered.sort(key=lambda e: e.get("timestamp") or 0)
    if limit:
        filtered = filtered[:limit]

    return jsonify({
        "success": True,
        "data": filtered,
        "meta": {
            "range": range_key,
            "start": start_str,
            "end": end_str,
            "tz_offset_minutes": tz_offset_minutes,
            "source": source_used or "unknown",
            "filters": {
                "impacts": sorted(list(impacts_filter)),
                "currencies": sorted(list(currencies_filter)),
            },
            "warnings": warnings
        }
    })


@app.route("/api/supertrend/radar", methods=["POST"])
@auth.login_required
def api_supertrend_radar():
    if not pplx_client:
        return jsonify({"success": False, "error": "Perplexity API key not configured."}), 503
    if not FINNHUB_API_KEY:
        return jsonify({"success": False, "error": "Finnhub API key not configured."}), 503

    try:
        payload = request.get_json(silent=True) or {}
    except Exception:
        payload = {}

    period = int(payload.get("atr_period") or 10)
    multiplier = float(payload.get("atr_multiplier") or 3.0)
    period = max(4, min(period, 50))
    multiplier = max(1.0, min(multiplier, 6.0))
    try:
        tickers = parse_ticker_list(payload.get("tickers"))
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    signals, local_warnings = compute_supertrend_signals(tickers, period=period, multiplier=multiplier)
    if not signals:
        return jsonify({"success": False, "error": "Unable to compute Supertrend data for the requested tickers."}), 502

    context_blob = {
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "settings": {"atr_period": period, "multiplier": multiplier},
        "signals": signals,
        "tickers": tickers,
        "instructions": "Focus on the provided tickers only. Evaluate Supertrend buys and note catalysts within 72h."
    }

    user_prompt = (
        "Use the following JSON telemetry to evaluate Supertrend buy opportunities:\n\n"
        f"{json.dumps(context_blob, ensure_ascii=False)}\n\n"
        "Return strictly one JSON object following the provided schema."
    )

    ai_payload = {"ideas": []}
    warnings = list(local_warnings)
    try:
        resp = pplx_client.chat.completions.create(
            model="sonar-reasoning-pro",
            messages=[
                {"role": "system", "content": SUPER_TREND_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            extra_body={"search_recency_filter": "day"}
        )
        raw_content = resp.choices[0].message.content
        json_str = extract_json_from_text(raw_content)
        ai_payload = json.loads(json_str)
    except Exception as exc:
        logging.error(f"Supertrend radar AI error: {type(exc).__name__} - {exc}")
        warnings.append("AI analysis unavailable; showing raw signals only.")

    ideas = ai_payload.get("ideas") or []
    idea_map = {str(item.get("ticker", "")).upper(): item for item in ideas if isinstance(item, dict)}

    combined_signals: List[Dict[str, Any]] = []
    for sig in signals:
        ticker = sig.get("ticker")
        idea = idea_map.get(ticker)
        merged = dict(sig)
        if idea:
            merged.update({
                "ai_entry": idea.get("entry"),
                "ai_stop": idea.get("stop"),
                "ai_target": idea.get("target"),
                "ai_trend_since": idea.get("trend_since"),
                "ai_notes": idea.get("notes"),
                "ai_catalyst": idea.get("catalyst"),
                "ai_confidence": idea.get("confidence"),
                "ai_news_window": idea.get("news_window"),
                "ai_timeframe": idea.get("timeframe"),
            })
        combined_signals.append(merged)

    return jsonify({
        "success": True,
        "data": {
            "signals": combined_signals,
            "meta": {
                "generated_at": ai_payload.get("generated_at") or context_blob["generated_at"],
                "model": "sonar-reasoning-pro",
                "warnings": warnings,
                "tickers": tickers,
                "settings": {"atr_period": period, "atr_multiplier": multiplier},
            }
        }
    })


@app.route("/api/zigzag/radar", methods=["POST"])
@auth.login_required
def api_zigzag_radar():
    if not pplx_client:
        return jsonify({"success": False, "error": "Perplexity API key not configured."}), 503
    if not FINNHUB_API_KEY:
        return jsonify({"success": False, "error": "Finnhub API key not configured."}), 503

    try:
        payload = request.get_json(silent=True) or {}
    except Exception:
        payload = {}

    deviation = float(payload.get("deviation") or 5)
    backstep = int(payload.get("backstep") or 5)
    deviation = max(1.0, min(deviation, 20.0))
    backstep = max(2, min(backstep, 30))

    try:
        tickers = parse_ticker_list(payload.get("tickers"))
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    signals, local_warnings = compute_zigzag_signals(tickers, deviation=deviation, backstep=backstep)
    if not signals:
        return jsonify({"success": False, "error": "Unable to compute ZigZag data for the requested tickers."}), 502

    context_blob = {
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "settings": {"deviation_percent": deviation, "backstep_bars": backstep},
        "signals": signals,
        "tickers": tickers,
        "instructions": "Focus on the provided tickers only and highlight actionable ZigZag pivots."
    }

    user_prompt = (
        "Use the following JSON telemetry to evaluate ZigZag swing opportunities:\n\n"
        f"{json.dumps(context_blob, ensure_ascii=False)}\n\n"
        "Return strictly one JSON object following the provided schema."
    )

    ai_payload = {"ideas": []}
    warnings = list(local_warnings)
    try:
        resp = pplx_client.chat.completions.create(
            model="sonar-reasoning-pro",
            messages=[
                {"role": "system", "content": ZIGZAG_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            extra_body={"search_recency_filter": "day"}
        )
        raw_content = resp.choices[0].message.content
        json_str = extract_json_from_text(raw_content)
        ai_payload = json.loads(json_str)
    except Exception as exc:
        logging.error(f"ZigZag radar AI error: {type(exc).__name__} - {exc}")
        warnings.append("AI analysis unavailable; showing raw signals only.")

    ideas = ai_payload.get("ideas") or []
    idea_map = {str(item.get("ticker", "")).upper(): item for item in ideas if isinstance(item, dict)}

    combined_signals: List[Dict[str, Any]] = []
    for sig in signals:
        ticker = sig.get("ticker")
        idea = idea_map.get(ticker)
        merged = dict(sig)
        if idea:
            merged.update({
                "ai_entry": idea.get("entry"),
                "ai_stop": idea.get("stop"),
                "ai_target": idea.get("target"),
                "ai_notes": idea.get("notes"),
                "ai_catalyst": idea.get("catalyst"),
                "ai_confidence": idea.get("confidence"),
                "ai_timeframe": idea.get("timeframe"),
                "ai_direction": idea.get("direction"),
            })
        combined_signals.append(merged)

    return jsonify({
        "success": True,
        "data": {
            "signals": combined_signals,
            "meta": {
                "generated_at": ai_payload.get("generated_at") or context_blob["generated_at"],
                "model": "sonar-reasoning-pro",
                "warnings": warnings,
                "settings": {"deviation": deviation, "backstep": backstep},
                "tickers": tickers,
            }
        }
    })


@app.route("/api/daytrade/radar", methods=["POST"])
@auth.login_required
def api_daytrade_radar():
    if yf is None or pd is None:
        return jsonify({"success": False, "error": "yfinance dependency not available on server."}), 503

    try:
        payload = request.get_json(silent=True) or {}
    except Exception:
        payload = {}

    interval = (payload.get("interval") or "15m").lower()
    if interval not in {"1m", "5m", "15m"}:
        interval = "15m"
    atr_period = int(payload.get("atr_period") or 7)
    atr_mult = float(payload.get("atr_multiplier") or 2.0)
    deviation = float(payload.get("deviation") or 2.5)
    backstep = int(payload.get("backstep") or 4)

    atr_period = max(4, min(atr_period, 30))
    atr_mult = max(1.0, min(atr_mult, 4.0))
    deviation = max(1.0, min(deviation, 10.0))
    backstep = max(2, min(backstep, 15))

    try:
        tickers = parse_ticker_list(payload.get("tickers"))
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for symbol in tickers:
        candles = fetch_intraday_candles_yf(symbol, interval=interval)
        if not candles:
            warnings.append(f"No intraday data for {symbol}.")
            continue
        closes = candles.get("c") or []
        if not closes:
            warnings.append(f"Missing close prices for {symbol}.")
            continue
        st = _compute_supertrend(symbol, candles, period=atr_period, multiplier=atr_mult)
        zz = _compute_zigzag(symbol, candles, deviation=deviation, backstep=backstep)
        session_vwap, session_delta, session_start = compute_session_vwap(candles)
        last_close = closes[-1]
        row = {
            "ticker": symbol,
            "price": round(float(last_close), 2) if last_close is not None else None,
            "session_vwap": session_vwap,
            "session_vwap_delta_pct": session_delta,
            "session_start": session_start,
            "supertrend": st,
            "zigzag": zz,
        }
        rows.append(row)

    if not rows:
        return jsonify({"success": False, "error": "Unable to compute day-trade signals for the requested tickers."}), 502

    return jsonify({
        "success": True,
        "data": {
            "signals": rows,
            "meta": {
                "tickers": tickers,
                "interval": interval,
                "settings": {
                    "atr_period": atr_period,
                    "atr_multiplier": atr_mult,
                    "zigzag_deviation": deviation,
                    "zigzag_backstep": backstep
                },
                "warnings": warnings
            }
        }
    })


@app.route("/api/high-growth/analysis", methods=["POST"])
@auth.login_required
def high_growth_analysis_api():
    if not pplx_client:
        return jsonify({
            "success": False,
            "error": "PPLX_API_KEY is missing or invalid. Cannot connect to Perplexity."
        }), 503

    payload = request.get_json(silent=True) or {}
    candidate = payload.get("candidate") or {}
    ticker = (candidate.get("ticker") or payload.get("ticker") or "").strip().upper()
    if not ticker:
        return jsonify({
            "success": False,
            "error": "Ticker is required for analysis."
        }), 400

    context = format_candidate_context(candidate)
    user_prompt = (
        "High-growth candidate details (use exactly as provided; note missing data explicitly):\n"
        f"{context}\n\n"
        "Return ONLY the JSON object described in the system prompt."
    )

    try:
        response = pplx_client.chat.completions.create(
            model="sonar-reasoning-pro",
            messages=[
                {"role": "system", "content": HIGH_GROWTH_ANALYSIS_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            extra_body={"search_recency_filter": "week"}
        )
        raw = response.choices[0].message.content or ""
        json_str = extract_json_from_text(raw)
        data = json.loads(json_str)
        data.setdefault("ticker", ticker)
        return jsonify({"success": True, "data": data})
    except Exception as exc:
        logging.error(f"High growth analysis error: {type(exc).__name__} - {exc}")
        return jsonify({
            "success": False,
            "error": "AI analysis failed."
        }), 500


def count_alerts(data: Dict[str, Any]) -> int:
    total = 0
    for bucket in ["SmallCap", "MidCap", "LargeCap"]:
        alerts = data.get(bucket) or []
        total += len(alerts)
    return total


def dedupe_alerts(data: Dict[str, Any]) -> Dict[str, Any]:
    seen = set()
    for bucket in ["SmallCap", "MidCap", "LargeCap"]:
        unique = []
        for alert in data.get(bucket, []) or []:
            ticker = (alert.get("Ticker") or "").upper()
            key = (ticker, alert.get("SetupType"), alert.get("PrimaryCatalyst"))
            if ticker and key not in seen:
                seen.add(key)
                unique.append(alert)
        data[bucket] = unique
    return data


def normalize_high_growth_payload(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    timestamp = raw.get("timestamp")
    if not timestamp:
        timestamp = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    candidates = raw.get("candidates")
    if not isinstance(candidates, list):
        candidates = []
    normalized_candidates = [c for c in candidates if isinstance(c, dict)]
    return {
        "timestamp": timestamp,
        "candidates": normalized_candidates
    }


def _api_error_response(message: str, status_code: int):
    return jsonify({
        "success": False,
        "error": message or "Unexpected server error.",
        "data": {
            "SmallCap": [],
            "MidCap": [],
            "LargeCap": []
        }
    }), status_code


def should_trigger_fallback(profile: str, data: Dict[str, Any]) -> bool:
    high_output_profiles = ("pro_trader", "catalyst", "immediate_breakout")
    min_total = 6 if profile in high_output_profiles else 3
    min_bucket = 2 if profile in high_output_profiles else 0
    total = count_alerts(data)
    if total >= min_total:
        return False
    for bucket in ["SmallCap", "MidCap", "LargeCap"]:
        if len(data.get(bucket) or []) < min_bucket:
            return True
    return False


def merge_scanner_data(primary: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    merged = {k: list(primary.get(k, [])) for k in ["SmallCap", "MidCap", "LargeCap"]}
    for bucket in ["SmallCap", "MidCap", "LargeCap"]:
        merged[bucket].extend(fallback.get(bucket, []) or [])
    return dedupe_alerts(merged)


def build_user_prompt(profile: str) -> str:
    if profile == "high_growth":
        return (
            "Begin the high-growth innovators scan now. Surface AT LEAST six (6) and up to twelve"
            " total candidates that satisfy the system instructions—even if you must rely on proxied"
            " telemetry or older filings (label those cases explicitly).\n"
            "Your ONLY response must be one JSON object with this shape:\n"
            "{\"timestamp\": \"YYYY-MM-DDTHH:MM:SSZ\", \"candidates\": [{...}]}.\n"
            "Do not provide markdown, commentary, or multiple JSON blocks. Rank the candidates"
            " strongest→weakest within the array and avoid returning an empty list."
        )
    return (
        "Begin scan now. For each of SmallCap, MidCap, and LargeCap, "
        "try to return between 5 and 10 candidates that satisfy the rules in the system prompt.\n"
        "Only exclude names if they clearly fail the risk/reward or catalyst conditions.\n"
        "Rank the candidates in each bucket from strongest to weakest based on their risk/reward "
        "and quality of setup (strongest at the top).\n\n"
        "IMPORTANT: Your ENTIRE reply must be ONE JSON object only:\n"
        "{\"SmallCap\": [...], \"MidCap\": [...], \"LargeCap\": [...]} \n"
        "Do NOT include any explanation, commentary, or markdown outside of this JSON."
    )


def build_fallback_prompt(profile: str) -> str:
    if profile == "pro_trader":
        return (
            "The previous scan returned too few momentum setups. Relax the filters slightly: "
            "allow VolumeVsAvg down to 1.0 and LiquidityUSD down to 2,000,000 while still meeting the profile rules. "
            "Return NEW symbols not already surfaced in this session, maintaining JSON schema compliance."
        )
    if profile == "catalyst":
        return (
            "The previous scan returned too few catalyst setups. Expand the search to include credible sector/macro catalysts "
            "and events up to 5 days old. Every alert still requires a specific catalyst detail with CatalystType set accurately. "
            "Return NEW tickers not already provided."
        )
    if profile == "immediate_breakout":
        return (
            "The prior breakout scan was sparse. Expand the universe to include liquid ADRs and ETFs, allow VolumeVsAvg down to 1.0, "
            "and accept setups where the breakout trigger may occur within 10 days (state the timing explicitly). "
            "All symbols must still present actionable levels and conform to the JSON schema. Return fresh tickers not already used."
        )
    return (
        "The previous scan returned too few valid ideas. Relax non-critical filters slightly but maintain "
        "risk discipline and schema requirements. Provide additional unique tickers."
    )


def format_candidate_context(candidate: Dict[str, Any]) -> str:
    def _fmt(key: str, label: str) -> str:
        val = candidate.get(key)
        if val is None:
            return f"{label}: N/A"
        return f"{label}: {val}"

    lines = [
        _fmt("ticker", "Ticker"),
        _fmt("company_name", "Company"),
        _fmt("sector", "Sector"),
        _fmt("market_cap", "Market cap"),
        _fmt("growth_metric", "Growth metric"),
        _fmt("growth_period", "Growth period"),
        _fmt("catalyst", "Catalyst"),
        _fmt("data_source", "Source"),
        _fmt("institutional_ownership", "Institutional ownership"),
        _fmt("risk_reward", "Risk/Reward"),
        _fmt("technical_indicators", "Technical context"),
        _fmt("notes", "Notes")
    ]
    return "\n".join(lines)


def call_perplexity(
    system_prompt: str,
    user_prompt: str,
    extra_user_prompt: str = "",
    model_name: str = None
) -> Dict[str, Any]:
    raw_json_string = ""
    retries = 2
    retry_extra = extra_user_prompt or ""
    effective_model = model_name or DEFAULT_PPLX_MODEL

    for attempt in range(retries):
        try:
            messages = [{"role": "system", "content": system_prompt}]
            if retry_extra:
                user_content = f"{user_prompt}\n\nAdditional instructions:\n{retry_extra}"
            else:
                user_content = user_prompt
            messages.append({"role": "user", "content": user_content})

            response = pplx_client.chat.completions.create(
                model=effective_model,
                messages=messages,
                extra_body={"search_recency_filter": "day"}
            )
            raw_json_string = response.choices[0].message.content or ""
            json_string = extract_json_from_text(raw_json_string)
            data = json.loads(json_string)
            if not isinstance(data, dict):
                raise ValueError("Perplexity response was not a JSON object.")
            for key in ["SmallCap", "MidCap", "LargeCap"]:
                data.setdefault(key, [])
            return data
        except JSONDecodeError as exc:
            logging.warning("Perplexity JSON decode failed (attempt %d/%d): %s", attempt + 1, retries, exc)
            logging.warning("RAW AI OUTPUT RECEIVED: %r", raw_json_string[:1000])
            if attempt == retries - 1:
                logging.error("Perplexity call error: JSONDecodeError - %s", exc)
                raise
            retry_extra = (
                (extra_user_prompt + "\n\n") if extra_user_prompt else ""
            ) + "Your previous response was invalid JSON. Reply again with STRICT valid JSON only, matching the ScannerResponse schema exactly, and include required commas between properties."
            time.sleep(1)
        except Exception as exc:
            logging.error(f"Perplexity call error: {type(exc).__name__} - {exc}")
            logging.error(f"RAW AI OUTPUT RECEIVED: {repr(raw_json_string)}")
            raise


@app.route("/api/scan", methods=["POST"])
@auth.login_required
def run_scanner_api():
    """
    Scanner endpoint:
    - profile: 'hedge_fund' | 'pro_trader' | 'catalyst' (JSON body)
    - Uses Perplexity for raw ideas, then:
        * Enriches prices with Finnhub (EntryPrice & PotentialGainPercent),
        * NEVER filters / removes any ideas.
        * Records a snapshot into SQLite-based alert history for later reference.
    """
    if not pplx_client:
        return jsonify({
            "success": False,
            "error": "PPLX_API_KEY is missing or invalid. Cannot connect to Perplexity."
        }), 503

    try:
        payload = request.get_json(silent=True) or {}
        profile = (payload.get("profile") or "hedge_fund").strip().lower()
        profile = profile.replace("-", "_")
        if profile in {"growth", "highgrowth", "growth_analyst"}:
            profile = "high_growth"
        allowed_profiles = {"hedge_fund", "pro_trader", "catalyst", "bio_analyst", "immediate_breakout", "high_growth"}
        if profile not in allowed_profiles:
            profile = "hedge_fund"
        requested_model = payload.get("model_variant") or payload.get("model")
        selected_model = resolve_pplx_model(requested_model or "")
        user_allow_fallback = bool(payload.get("allow_fallback"))
        is_growth_profile = (profile == "high_growth")

        system_prompt = select_scanner_prompt(profile, selected_model)

        fallback_allowed = (
            SCANNER_ALLOW_FALLBACK or user_allow_fallback
        ) and profile not in {"hedge_fund", "high_growth"}

        logging.info("Sending request to Perplexity API for scanner profile: %s model=%s", profile, selected_model)
        primary = call_perplexity(
            system_prompt,
            build_user_prompt(profile),
            model_name=selected_model
        )

        fallback_used = False
        fallback_suppressed = False
        if (not is_growth_profile) and profile != "hedge_fund" and should_trigger_fallback(profile, primary):
            if fallback_allowed:
                logging.info("Primary scan sparse for %s; attempting fallback with model %s.", profile, selected_model)
                try:
                    fallback = call_perplexity(
                        system_prompt,
                        build_user_prompt(profile),
                        build_fallback_prompt(profile),
                        model_name=selected_model
                    )
                    primary = merge_scanner_data(primary, fallback)
                    fallback_used = True
                except Exception as exc:
                    logging.error(f"Fallback scan failed for {profile}: {exc}")
            else:
                fallback_suppressed = True
                logging.info("Fallback suppressed for profile %s (manual toggle off).", profile)

        if is_growth_profile:
            data = normalize_high_growth_payload(primary)
        else:
            data = dedupe_alerts(primary)
            if fallback_used:
                for bucket in ["SmallCap", "MidCap", "LargeCap"]:
                    for alert in data.get(bucket, []) or []:
                        alert.setdefault("Notes", "Auto-expanded criteria to satisfy idea quota.")

            try:
                data = enrich_scanner_with_realtime_prices(data)
            except Exception as e:
                logging.error(f"Price enrichment failed: {e}")

            try:
                record_alert_history(profile, data, selected_model)
            except Exception as e:
                logging.error(f"Failed to record alert history: {e}")

        meta = {
            "model": selected_model,
            "fallback_allowed": fallback_allowed,
            "fallback_used": fallback_used,
            "fallback_suppressed": fallback_suppressed
        }
        if fallback_used:
            meta["notes"] = "Relaxed criteria to round out this scan."
        elif fallback_suppressed:
            meta["notes"] = "Fallback disabled. Enable it if you need more names."

        response_payload = {"success": True, "data": data, "meta": meta}

        return jsonify(response_payload)

    except Exception as e:
        logging.error(f"Scanner error: {type(e).__name__} - {e}")

        return jsonify({
            "success": False,
            "error": "Scanner failed due to server error.",
            "data": {"SmallCap": [], "MidCap": [], "LargeCap": []}
        }), 500


@app.route("/")
@auth.login_required
def dashboard_view():
    return render_template("index.html")


@app.errorhandler(HTTPException)
def handle_http_error(error):
    if request.path.startswith("/api/"):
        return _api_error_response(error.description or error.name, error.code)
    if error.code == 401:
        return (
            "Login Required: Please enter your credentials to access the Black Box Scanner.",
            401
        )
    return error


@app.errorhandler(Exception)
def handle_unexpected_exception(error):
    logging.exception("Unhandled server error: %s", error)
    if request.path.startswith("/api/"):
        return _api_error_response("Scanner failed due to server error.", 500)
    return InternalServerError()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
