import os
import json
import re
import logging
import ast
import sqlite3
import time
from json import JSONDecodeError
from datetime import datetime, date
from typing import List, Dict, Any, Tuple, Iterable

import requests
from flask import Flask, jsonify, render_template, request
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import HTTPException, InternalServerError
from dotenv import load_dotenv
from openai import OpenAI
import xml.etree.ElementTree as ET
from prompts import (
    HIGH_GROWTH_ANALYSIS_PROMPT,
    HISTORY_DEEP_DIVE_PROMPT,
    NEWS_SYSTEM_PROMPT,
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
                detailed_analysis TEXT
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
                detailed_analysis
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows
        )
        conn.commit()
    finally:
        conn.close()


ALERT_HISTORY_CACHE: List[Dict[str, Any]] = []
FINNHUB_QUOTE_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
LAST_FINNHUB_REQUEST = 0.0


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
                detailed_analysis
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
                detailed_analysis
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
                scan_price
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
            "scan_price": r.get("scan_price")
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


def record_alert_history(profile: str, data: Dict[str, Any]) -> None:
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
                "detailed_analysis": alert.get("DetailedAnalysis") or ""
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
                    rec["detailed_analysis"]
                )
            )

    insert_alert_rows(rows_to_insert)


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
        entry = r.get("entry")
        if current_price is not None and entry:
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
        entry = r.get("entry")

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
                "last_entry": entry,
                "last_ai_target_price": r.get("ai_target_price"),
                "last_ai_potential_gain_pct": r.get("ai_potential_gain_pct"),
                "last_scan_price": r.get("scan_price")
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
            a["last_entry"] = entry
            a["last_ai_target_price"] = r.get("ai_target_price")
            a["last_ai_potential_gain_pct"] = r.get("ai_potential_gain_pct")
            a["last_scan_price"] = r.get("scan_price")

    # Fetch current prices and compute performance from last entry
    for ticker, a in agg.items():
        entry = _safe_float(a.get("last_entry"))
        scan_price = _safe_float(a.get("last_scan_price"))
        if scan_price is not None and entry not in (None, 0):
            pct = (scan_price - entry) / entry * 100.0
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
                record_alert_history(profile, data)
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
