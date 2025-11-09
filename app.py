import os
import json
import re
import logging
import ast
import sqlite3
from json import JSONDecodeError
from datetime import datetime, date
from typing import List, Dict, Any, Tuple, Iterable

import requests
from flask import Flask, jsonify, render_template, request
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from openai import OpenAI
import xml.etree.ElementTree as ET

# -------------------------------------------------------------
#  Setup
# -------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
load_dotenv()

app = Flask(__name__)
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

# --- External API keys ---
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
BENZINGA_API_KEY = os.environ.get("BENZINGA_API_KEY")

PROFILE_LABELS = {
    "hedge_fund": "Hedge Fund PM",
    "pro_trader": "Pro Momentum",
    "catalyst": "News / SEC"
}


def format_profile_label(profile_key: str) -> str:
    if not profile_key:
        return ""
    return PROFILE_LABELS.get(profile_key, profile_key.replace("_", " ").title())

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
                stop REAL,
                scan_price REAL,
                potential_gain_pct REAL,
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
                stop,
                scan_price,
                potential_gain_pct,
                primary_catalyst,
                detailed_analysis
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows
        )
        conn.commit()
    finally:
        conn.close()


ALERT_HISTORY_CACHE: List[Dict[str, Any]] = []


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
                stop,
                scan_price,
                potential_gain_pct,
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
                stop,
                scan_price,
                potential_gain_pct,
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
                entry
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
            "entry": r.get("entry")
        }
        for r in ALERT_HISTORY_CACHE
    ]


init_alert_history_db()
load_alert_history_cache()

# -------------------------------------------------------------
#  SYSTEM PROMPTS
# -------------------------------------------------------------

# 1) Original Hedge Fund prompt (unchanged)
SCANNER_RESPONSE_SCHEMA = """
Unified JSON schema (copy/paste into your prompt)
{
  "type": "object",
  "title": "ScannerResponse",
  "required": ["SmallCap", "MidCap", "LargeCap"],
  "properties": {
    "SmallCap": { "type": "array", "items": { "$ref": "#/definitions/Alert" } },
    "MidCap":   { "type": "array", "items": { "$ref": "#/definitions/Alert" } },
    "LargeCap": { "type": "array", "items": { "$ref": "#/definitions/Alert" } }
  },
  "definitions": {
    "Alert": {
      "type": "object",
      "required": [
        "Ticker",
        "EntryPrice",
        "TargetPrice",
        "StopPrice",
        "RiskReward",
        "PotentialGainPercent",
        "SetupType",
        "TrendState",
        "Conviction",
        "PrimaryCatalyst",
        "CatalystType",
        "DecisionFactors",
        "DetailedAnalysis",
        "DataFreshness"
      ],
      "properties": {
        "Ticker": { "type": "string", "pattern": "^[A-Z.]{1,10}$" },

        "EntryPrice": { "type": "number" },
        "TargetPrice": { "type": "number" },
        "StopPrice":   { "type": "number" },

        "RiskReward": { "type": "number", "minimum": 0 }, 
        "PotentialGainPercent": { "type": "number" }, 

        "SetupType": {
          "type": "string",
          "enum": [
            "Breakout",
            "Retest/Support Buy",
            "Momentum Continuation",
            "Volatility Contraction",
            "Reversal",
            "News-Driven"
          ]
        },

        "TrendState": {
          "type": "string",
          "enum": ["Uptrend", "Sideways", "Downtrend"]
        },

        "Conviction": { "type": "integer", "minimum": 1, "maximum": 5 },

        "PrimaryCatalyst": { "type": "string" },

        "CatalystType": {
          "type": ["string", "null"],
          "enum": [
            "FDA",
            "SEC",
            "Earnings",
            "M&A / Strategic",
            "Guidance/Analyst",
            "Macro/Sector",
            "None",
            null
          ]
        },

        "DecisionFactors": {
          "type": "array",
          "minItems": 3,
          "maxItems": 6,
          "items": { "type": "string" },
          "description": "Most important reasons a trader would act (concise bullets)."
        },

        "DetailedAnalysis": {
          "type": "string",
          "description": "3 bullet points: (1) structure/mechanism, (2) flow/context, (3) risk/invalidations."
        },

        "DataFreshness": {
          "type": "string",
          "format": "date-time",
          "description": "ISO8601 timestamp of when this idea was formed (UTC)."
        },

        "MomentumScore": { "type": ["number", "null"], "minimum": 0, "maximum": 100 },
        "LiquidityUSD":  { "type": ["number", "null"], "description": "Avg daily dollar volume" },
        "ShortInterestFloat": { "type": ["number", "null"], "minimum": 0, "maximum": 100 },
        "RelativeStrengthVsSector": { "type": ["number", "null"], "description": "RS ratio or percentile vs sector ETF" },
        "ATRPercent": { "type": ["number", "null"], "description": "ATR as % of price" },
        "VolumeVsAvg": { "type": ["number", "null"], "description": "Current volume vs 30d avg, e.g., 1.8 = 180%" },

        "Notes": { "type": ["string", "null"] },

        "AIEntryPrice": { "type": ["number", "null"], "description": "Optional: original AI entry before any downstream overwrite." }
      },
      "additionalProperties": false
    }
  },
  "additionalProperties": false
}

Output rules to include at the end of each system prompt

STRICT OUTPUT: Return exactly one JSON object conforming to ScannerResponse.

No markdown, no explanations, no prose outside JSON.

Numbers as numbers (no currency symbols or % signs).

If a value is unavailable, use null (not empty strings).

Aim 5–10 alerts per bucket when available; otherwise return as many valid as exist (can be 0).

Order each bucket strongest→weakest based on setup quality + R/R + catalyst power (for catalyst profile).

Example (one alert object only, for clarity)
{
  "SmallCap": [
    {
      "Ticker": "RXRX",
      "EntryPrice": 9.8,
      "TargetPrice": 12.5,
      "StopPrice": 8.9,
      "RiskReward": 3.0,
      "PotentialGainPercent": 27.55,
      "SetupType": "News-Driven",
      "TrendState": "Uptrend",
      "Conviction": 4,
      "PrimaryCatalyst": "FDA fast track for lead asset; strong volume expansion",
      "CatalystType": "FDA",
      "DecisionFactors": [
        "Powerful FDA catalyst with immediate price/volume confirmation",
        "Funds rotation into SMID biotech; RS vs XBI elevated",
        "Defined invalidation below post-news range low"
      ],
      "DetailedAnalysis": "- Breakout from multi-week base on news\\n- Options flow and RVOL confirm demand\\n- Invalidate on close below 8.9 to protect capital",
      "DataFreshness": "2025-11-09T02:15:00Z",
      "MomentumScore": 78,
      "LiquidityUSD": 145000000,
      "ShortInterestFloat": 12.4,
      "RelativeStrengthVsSector": 1.23,
      "ATRPercent": 6.1,
      "VolumeVsAvg": 2.6,
      "Notes": null,
      "AIEntryPrice": null
    }
  ],
  "MidCap": [],
  "LargeCap": []
}
"""

# 1) Original Hedge Fund prompt (risk-first, asymmetric R/R)
HEDGE_FUND_PROMPT = """
SYSTEM
You are the lead Quant analyst for a private hedge fund. Identify high-alpha swing trades with asymmetric risk/reward and institutional quality. Scan all US equities.

Segregation: Bucket ideas into SmallCap, MidCap, LargeCap.

Technical admission (must have ≥1):

5-day avg volume ≥ 200% of 20-day avg, or

Volatility contraction (ATR near 10-day low), or

OBV accumulation divergence over last 48h.

Flow/positioning (must have ≥1):

Short interest ≥ 10% float or Days-to-Cover > 3,

Relative strength vs its sector ETF over past 5 days.

Catalyst preference: SEC (8-K / Form 4 cluster), FDA, earnings surprises, credible corp. actions in last 72h.

Risk discipline: R/R ≥ 2.5:1 ( (Target−Entry)/(Entry−Stop) ).

Quantity & ranking: Aim 5–10 per bucket when possible. Rank strongest→weakest by quality of setup and R/R.

OUTPUT: Return one JSON object only conforming exactly to the ScannerResponse schema below. No prose, no markdown.
""" + SCANNER_RESPONSE_SCHEMA

# 2) Pro Momentum Trader (idea breadth, liquid momentum)
PRO_TRADER_PROMPT = """
SYSTEM
You are an aggressive momentum trader surfacing liquid swing setups (days→weeks).

Segregation: SmallCap / MidCap / LargeCap.

Filters: Avoid illiquid micros; focus on adequate dollar volume.

Momentum admission (≥1):

Breakout or base retest at key level,

Trend continuation after MA pullback,

Short-term momentum spike with volume confirmation.

Catalyst: Nice-to-have; tag if present.

Risk discipline: R/R ≥ 2.0:1.

Momentum telemetry: Only include tickers where you can provide BOTH MomentumScore (0‑100) and VolumeVsAvg (≥1.2). If that data is unavailable, pick another name. Favor liquid names (LiquidityUSD ≥ $5M) and ensure these setups feel distinct from catalyst ideas (no reliance on fresh news).

Quantity & ranking: 5–10 per bucket if feasible, rank strongest→weakest.

OUTPUT: Exactly the ScannerResponse schema. JSON only.
""" + SCANNER_RESPONSE_SCHEMA

# 3) News / Catalyst Hunter (event-driven)
CATALYST_HUNTER_PROMPT = """
SYSTEM
You are a catalyst trader prioritizing fresh, high-impact events in last 72h:

FDA actions,

SEC filings with impact (8-K, merger, Form 4 clusters),

Earnings shocks/guidance,

M&A / strategic deals / buybacks / spin-offs,

Major macro/sector news that directly affects the ticker.

Technical preference: Clean reaction (breakout/base/trend resumption). Avoid one-off illiquid spikes.

Risk guideline: Prefer R/R ≥ 2.0:1; include slightly lower only if catalyst is exceptionally strong (flag risk in analysis).

Quantity & ranking: 5–10 per bucket if news flow allows. Rank strongest→weakest by catalyst power, technicals, and R/R.

Catalyst enforcement: Every alert MUST cite a concrete event (FDA, SEC, Earnings, M&A / Strategic, Guidance/Analyst, Macro/Sector). Set CatalystType accordingly (never "None"), describe the event inside PrimaryCatalyst, and ensure DecisionFactors reference it specifically. Skip any ticker lacking a verifiable catalyst within 72h.

OUTPUT: Exactly the ScannerResponse schema. JSON only.
""" + SCANNER_RESPONSE_SCHEMA

# News AI system prompt (emphasize decision factors)
NEWS_SYSTEM_PROMPT = """
You are an equity research analyst. 
Your job is to read a single news item about one or more stocks and explain:

1. What is happening (plain English, 2–3 sentences max).
2. How traders might view this in the short term (bullish, bearish, mixed, or neutral).
3. Whether the news is likely to have low, medium, or high price impact in the near term.
4. The most important decision-making factors for traders (3 concise bullets).
5. Key risks and what could go wrong.

You are NOT giving personalized investment advice. 
You are only describing how a typical market participant might interpret this headline.
Always be conservative and remind the user this is not financial advice.

You MUST respond with a single JSON object with these keys:

{
  "stance": "bullish | bearish | mixed | neutral",
  "impact_level": "low | medium | high",
  "summary": "short paragraph summary of the news",
  "rationale": [
    "factor 1: most important trading driver (catalyst, revenue, growth, guidance, etc.)",
    "factor 2: positioning / sentiment / flow or competitive context",
    "factor 3: key uncertainty or condition that really matters for traders"
  ],
  "risk_notes": "1–2 sentence risk disclaimer about volatility and uncertainty",
  "disclaimer": "This is not personalized financial advice."
}

Return ONLY this JSON object and nothing else.
"""

HISTORY_DEEP_DIVE_PROMPT = """
You are an elite multi-factor trading desk assistant. Given context about a US-listed equity
(recent alerts, trader profile, price snapshot, volume stats), produce a dense JSON advisory
that helps a discretionary trader decide what to do right now. Blend technical, momentum, flow,
and catalyst-driven reasoning. Be concise but actionable.

Return ONLY a JSON object with this exact shape:
{
  "stance": "bullish | bearish | neutral",
  "summary": "2-3 sentences synthesizing price action, catalysts, and risk/reward",
  "catalysts": ["bullet about catalyst or news", "..."],
  "levels": {
    "immediate_support": "price + why",
    "immediate_resistance": "price + why",
    "support_zones": ["price + reason", "..."],
    "resistance_zones": ["price + reason", "..."]
  },
  "momentum": [
    {"indicator": "RSI", "reading": "value/condition", "bias": "bullish/bearish/neutral"},
    {"indicator": "MACD", "reading": "...", "bias": "..."}
  ],
  "volume": {
    "today_vs_avg": "describe current vs 10/30 day average volume",
    "notes": "anything notable about participation or liquidity"
  },
  "action_plan": {
    "bias": "Buy breakout / Short pop / Wait, etc.",
    "entries": ["price zone and trigger"],
    "targets": ["near-term target with rationale", "..."],
    "stops": ["protective level and why"]
  },
  "future_view": "short paragraph giving next 1-2 week outlook (include Fibonacci/ADX/CCI references when useful)",
  "risk_notes": "how it could fail, gaps, catalysts to monitor",
  "disclaimer": "This is not personalized financial advice."
}

Always ground levels in the provided price snapshot. If data is missing, acknowledge it briefly.
"""

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
    params = {
        "symbol": symbol.upper(),
        "token": FINNHUB_API_KEY
    }
    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logging.error(f"Finnhub quote error for {symbol}: {e}")
    return {}


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

    quotes: Dict[str, float] = {}
    for t in tickers:
        price = get_finnhub_quote(t)
        if price:
            quotes[t] = price

    for b in buckets:
        alerts = data.get(b, []) or []
        for alert in alerts:
            ticker = (alert.get("Ticker") or "").strip().upper()
            if not ticker:
                continue

            rt_price = quotes.get(ticker)
            if rt_price is None:
                continue

            alert["RealTimePrice"] = round(rt_price, 2)

            def _to_float(val):
                try:
                    return float(val)
                except Exception:
                    return None

            ai_entry = _to_float(alert.get("EntryPrice"))
            if ai_entry is not None:
                alert["AIEntryPrice"] = ai_entry

            alert["EntryPrice"] = round(rt_price, 2)

            target = _to_float(alert.get("TargetPrice"))
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

            entry = _safe_float(alert.get("EntryPrice"))
            target = _safe_float(alert.get("TargetPrice"))
            stop = _safe_float(alert.get("StopPrice"))
            rt_price = _safe_float(alert.get("RealTimePrice"))
            pot_gain = _safe_float(alert.get("PotentialGainPercent"))

            rec = {
                "timestamp_iso": timestamp_iso,
                "date": date_str,
                "time": time_str,
                "profile": profile,
                "tier": tier,
                "ticker": ticker,
                "entry": entry,
                "target": target,
                "stop": stop,
                "scan_price": rt_price,
                "potential_gain_pct": pot_gain,
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
                    rec["stop"],
                    rec["scan_price"],
                    rec["potential_gain_pct"],
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
    alert_rows = fetch_all_alert_rows(limit=2000)
    if not alert_rows:
        return jsonify({"success": True, "data": []})

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
                "last_entry": entry
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

    # Fetch current prices and compute performance from last entry
    for ticker, a in agg.items():
        cp = get_finnhub_quote(ticker)
        entry = a.get("last_entry")
        if cp is not None and entry:
            pct = (cp - entry) / entry * 100.0
            a["current_price"] = round(cp, 2)
            a["current_change_pct"] = round(pct, 2)
        else:
            a["current_price"] = None
            a["current_change_pct"] = None

        if "last_profile_key" in a:
            a["last_profile"] = format_profile_label(a.get("last_profile_key"))

        # clean internal timestamps
        del a["first_timestamp"]
        del a["last_timestamp"]

    items = list(agg.values())
    # Sort: best performers (highest % from last entry) first,
    # those with no current_change_pct go to bottom.
    items.sort(key=lambda x: (x["current_change_pct"] is None, -(x["current_change_pct"] or 0.0)))

    return jsonify({"success": True, "data": items})


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
            model="sonar-pro",
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
            model="sonar-pro",
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


def should_trigger_fallback(profile: str, data: Dict[str, Any]) -> bool:
    min_total = 6 if profile in ("pro_trader", "catalyst") else 3
    min_bucket = 2 if profile in ("pro_trader", "catalyst") else 0
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
    return (
        "The previous scan returned too few valid ideas. Relax non-critical filters slightly but maintain "
        "risk discipline and schema requirements. Provide additional unique tickers."
    )


def call_perplexity(system_prompt: str, user_prompt: str, extra_user_prompt: str = "") -> Dict[str, Any]:
    raw_json_string = ""
    try:
        messages = [{"role": "system", "content": system_prompt}]
        if extra_user_prompt:
            user_content = f"{user_prompt}\n\nAdditional instructions:\n{extra_user_prompt}"
        else:
            user_content = user_prompt
        messages.append({"role": "user", "content": user_content})

        response = pplx_client.chat.completions.create(
            model="sonar-pro",
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
        profile = (payload.get("profile") or "hedge_fund").lower()

        if profile == "pro_trader":
            system_prompt = PRO_TRADER_PROMPT
        elif profile == "catalyst":
            system_prompt = CATALYST_HUNTER_PROMPT
        else:
            system_prompt = HEDGE_FUND_PROMPT
            profile = "hedge_fund"

        logging.info(f"Sending request to Perplexity API for scanner profile: {profile}")
        primary = call_perplexity(system_prompt, build_user_prompt(profile))

        fallback_used = False
        if profile != "hedge_fund" and should_trigger_fallback(profile, primary):
            logging.info(f"Primary scan sparse for profile {profile}; attempting fallback.")
            try:
                fallback = call_perplexity(
                    system_prompt,
                    build_user_prompt(profile),
                    build_fallback_prompt(profile)
                )
                primary = merge_scanner_data(primary, fallback)
                fallback_used = True
            except Exception as exc:
                logging.error(f"Fallback scan failed for {profile}: {exc}")

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

        meta = {}
        if fallback_used:
            meta["notes"] = "Relaxed criteria to round out this scan."

        response_payload = {"success": True, "data": data}
        if meta:
            response_payload["meta"] = meta

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


@app.errorhandler(401)
def unauthorized(error):
    return 'Login Required: Please enter your credentials to access the Black Box Scanner.', 401


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
