import os
import json
import re
import time
import hashlib
import logging
from datetime import datetime
from email.utils import parsedate_to_datetime

from flask import Flask, jsonify, render_template, request
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

from openai import OpenAI, APIError  # Perplexity-compatible client
import requests
from lxml import etree

# ----------------------------------
# INIT & CONFIG
# ----------------------------------
logging.basicConfig(level=logging.INFO)
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "generate_a_strong_key_in_render")

# --- AUTH SETUP ---
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


# --- Perplexity / OpenAI CONFIG ---
PERPLEXITY_API_KEY = os.environ.get("PPLX_API_KEY")

if not PERPLEXITY_API_KEY:
    logging.error("PPLX_API_KEY not found. Please set it securely.")
    client = None
else:
    client = OpenAI(
        api_key=PERPLEXITY_API_KEY,
        base_url="https://api.perplexity.ai"
    )

# --- NEWS FEED CONFIG (Finnhub + Benzinga) ---
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
BENZINGA_API_KEY = os.environ.get("BENZINGA_API_KEY")

if not FINNHUB_API_KEY:
    logging.warning("FINNHUB_API_KEY missing – /api/news/headlines will skip Finnhub.")

if not BENZINGA_API_KEY:
    logging.warning("BENZINGA_API_KEY missing – /api/news/headlines will skip Benzinga.")

NEWS_LIMIT_DEFAULT = 50  # per-source

# In-memory cache for news analysis (Perplexity)
NEWS_ANALYSIS_CACHE = {}  # {cache_key: json_obj}


# ----------------------------------
# SYSTEM PROMPTS
# ----------------------------------

SYSTEM_PROMPT = """
You are the lead Quantitative Analyst for a private hedge fund. Your primary function is to identify high-alpha swing trade signals, prioritizing **asymmetric risk/reward profiles**. The analysis must be runnable 24/7.

Execute a market scan focusing on imminent, explosive breakout potential across all US equity listings.

**I. SCREENING AND SEGREGATION:**
1.  **Segregation:** Group results into 'SmallCap', 'MidCap', and 'LargeCap'.
2.  **Breakout Confirmation (Technical):** Candidate MUST exhibit at least ONE of the following three conditions:
    a) **Volume Spike:** Current 5-day average volume is > 200% of the 20-day average. OR
    b) **Volatility Contraction:** Price is currently experiencing Volatility Contraction (ATR near 10-day low). OR
    c) **Momentum Divergence:** Confirmed **On-Balance Volume (OBV) Accumulation divergence** over the last 48 hours.

**II. ALPHA FACTOR CONFIRMATION:**
3.  **Short/Flow Factor:** Candidate MUST meet at least ONE of these conditions:
    a) **Short Squeeze Fuel:** Short Interest % of Float is > 10% OR Days to Cover (DTC) is > 3.
    b) **Sector Flow Alignment:** Stock's **Relative Strength (RS)** is outperforming its **Sector ETF** over the past 5 days.
4.  **Catalyst Validation:** Prioritize stocks with an **SEC Filing (Form 4/8-K)** or **Major News** in the last 48 hours.

**III. MANDATORY DISCIPLINE (R/R ENFORCEMENT):**
5.  **Risk/Reward Filter:** The potential profit (Target 1 - Entry) divided by the potential loss (Entry - StopLoss) **MUST be greater than or equal to 2.5:1.** If this ratio cannot be confidently established, the stock MUST be excluded from the final output.

**IV. MANDATORY OUTPUT SCHEMA:**
6.  **Output Fields:** For each candidate, you MUST provide the following keys: Ticker, EntryPrice, TargetPrice, StopPrice, PotentialGainPercent, PrimaryCatalyst (with SEC/News source), and DetailedAnalysis.
7.  **DetailedAnalysis (3-point thesis):** Point 1: **Breakout Mechanism**. Point 2: **Alpha Factor Synthesis**. Point 3: **Risk Management**.

**STRICT RULE:** If NO candidates meet the **2.5:1 R/R Filter**, return this empty JSON structure: {"SmallCap": [], "MidCap": [], "LargeCap": []}.
"""

NEWS_SYSTEM_PROMPT = """
You are an equity research analyst. 
Your job is to read a single news item about one or more stocks and explain:

1. What is happening (plain English, 2–3 sentences max).
2. How traders might view this in the short term (bullish, bearish, mixed, or neutral).
3. Key risks and what could go wrong.

You are NOT giving personalized investment advice. 
You are only describing how a typical market participant might interpret this headline.
Always be conservative and remind the user this is not financial advice.

You MUST respond with a single JSON object with these keys:

{
  "stance": "bullish | bearish | mixed | neutral",
  "summary": "short paragraph summary of the news",
  "rationale": ["bullet point 1", "bullet point 2", "bullet point 3"],
  "risk_notes": "1–2 sentence risk disclaimer",
  "disclaimer": "This is not personalized financial advice."
}
"""


# ----------------------------------
# HELPERS
# ----------------------------------

def extract_json_from_text(text: str) -> str:
    """Aggressively extracts the JSON object from noisy AI output."""
    if not text:
        return "{}"

    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'```json\s*', '', text, flags=re.DOTALL)
    text = text.replace('```', '')

    match = re.search(r'\{.*\}', text.strip(), re.DOTALL)
    if match:
        return match.group(0)
    return text


def _now_utc_ts() -> int:
    return int(time.time())


def _article_id(source: str, published: str, headline: str) -> str:
    """Stable id for cache: hash of source + time + headline."""
    h = hashlib.sha1(f"{source}|{published}|{headline}".encode("utf-8"))
    return h.hexdigest()


def fetch_finnhub_news(limit: int = 30):
    """Get latest general market news from Finnhub."""
    if not FINNHUB_API_KEY:
        return []

    url = "https://finnhub.io/api/v1/news"
    params = {
        "category": "general",
        "token": FINNHUB_API_KEY
    }
    resp = requests.get(url, params=params, timeout=5)
    resp.raise_for_status()
    data = resp.json()

    items = []
    for item in data[:limit]:
        ts = item.get("datetime")
        if isinstance(ts, (int, float)):
            published_ts = int(ts)
        else:
            published_ts = _now_utc_ts()

        headline = (item.get("headline") or "").strip()
        article_id = _article_id("finnhub", str(published_ts), headline)

        items.append({
            "id": article_id,
            "source": "Finnhub",
            "provider": item.get("source"),
            "headline": headline,
            "summary": item.get("summary") or "",
            "url": item.get("url"),
            "tickers": item.get("related", "").split(",") if item.get("related") else [],
            "published_ts": published_ts
        })
    return items


def fetch_benzinga_news(limit: int = 30):
    """
    Get latest news from Benzinga /api/v2/news (XML).
    Sorted by created time desc.
    """
    if not BENZINGA_API_KEY:
        logging.warning("BENZINGA_API_KEY missing – skipping Benzinga.")
        return []

    url = "https://api.benzinga.com/api/v2/news"
    params = {
        "token": BENZINGA_API_KEY,
        "displayOutput": "compact",
        "pageSize": min(limit, 100),
        "sort": "created:desc"
    }

    logging.info("Calling Benzinga news API (XML)...")
    resp = requests.get(url, params=params, timeout=5)
    resp.raise_for_status()

    root = etree.fromstring(resp.content)

    items = []
    for item_el in root.findall(".//item")[:limit]:
        def get_text(tag):
            el = item_el.find(tag)
            return el.text.strip() if el is not None and el.text else ""

        headline = get_text("title")
        if not headline:
            continue

        created_str = get_text("created")
        published_ts = _now_utc_ts()
        if created_str:
            try:
                dt = parsedate_to_datetime(created_str)
                published_ts = int(dt.timestamp())
            except Exception:
                pass

        article_id = get_text("id") or _article_id("benzinga", created_str, headline)
        url_news = get_text("url")

        tickers = []
        stocks_el = item_el.find("stocks")
        if stocks_el is not None:
            for s in stocks_el.findall("item"):
                name_el = s.find("name")
                if name_el is not None and name_el.text:
                    tickers.append(name_el.text.strip())

        items.append({
            "id": article_id,
            "source": "Benzinga",
            "provider": "Benzinga",
            "headline": headline,
            "summary": "",
            "url": url_news,
            "tickers": tickers,
            "published_ts": published_ts
        })

    logging.info(f"Benzinga parsed {len(items)} items from XML")
    return items


# ----------------------------------
# ROUTES – SCANNER API
# ----------------------------------

@app.route('/api/scan', methods=['POST'])
@auth.login_required
def run_scanner_api():
    if not client:
        return jsonify({
            "success": False,
            "error": "API Key is missing or invalid. Cannot connect to Perplexity."
        }), 503

    raw_json_string = ""

    try:
        logging.info("Sending request to Perplexity API with R/R filter.")

        response = client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "BEGIN SWING TRADE SCAN: "
                        "Find up to 10 candidates for each market cap tier now. "
                        "Present all price points formatted to two decimal places."
                    )
                }
            ],
            extra_body={
                "search_recency_filter": "day",
            }
        )

        raw_json_string = response.choices[0].message.content
        json_string = extract_json_from_text(raw_json_string)
        data = json.loads(json_string)

        return jsonify({"success": True, "data": data})

    except APIError as e:
        error_message = str(e)
        logging.error(f"Perplexity API Failure: {error_message}")
        if "429" in error_message or "rate limit" in error_message.lower():
            return jsonify({
                "success": False,
                "error": "Rate limit reached (429). Decrease frequency or check credits."
            }), 429
        return jsonify({
            "success": False,
            "error": f"API Request Error: {error_message}"
        }), 500

    except Exception as e:
        logging.error(f"Internal Scanner Error: {type(e).__name__} - {str(e)}")
        logging.error(f"RAW AI OUTPUT RECEIVED (Critical Debug Data): ---{repr(raw_json_string)}---")

        return jsonify({
            "success": False,
            "error": "Scanner failed. AI returned non-JSON text. Check server logs.",
            "data": {"SmallCap": [], "MidCap": [], "LargeCap": []}
        }), 500


# ----------------------------------
# ROUTES – NEWS HEADLINES + AI ANALYSIS
# ----------------------------------

@app.route("/api/news/headlines", methods=["GET"])
@auth.login_required
def api_news_headlines():
    """
    Return latest Finnhub + Benzinga headlines.

    `limit` is PER SOURCE (so you can get up to 2 * limit items).
    """
    try:
        per_source_limit = int(request.args.get("limit", NEWS_LIMIT_DEFAULT))
    except ValueError:
        per_source_limit = NEWS_LIMIT_DEFAULT

    all_items = []
    errors = []

    # Finnhub
    finnhub_items = []
    if FINNHUB_API_KEY:
        try:
            finnhub_items = fetch_finnhub_news(per_source_limit)
        except Exception as e:
            logging.error(f"Finnhub news failed: {e}")
            errors.append("finnhub")

    # Benzinga
    benzinga_items = []
    if BENZINGA_API_KEY:
        try:
            benzinga_items = fetch_benzinga_news(per_source_limit)
        except Exception as e:
            logging.error(f"Benzinga news failed: {e}")
            errors.append("benzinga")

    all_items = finnhub_items + benzinga_items
    all_items.sort(key=lambda x: x.get("published_ts", 0), reverse=True)

    logging.info(
        f"/api/news/headlines -> {len(all_items)} items "
        f"({len(finnhub_items)} Finnhub, {len(benzinga_items)} Benzinga)"
    )

    return jsonify({
        "success": True,
        "errors": errors,
        "data": all_items
    })


@app.route("/api/news/analysis", methods=["POST"])
@auth.login_required
def api_news_analysis():
    if not client:
        return jsonify({
            "success": False,
            "error": "Perplexity API is not configured."
        }), 503

    payload = request.get_json(silent=True) or {}
    article = payload.get("article") or {}
    symbol = payload.get("symbol")

    headline = (article.get("headline") or "").strip()
    summary = (article.get("summary") or "").strip()
    source = article.get("source") or ""
    url = article.get("url") or ""
    tickers = article.get("tickers") or []
    published_ts = article.get("published_ts")

    if not headline:
        return jsonify({"success": False, "error": "Missing headline"}), 400

    cache_key = _article_id(source or "unknown", str(published_ts), headline)
    if cache_key in NEWS_ANALYSIS_CACHE:
        return jsonify({
            "success": True,
            "cached": True,
            "data": NEWS_ANALYSIS_CACHE[cache_key]
        })

    user_content = f"""
News source: {source}
Headline: {headline}
Summary: {summary}
URL: {url}
Tickers: {', '.join(tickers) if tickers else 'N/A'}
Primary symbol focus: {symbol or 'N/A'}
"""

    try:
        response = client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {"role": "system", "content": NEWS_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
        )
        raw = response.choices[0].message.content
        json_string = extract_json_from_text(raw)
        data = json.loads(json_string)

        NEWS_ANALYSIS_CACHE[cache_key] = data

        return jsonify({"success": True, "cached": False, "data": data})

    except Exception as e:
        logging.error(f"News analysis failed: {e}")
        return jsonify({
            "success": False,
            "error": "AI analysis failed. Try again later."
        }), 500


# ----------------------------------
# ROUTES – UI + ERROR HANDLER
# ----------------------------------

@app.route('/')
@auth.login_required
def dashboard_view():
    return render_template('index.html')


@app.errorhandler(401)
def unauthorized(error):
    return 'Login Required: Please enter your credentials to access the Black Box Scanner.', 401


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
