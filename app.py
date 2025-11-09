import os
import json
import re
import logging
import ast
from json import JSONDecodeError
from datetime import datetime, date
from typing import List, Dict, Any, Tuple, Optional

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

app = Flask(__name__, template_folder="templates", static_folder="static")
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

# --- Perplexity Client (optional) ---
PERPLEXITY_API_KEY = os.environ.get("PPLX_API_KEY")
if not PERPLEXITY_API_KEY:
    logging.warning("PPLX_API_KEY not found. Perplexity features will be disabled; using local fallbacks.")
    pplx_client = None
else:
    pplx_client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")

# --- External API keys ---
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
BENZINGA_API_KEY = os.environ.get("BENZINGA_API_KEY")

# -------------------------------------------------------------
#  SYSTEM PROMPTS (your original ones)
# -------------------------------------------------------------

# 1) Original Hedge Fund prompt (unchanged)
HEDGE_FUND_PROMPT = """
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

**STRICT RULE:** If NO candidates meet the **2.5:1 R/R Filter**, return this empty JSON structure:
{"SmallCap": [], "MidCap": [], "LargeCap": []}.

**OUTPUT FORMAT (NON-NEGOTIABLE):**
You MUST respond with a single JSON object and NOTHING ELSE.
No prose, no markdown, no backticks.

The JSON must have exactly these top-level keys:
{
  "SmallCap": [...],
  "MidCap": [...],
  "LargeCap": [...]
}
"""

# 2) Pro Trader / momentum mode – tries to give 5–10 per bucket
PRO_TRADER_PROMPT = """
You are acting as an aggressive professional swing trader focused on **liquid momentum setups**.
You scan US equities for near-term moves (several days to a few weeks).

Your job is to **surface as many valid ideas as possible**, not just 2–3 perfect examples.
Aim for **between 5 and 10 candidates per market cap bucket** if the market conditions allow it.
Within each bucket, **rank the candidates from strongest to weakest** based on risk/reward and quality of setup
(the strongest, cleanest momentum / R/R trades should be listed first).

**I. SCREENING AND SEGREGATION:**
1. **Segregation:** Group results into 'SmallCap', 'MidCap', and 'LargeCap'.
2. **Price & Liquidity:** 
   - Ignore micro illiquid names. Focus on tickers with sufficient average daily dollar volume.
3. **Momentum / Breakout Template:** Each candidate MUST show at least ONE:
   a) Breakout or retest of a key level (recent highs, range break, multi-day base).  
   b) Strong momentum continuation after a pullback to moving averages.  
   c) Intraday/short-term momentum spike with volume confirmation.

**II. CONTEXT FACTORS:**
4. **Flow & Positioning:** Prefer:
   - Elevated relative volume
   - Options activity / short interest context where applicable
5. **News/Catalyst:** Helpful but NOT strictly required. Mark when a catalyst is present 
   (earnings, guidance change, rating change, etc.).

**III. RISK/REWARD DISCIPLINE:**
6. The potential profit (Target 1 - Entry) divided by the potential loss (Entry - StopLoss) 
   **MUST be ≥ 2.0:1**.  
   If 2.0:1 cannot be reasonably defined, exclude the stock.

**IV. OUTPUT SCHEMA:**
7. For each candidate, output **exactly**:
   Ticker, EntryPrice, TargetPrice, StopPrice, PotentialGainPercent, 
   PrimaryCatalyst (or "None / purely technical"), DetailedAnalysis.

**DetailedAnalysis (3 bullet thesis):**
   - Point 1: Momentum / breakout structure.
   - Point 2: Supporting flows / context.
   - Point 3: Risk management and invalidation.

**IMPORTANT QUANTITY RULE:**
- Do NOT arbitrarily stop after 2–3 ideas.
- If the market supports it, keep listing candidates until you reach **about 10 per bucket**
  or you genuinely run out of names that satisfy the rules.

If truly **no** tickers satisfy R/R ≥ 2.0:1, you MUST return:
{"SmallCap": [], "MidCap": [], "LargeCap": []}.

**OUTPUT FORMAT (NON-NEGOTIABLE):**
You MUST respond with a single JSON object and NOTHING ELSE.
No prose, no markdown, no backticks.

The JSON must have exactly these top-level keys:
{
  "SmallCap": [...],
  "MidCap": [...],
  "LargeCap": [...]
}
"""

# 3) Catalyst hunter – news-driven, 5–10 per bucket if possible
CATALYST_HUNTER_PROMPT = """
You are a **news catalyst trader** whose entire focus is on stocks with **fresh, high-impact events**.
Scan all US equities for **the strongest near-term catalysts**.

Your goal is to surface **as many high-quality catalyst trades as possible**, not just a few.
Aim for **5–10 candidates per market cap bucket** if the news flow supports it.
Within each bucket, **rank the candidates from strongest to weakest** based on:
  - Power of the catalyst (FDA > SEC filing > earnings > misc news, etc.)
  - Clean technical reaction
  - Risk/reward profile.

**I. CATALYST FIRST (MANDATORY):**
Each candidate MUST have at least ONE of the following in the last 72 hours:
  - **FDA** action (approval, CRL, major trial data)  
  - **SEC** filing impact (8-K, merger agreement, major 10-K/10-Q surprise, Form 4 cluster buys/sells)  
  - **Earnings**: major beat/miss, big guidance change, or unusual reaction vs expectations  
  - **Corporate events**: M&A, strategic partnership, spin-off, large buyback, special dividend  
  - **Major macro/sector news** that directly affects the ticker.

If there is no meaningful new catalyst, the stock MUST be excluded.

**II. TECHNICAL / POSITIONING CONTEXT:**
1. Prefer clean technical structures:
   - Breakout from base or strong trend resumption.
   - Avoid random illiquid spikes with no follow-through.
2. Highlight when:
   - Short interest is significant (>10% of float) or
   - Stock is heavily owned by funds / in key ETFs.

**III. RISK/REWARD FRAMEWORK:**
3. The potential profit (Target 1 - Entry) divided by the potential loss (Entry - StopLoss) 
   **should be ≥ 2.0:1**.  
   If the catalyst is extremely strong but the R/R is slightly lower, you may still include it, 
   but clearly flag risk in the DetailedAnalysis.

**IV. OUTPUT SCHEMA (STRICT):**
4. For each candidate, you MUST output a JSON structure with three buckets: 
   "SmallCap", "MidCap", "LargeCap".
5. Each entry MUST include:
   - Ticker
   - EntryPrice
   - TargetPrice
   - StopPrice
   - PotentialGainPercent
   - PrimaryCatalyst
   - DetailedAnalysis

**DetailedAnalysis (3 bullet thesis):**
   - Point 1: What the catalyst actually is and why it matters.
   - Point 2: How price/volume is reacting to it.
   - Point 3: Risk management and what invalidates the catalyst trade.

**IMPORTANT QUANTITY RULE:**
- Do NOT arbitrarily stop after 2–3 names.
- As long as the catalysts are real and recent, keep adding candidates until you reach 
  about **10 per bucket**, or you genuinely run out of strong setups.

If **no** strong catalysts are available, return:
{"SmallCap": [], "MidCap": [], "LargeCap": []}.

**OUTPUT FORMAT (NON-NEGOTIABLE):**
You MUST respond with a single JSON object and NOTHING ELSE.
No prose, no markdown, no backticks.

The JSON must have exactly these top-level keys:
{
  "SmallCap": [...],
  "MidCap": [...],
  "LargeCap": [...]
}
"""

# News AI system prompt
NEWS_SYSTEM_PROMPT = """
You are an equity research analyst. 
Your job is to read a single news item about one or more stocks and explain:

1. What is happening (plain English, 2–3 sentences max).
2. How traders might view this in the short term (bullish, bearish, mixed, or neutral).
3. Whether the news is likely to have low, medium, or high price impact in the near term.
4. Key risks and what could go wrong.

You are NOT giving personalized investment advice. 
You are only describing how a typical market participant might interpret this headline.
Always be conservative and remind the user this is not financial advice.

You MUST respond with a single JSON object with these keys:

{
  "stance": "bullish | bearish | mixed | neutral",
  "impact_level": "low | medium | high",
  "summary": "short paragraph summary of the news",
  "rationale": ["bullet point 1", "bullet point 2", "bullet point 3"],
  "risk_notes": "1–2 sentence risk disclaimer",
  "disclaimer": "This is not personalized financial advice."
}
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
#  External news helpers
# -------------------------------------------------------------
def parse_benzinga_created(created_str: str) -> int:
    if not created_str:
        return 0
    try:
        dt = datetime.strptime(created_str, "%a, %d %b %Y %H:%M:%S %z")
        return int(dt.timestamp())
    except Exception:
        return 0

def fetch_finnhub_news(limit: int = 50) -> Tuple[List[Dict[str, Any]], str]:
    if not FINNHUB_API_KEY:
        return [], "finnhub_missing_key"
    url = "https://finnhub.io/api/v1/news"
    params = {"category": "general", "token": FINNHUB_API_KEY}
    try:
        resp = requests.get(url, params=params, timeout=12)
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
            "id": str(it.get("id") or f"finnhub_{ts}_{it.get('headline','')[:20]}"),
            "headline": it.get("headline") or "",
            "source": "Finnhub",
            "provider": it.get("source") or "",
            "summary": it.get("summary") or "",
            "url": it.get("url") or "",
            "tickers": tickers,
            "published_ts": int(ts) if ts else 0,
        })
    return items, ""

def fetch_benzinga_news(limit: int = 50) -> Tuple[List[Dict[str, Any]], str]:
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
        resp = requests.get(url, params=params, timeout=12)
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
        news_id = (item.findtext("id") or "").strip()
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
def get_finnhub_quote(symbol: str) -> Optional[float]:
    if not FINNHUB_API_KEY or not symbol:
        return None
    url = "https://finnhub.io/api/v1/quote"
    params = {"symbol": symbol.upper(), "token": FINNHUB_API_KEY}
    try:
        resp = requests.get(url, params=params, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        price = data.get("c")
        if isinstance(price, (int, float)) and price > 0:
            return float(price)
    except Exception as e:
        logging.error(f"Finnhub quote error for {symbol}: {e}")
    return None

def enrich_scanner_with_realtime_prices(data: Dict[str, Any]) -> Dict[str, Any]:
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
        p = get_finnhub_quote(t)
        if p:
            quotes[t] = p
    for b in buckets:
        for alert in data.get(b, []) or []:
            t = alert.get("Ticker")
            rt = quotes.get(t)
            if rt is None:
                continue
            alert["RealTimePrice"] = round(rt, 2)
            # Preserve AI entry if provided
            try:
                ai_entry = float(alert.get("EntryPrice"))
                alert["AIEntryPrice"] = ai_entry
            except Exception:
                ai_entry = None
            # Override EntryPrice with real-time
            alert["EntryPrice"] = round(rt, 2)
            # Recompute gain if target is numeric
            try:
                target = float(alert.get("TargetPrice"))
                gain = (target - rt) / rt * 100.0
                alert["PotentialGainPercent"] = round(gain, 2)
            except Exception:
                pass
    return data

# -------------------------------------------------------------
#  NEW: Coerce AI payload into proper {Ticker,...} objects
# -------------------------------------------------------------
def coerce_scanner_payload(raw: Any) -> Dict[str, Any]:
    """
    Accepts whatever the AI sent and tries to turn it into:
    { "SmallCap":[{...}], "MidCap":[{...}], "LargeCap":[{...}] }

    - Keeps existing dict objects as-is.
    - Converts numbered string lines like "1. Microsoft (MSFT)..."
      into full alert dicts with at least Ticker + DetailedAnalysis.
    - Ignores descriptive lines (e.g. "Small-cap companies with ...").
    """
    buckets = ["SmallCap", "MidCap", "LargeCap"]
    result = {b: [] for b in buckets}

    if not isinstance(raw, dict):
        return result

    for bucket in buckets:
        alerts = raw.get(bucket, []) or []
        normalized: List[Dict[str, Any]] = []

        for item in alerts:
            # Already well-structured
            if isinstance(item, dict):
                normalized.append(item)
                continue

            # We only try to parse string lines
            if isinstance(item, str):
                # Only lines that look like "1. Something..."
                if not re.match(r"^\s*\d+\.", item):
                    continue
                # Try to extract ticker in parentheses (MSFT), (NVDA), etc.
                m = re.search(r"\(([A-Z\.]{1,6})\)", item)
                if not m:
                    continue
                ticker = m.group(1).strip().upper()
                normalized.append({
                    "Ticker": ticker,
                    "EntryPrice": None,
                    "TargetPrice": None,
                    "StopPrice": None,
                    "PotentialGainPercent": None,
                    "PrimaryCatalyst": "Example candidate from AI without numeric levels.",
                    "DetailedAnalysis": [item]
                })

        result[bucket] = normalized

    return result

# -------------------------------------------------------------
#  Local fallbacks (no external AI)
# -------------------------------------------------------------
def fallback_scan() -> Dict[str, Any]:
    """Return a small, valid scan payload when Perplexity fails, so UI remains usable."""
    sample = {
        "SmallCap": [],
        "MidCap": [
            {
                "Ticker": "RWAY",
                "EntryPrice": 9.89,
                "TargetPrice": 11.03,
                "StopPrice": 10.22,
                "PotentialGainPercent": 7.61,
                "PrimaryCatalyst": "None / purely technical",
                "DetailedAnalysis": [
                    "Momentum breakout over recent resistance, multi-timeframe alignment.",
                    "Elevated relative volume at/after trigger; constructive pullback behavior.",
                    "Stop under $10.22; invalidation on close back into prior range."
                ]
            }
        ],
        "LargeCap": [
            {
                "Ticker": "AA",
                "EntryPrice": 37.32,
                "TargetPrice": 56.20,
                "StopPrice": 34.10,
                "PotentialGainPercent": 50.59,
                "PrimaryCatalyst": "Sector momentum / aluminum pricing strength",
                "DetailedAnalysis": [
                    "Trend resumption after sideways base; holds key moving averages.",
                    "Improving commodity backdrop; flow supportive versus sector ETF.",
                    "Stop below base; maintain position sizing per risk."
                ]
            }
        ]
    }
    try:
        return enrich_scanner_with_realtime_prices(sample)
    except Exception:
        return sample

def basic_news_analysis(article: Dict[str, Any], symbol: Optional[str]) -> Dict[str, Any]:
    headline = (article or {}).get("headline", "")
    summary = (article or {}).get("summary", "")
    txt = f"{headline} {summary}".lower()
    stance = "neutral"
    impact = "medium"
    if any(w in txt for w in ["beats", "beat estimates", "raises guidance", "approval", "partner", "acquire", "acquisition", "buyback"]):
        stance = "bullish"
        impact = "high"
    elif any(w in txt for w in ["misses", "miss", "cut guidance", "guidance cut", "lawsuit", "probe", "sec charges", "recall", "layoff"]):
        stance = "bearish"
        impact = "high"
    elif any(w in txt for w in ["mixed", "volatile", "uncertain"]):
        stance = "mixed"
        impact = "medium"

    return {
        "stance": stance,
        "impact_level": impact,
        "summary": summary or headline,
        "rationale": [
            "Heuristic local analysis used because external AI was unavailable.",
            "Signal derived from keywords in headline/summary (approximate).",
            "Treat as a starting point only; read the full article for nuance."
        ],
        "risk_notes": "Trading around catalysts is volatile and can gap; manage size and stops.",
        "disclaimer": "This is not personalized financial advice."
    }

# -------------------------------------------------------------
#  ROUTES
# -------------------------------------------------------------
@app.route("/api/news/headlines", methods=["GET"])
@auth.login_required
def api_news_headlines():
    limit_param = request.args.get("limit", "60")
    try:
        limit = max(1, min(int(limit_param), 200))
    except ValueError:
        limit = 60

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

    return jsonify({"success": True, "data": all_items, "errors": errors})

@app.route("/api/news/analysis", methods=["POST"])
@auth.login_required
def api_news_analysis():
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"success": False, "error": "Invalid JSON body"}), 400

    article = payload.get("article") or {}
    symbol = payload.get("symbol")

    # Try Perplexity first
    if pplx_client:
        try:
            headline = article.get("headline", "")
            summary = article.get("summary", "")
            provider = article.get("provider", "") or article.get("source", "")
            url = article.get("url", "")
            tickers = article.get("tickers") or []
            tickers_str = ", ".join(tickers)

            user_prompt = (
                "Here is a single news item about one or more stocks.\n\n"
                f"Headline: {headline}\n"
                f"Source/Provider: {provider}\n"
                f"Tickers: {tickers_str}\n"
                f"Summary: {summary}\n"
                f"URL: {url}\n"
                f"Primary symbol of interest: {symbol or 'N/A'}\n\n"
                "Return ONLY the JSON object described in the system prompt."
            )

            resp = pplx_client.chat.completions.create(
                model="sonar-pro",
                messages=[
                    {"role": "system", "content": NEWS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                extra_body={"search_recency_filter": "day"},
            )
            raw = resp.choices[0].message.content
            json_str = extract_json_from_text(raw)
            data = json.loads(json_str)
            data.setdefault("stance", "neutral")
            data.setdefault("impact_level", "medium")
            data.setdefault("summary", summary or headline)
            data.setdefault("rationale", [])
            data.setdefault("risk_notes", "Trading around news involves volatility and gap risk.")
            data.setdefault("disclaimer", "This is not personalized financial advice.")
            return jsonify({"success": True, "data": data})
        except Exception as e:
            logging.error(f"News analysis (AI) error: {type(e).__name__} - {e}")

    # Fallback local analysis
    data = basic_news_analysis(article, symbol)
    return jsonify({"success": True, "data": data, "warning": "fallback_local"})

@app.route("/api/scan", methods=["POST"])
@auth.login_required
def run_scanner_api():
    """
    Scanner endpoint with graceful fallback:
    - If Perplexity fails OR sends weird bullet-list JSON, we coerce it
      or fall back to a small local example so the UI always has something.
    """
    raw = ""
    profile = "hedge_fund"
    try:
        payload = request.get_json(silent=True) or {}
        profile = (payload.get("profile") or "hedge_fund").lower()
    except Exception:
        pass

    system_prompt = HEDGE_FUND_PROMPT
    if profile == "pro_trader":
        system_prompt = PRO_TRADER_PROMPT
    elif profile == "catalyst":
        system_prompt = CATALYST_HUNTER_PROMPT

    # Try Perplexity first
    if pplx_client:
        try:
            logging.info(f"Perplexity scan start: profile={profile}")
            response = pplx_client.chat.completions.create(
                model="sonar-pro",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": (
                        "Begin scan now. For each of SmallCap, MidCap, and LargeCap, "
                        "try to return between 5 and 10 candidates that satisfy the rules in the system prompt.\n"
                        "IMPORTANT: Respond ONLY with a single JSON object:\n"
                        "{\"SmallCap\": [...], \"MidCap\": [...], \"LargeCap\": [...]}."
                    )},
                ],
                extra_body={"search_recency_filter": "day"},
            )
            raw = response.choices[0].message.content or ""
            json_string = extract_json_from_text(raw)
            try:
                data = json.loads(json_string)
            except JSONDecodeError:
                data = ast.literal_eval(json_string)

            # 🔧 NEW: coerce weird bullet-list style output into objects
            data = coerce_scanner_payload(data)

            try:
                data = enrich_scanner_with_realtime_prices(data)
            except Exception as e:
                logging.error(f"Price enrichment failed: {e}")

            return jsonify({"success": True, "data": data})
        except Exception as e:
            logging.error(f"Scanner (AI) error: {type(e).__name__} - {e}")
            logging.info(f"RAW AI OUTPUT (truncated): {repr(raw[:400])}")

    # Fallback path (no 500 to the browser)
    data = fallback_scan()
    return jsonify({"success": True, "data": data, "warning": "fallback_local"})

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
