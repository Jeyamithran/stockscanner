import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

import requests
from openai import OpenAI

from config import settings


def _require_keys():
    if not settings.pplx_api_key:
        raise RuntimeError("PPLX_API_KEY is missing.")
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing.")


def should_check_news(ctx: Dict[str, Any]) -> bool:
    rvol = float(ctx.get("rvol") or 0)
    pct = abs(float(ctx.get("price_change_pct") or 0))
    last_check = ctx.get("last_news_check")
    if rvol >= 2.0 or pct >= 4:
        return True
    if not last_check:
        return True
    try:
        ts = datetime.fromisoformat(last_check.replace("Z", "+00:00"))
        return datetime.now(timezone.utc) - ts > timedelta(hours=4)
    except Exception:
        return True


def get_perplexity_news(ticker: str) -> str:
    _require_keys()
    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {settings.pplx_api_key}"}
    system_prompt = (
        "You are a concise news scanner. Return bullet points (<=6) covering last 3-5 days catalysts: "
        "earnings, guidance, upgrades/downgrades, SEC/FDA/regulatory, macro directly impacting this ticker. "
        "Max 25 words per bullet. No trading advice."
    )
    user_prompt = f"Ticker: {ticker}. Provide short, factual bullets."
    body = {
        "model": "sonar-small-online",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=15)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return content or ""
    except Exception as exc:
        logging.error("Perplexity news fetch failed for %s: %s", ticker, exc)
        return ""


def get_chatgpt_signal(
    ticker: str,
    timeframe: str,
    technical_context: Dict[str, Any],
    news_summary: Optional[str] = None,
) -> Dict[str, Any]:
    _require_keys()
    client = OpenAI(api_key=settings.openai_api_key)
    system_prompt = (
        "You are a quantitative trading signal engine. Use the technical data and optional news summary. "
        "Return ONLY valid JSON with keys: "
        '{"ticker","timeframe","signal","entry","stop","target","risk_reward","confidence",'
        '"reasoning_bullets","news_influence"}. Signals: LONG/SHORT/HOLD. '
        "confidence 0-1. Do NOT include text outside JSON."
    )
    user_prompt = (
        f"TICKER: {ticker}\nTIMEFRAME: {timeframe}\n"
        f"TECHNICAL_CONTEXT: {json.dumps(technical_context, ensure_ascii=False)}\n"
        f"NEWS_SUMMARY: {news_summary or ''}"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content or ""
        data = json.loads(raw)
        data.setdefault("model", "gpt-4.1-mini")
        return data
    except Exception as exc:
        logging.error("ChatGPT signal failed for %s %s: %s", ticker, timeframe, exc)
        return {
            "ticker": ticker,
            "timeframe": timeframe,
            "signal": "HOLD",
            "entry": None,
            "stop": None,
            "target": None,
            "risk_reward": None,
            "confidence": 0,
            "reasoning_bullets": ["AI error encountered."],
            "news_influence": "NONE",
            "model": "gpt-4.1-mini",
            "error_message": str(exc),
        }


def generate_hybrid_signal(signal_context: Dict[str, Any]) -> Dict[str, Any]:
    ticker = signal_context.get("ticker") or signal_context.get("symbol") or ""
    timeframe = signal_context.get("timeframe") or ""
    technical_context = signal_context.get("technical_context") or {}

    news_summary = ""
    news_model = None
    try:
        if should_check_news(signal_context):
            news_summary = get_perplexity_news(ticker)
            news_model = "sonar-small-online"
    except Exception as exc:
        logging.error("News routing failed for %s: %s", ticker, exc)
        news_summary = ""

    result = get_chatgpt_signal(ticker, timeframe, technical_context, news_summary)
    result.setdefault("model", "gpt-4.1-mini")
    if news_model:
        result.setdefault("news_model", news_model)
    return result
