import json
import logging
from typing import Dict

from openai import OpenAI

from config import settings

TECH_SYSTEM_PROMPT = (
    "You are a professional trading analyst. Return ONLY JSON with keys: "
    "signal, entry, stop, target, rr_ratio, confidence, time_horizon, reason_short."
)

TECH_USER_TEMPLATE = (
    "Trading mode: {mode}. Analysis mode: technical-only (no news).\n"
    "Here is the structured context JSON:\n{context_json}\n"
    "Decide strong_buy/buy/hold/sell/strong_sell. Provide entry/stop/target with RR >= 2.0 for buys. "
    "Return ONLY JSON."
)

NEWS_USER_TEMPLATE = (
    "Trading mode: {mode}. Analysis mode: news_tech (check latest news + combine with technicals).\n"
    "Here is the structured context JSON:\n{context_json}\n"
    "Decide strong_buy/buy/hold/sell/strong_sell. Provide entry/stop/target with RR >= 2.0 for buys. "
    "Return ONLY JSON."
)


client = None
if settings.pplx_api_key:
    client = OpenAI(api_key=settings.pplx_api_key, base_url="https://api.perplexity.ai")
else:
    logging.warning("PPLX_API_KEY not set; Perplexity calls will be disabled.")


def call_perplexity(context: Dict, mode: str, analysis_mode: str) -> str:
    if client is None:
        raise RuntimeError("Perplexity client is not configured.")

    context_json = json.dumps(context, ensure_ascii=False)
    if analysis_mode == "news_tech":
        user_prompt = NEWS_USER_TEMPLATE.format(mode=mode, context_json=context_json)
        model = settings.pplx_model_news_tech
    else:
        user_prompt = TECH_USER_TEMPLATE.format(mode=mode, context_json=context_json)
        model = settings.pplx_model_technical

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": TECH_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


def _extract_json_from_text(text: str) -> str:
    """
    Grab the first JSON object from free-form text (e.g., markdown fences).
    Falls back to raw text if braces cannot be matched.
    """
    if not text:
        return ""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def parse_ai_response(raw_content: str) -> Dict:
    try:
        return json.loads(_extract_json_from_text(raw_content))
    except Exception as exc:
        logging.error("Failed to parse Perplexity response; defaulting to hold. Error: %s; Raw: %s", exc, (raw_content or "")[:500])
        return {
            "signal": "hold",
            "entry": None,
            "stop": None,
            "target": None,
            "rr_ratio": None,
            "confidence": 0,
            "time_horizon": "unknown",
            "reason_short": "AI response could not be parsed; defaulting to hold.",
        }
