import json
import logging
from typing import Dict, Any, Optional

from sqlalchemy.exc import SQLAlchemyError

from database import SessionLocal
from models import Signal
from services.bars import get_recent_bars
from services.context import build_technical_context
from ai_router import generate_hybrid_signal


def run_ai_signal(
    symbol: str,
    timeframe: str,
    mode: str,
    analysis_mode: str,
    breakout_event: Optional[Dict[str, Any]] = None,
) -> Optional[Signal]:
    bars = get_recent_bars(symbol, timeframe, limit=300)
    if not bars:
        return None

    raw = ""
    context: Dict[str, Any] = {}
    parsed: Dict[str, Any] = {
        "signal": "hold",
        "entry": None,
        "stop": None,
        "target": None,
        "rr_ratio": None,
        "confidence": 0,
        "time_horizon": "unknown",
        "reason_short": "Fallback: AI unavailable.",
    }
    try:
        context = build_technical_context(symbol, timeframe, mode, bars, breakout_event, analysis_mode)
        # Build router-friendly context
        price_change_pct = 0.0
        if bars:
            start = bars[0].close
            end = bars[-1].close
            if start:
                price_change_pct = ((end - start) / start) * 100.0
        rvol = context.get("volume", {}).get("relative_volume") or 0
        signal_context = {
            "ticker": symbol,
            "symbol": symbol,
            "timeframe": timeframe,
            "technical_context": context,
            "rvol": rvol,
            "price_change_pct": price_change_pct,
            "last_news_check": None,
        }
        parsed = generate_hybrid_signal(signal_context)
        raw = json.dumps(parsed, ensure_ascii=False)
    except Exception as exc:
        logging.error("AI signal generation failed for %s %s: %s", symbol, timeframe, exc)
        parsed["reason_short"] = f"Fallback: AI error {type(exc).__name__}"

    # Normalize fields from ChatGPT output to existing schema
    signal_val = (parsed.get("signal") or "hold").lower()
    if signal_val in {"long", "buy"}:
        signal_val = "buy"
    elif signal_val in {"short", "sell"}:
        signal_val = "sell"
    rr_ratio = parsed.get("rr_ratio") or parsed.get("risk_reward")
    if rr_ratio is None:
        rr_ratio = parsed.get("risk_reward")
    reason_short = parsed.get("reason_short") or parsed.get("reasoning_bullets")
    if isinstance(reason_short, list):
        reason_short = "; ".join(str(x) for x in reason_short if x)

    if SessionLocal is None:
        return None

    signal_row = Signal(
        symbol=symbol.upper(),
        timeframe=timeframe,
        mode=mode,
        analysis_mode=analysis_mode,
        signal=str(signal_val or "hold"),
        entry=parsed.get("entry"),
        stop=parsed.get("stop"),
        target=parsed.get("target"),
        rr_ratio=rr_ratio,
        confidence=parsed.get("confidence"),
        time_horizon=parsed.get("time_horizon"),
        reason_short=reason_short,
        raw_json=str(raw or parsed),
        context_json=json.dumps(context or {}, ensure_ascii=False),
    )
    try:
        with SessionLocal() as session:
            session.add(signal_row)
            session.commit()
            session.refresh(signal_row)
            return signal_row
    except SQLAlchemyError as exc:
        logging.error("Failed to persist signal for %s %s: %s", symbol, timeframe, exc)
        return None
