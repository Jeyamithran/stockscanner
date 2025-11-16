from typing import Dict, Any, Optional

from sqlalchemy.exc import SQLAlchemyError

from database import SessionLocal
from models import Signal
from services.bars import get_recent_bars
from services.context import build_technical_context
from services.perplexity_client import call_perplexity, parse_ai_response


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

    context = build_technical_context(symbol, timeframe, mode, bars, breakout_event, analysis_mode)
    raw = call_perplexity(context, mode, analysis_mode)
    parsed = parse_ai_response(raw)

    if SessionLocal is None:
        return None

    signal_row = Signal(
        symbol=symbol.upper(),
        timeframe=timeframe,
        mode=mode,
        analysis_mode=analysis_mode,
        signal=str(parsed.get("signal") or "hold"),
        entry=parsed.get("entry"),
        stop=parsed.get("stop"),
        target=parsed.get("target"),
        rr_ratio=parsed.get("rr_ratio"),
        confidence=parsed.get("confidence"),
        time_horizon=parsed.get("time_horizon"),
        reason_short=parsed.get("reason_short"),
        raw_json=str(parsed),
    )
    try:
        with SessionLocal() as session:
            session.add(signal_row)
            session.commit()
            session.refresh(signal_row)
            return signal_row
    except SQLAlchemyError:
        return None
