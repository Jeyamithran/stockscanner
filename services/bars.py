from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from sqlalchemy.exc import SQLAlchemyError

from database import SessionLocal
from models import Bar


def _coerce_time(raw_time: Any) -> Optional[datetime]:
    if raw_time is None:
        return None
    if isinstance(raw_time, datetime):
        return raw_time if raw_time.tzinfo else raw_time.replace(tzinfo=timezone.utc)
    try:
        ts = int(float(raw_time))
    except (TypeError, ValueError):
        return None
    if ts > 1_000_000_000_000:  # ms -> seconds
        ts = int(ts / 1000)
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        return None


def save_bar_from_payload(payload: Dict[str, Any]) -> Optional[Bar]:
    if SessionLocal is None:
        return None
    symbol = (payload.get("symbol") or "").strip().upper()
    timeframe = str(payload.get("timeframe") or payload.get("tf") or "").strip()
    ts = payload.get("time") or payload.get("timestamp") or payload.get("bar_time")

    ohlc = payload.get("ohlc") or payload
    open_val = ohlc.get("open")
    high_val = ohlc.get("high")
    low_val = ohlc.get("low")
    close_val = ohlc.get("close")
    volume_val = ohlc.get("volume") or 0.0

    dt = _coerce_time(ts)
    if not symbol or not timeframe or dt is None:
        return None
    try:
        open_f = float(open_val)
        high_f = float(high_val)
        low_f = float(low_val)
        close_f = float(close_val)
        vol_f = float(volume_val)
    except (TypeError, ValueError):
        return None

    bar = Bar(
        symbol=symbol,
        timeframe=timeframe,
        time=dt,
        open=open_f,
        high=high_f,
        low=low_f,
        close=close_f,
        volume=vol_f,
    )
    try:
        with SessionLocal() as session:
            session.merge(bar)  # merge to respect unique constraint
            session.commit()
            session.refresh(bar)
            return bar
    except SQLAlchemyError:
        return None


def get_recent_bars(symbol: str, timeframe: str, limit: int = 300) -> List[Bar]:
    if SessionLocal is None:
        return []
    try:
        with SessionLocal() as session:
            rows = (
                session.query(Bar)
                .filter(Bar.symbol == symbol.upper(), Bar.timeframe == timeframe)
                .order_by(Bar.time.desc())
                .limit(limit)
                .all()
            )
            return list(reversed(rows))
    except SQLAlchemyError:
        return []
