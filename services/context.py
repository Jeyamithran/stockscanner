from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from models import Bar


def _ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period or period <= 0:
        return None
    k = 2 / (period + 1)
    ema_val = values[0]
    for price in values[1:]:
        ema_val = price * k + ema_val * (1 - k)
    return ema_val


def _sma(values: List[float], period: int) -> Optional[float]:
    if len(values) < period or period <= 0:
        return None
    return sum(values[-period:]) / period


def _rsi(values: List[float], period: int = 14) -> Optional[float]:
    if len(values) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(1, period + 1):
        delta = values[-i] - values[-i - 1]
        if delta >= 0:
            gains.append(delta)
        else:
            losses.append(abs(delta))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period if losses else 0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss if avg_loss else 0
    return 100 - (100 / (1 + rs))


def _atr(bars: List[Bar], period: int = 14) -> Optional[float]:
    if len(bars) < period + 1:
        return None
    trs: List[float] = []
    for idx in range(1, len(bars)):
        curr = bars[idx]
        prev = bars[idx - 1]
        tr = max(
            curr.high - curr.low,
            abs(curr.high - prev.close),
            abs(curr.low - prev.close),
        )
        trs.append(tr)
    if len(trs) < period:
        return None
    return sum(trs[-period:]) / period


def _macd(values: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, Optional[float]]:
    if len(values) < slow + signal:
        return {"macd": None, "signal": None, "hist": None}
    macd_line = (_ema(values, fast) or 0) - (_ema(values, slow) or 0)
    signal_line = _ema(values, signal) if macd_line is not None else None
    hist = macd_line - signal_line if macd_line is not None and signal_line is not None else None
    return {"macd": macd_line, "signal": signal_line, "hist": hist}


def _relative_volume(volumes: List[float], window: int = 20) -> Optional[float]:
    if len(volumes) < window or window <= 0:
        return None
    avg = sum(volumes[-window:]) / window
    if not avg:
        return None
    return volumes[-1] / avg


def _structure(bars: List[Bar]) -> Dict[str, Any]:
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    return {
        "day_high": max(highs) if highs else None,
        "day_low": min(lows) if lows else None,
        "recent_support_levels": lows[-5:] if len(lows) >= 1 else [],
        "recent_resistance_levels": highs[-5:] if len(highs) >= 1 else [],
    }


def build_technical_context(
    symbol: str,
    timeframe: str,
    mode: str,
    bars: List[Bar],
    breakout_event: Optional[Dict[str, Any]],
    analysis_mode: Optional[str] = None,
) -> Dict[str, Any]:
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [b.volume or 0.0 for b in bars]
    ema_9 = _ema(closes, 9)
    ema_21 = _ema(closes, 21)
    ema_50 = _ema(closes, 50)
    ema_200 = _ema(closes, 200)
    rsi_14 = _rsi(closes, 14)
    atr_14 = _atr(bars, 14)
    rv = _relative_volume(volumes, 20)
    macd_vals = _macd(closes)

    context = {
        "symbol": symbol,
        "timeframe": timeframe,
        "mode": mode,
        "analysis_mode": analysis_mode,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "recent_bars": [
            {
                "time": b.time.isoformat(),
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            }
            for b in bars[-200:]
        ],
        "indicators": {
            "ema": {"ema_9": ema_9, "ema_21": ema_21, "ema_50": ema_50, "ema_200": ema_200},
            "rsi": {"rsi_14": rsi_14},
            "macd": macd_vals,
            "atr": {"atr_14": atr_14},
        },
        "volume": {
            "last_volume": volumes[-1] if volumes else None,
            "relative_volume": rv,
            "volume_spike": rv is not None and rv > 1.5,
        },
        "structure": _structure(bars),
        "session": {},
        "event_flags": {},
        "breakout_context": breakout_event or {},
        "risk_settings": {"account_risk_percent": 1.0, "min_rr_for_buy": 2.0},
    }
    if breakout_event:
        event = breakout_event.get("event")
        context["event_flags"] = {
            "breakout_long": event == "breakout_long",
            "breakout_short": event == "breakout_short",
        }
    return context
