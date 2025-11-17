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


def _volume_split(bars: List[Bar], window: int = 20) -> Dict[str, Any]:
    """Compute simple buy/sell volume proxies based on candle direction for last N bars."""
    if not bars:
        return {
            "buy_volume": None,
            "sell_volume": None,
            "buy_volume_pct": None,
            "sell_volume_pct": None,
            "buy_sell_ratio": None,
            "dominant_side": None,
        }
    window = min(window, len(bars))
    total_buy = 0.0
    total_sell = 0.0
    for b in bars[-window:]:
        if b.close > b.open:
            total_buy += b.volume or 0.0
        elif b.close < b.open:
            total_sell += b.volume or 0.0
        else:
            # Flat candle: do not count toward either
            pass
    total_vol = total_buy + total_sell
    buy_pct = (total_buy / total_vol * 100.0) if total_vol else None
    sell_pct = (total_sell / total_vol * 100.0) if total_vol else None
    ratio = (total_buy / total_sell) if total_sell else (total_buy if total_buy else None)
    dominant = None
    if total_buy > total_sell:
        dominant = "buy"
    elif total_sell > total_buy:
        dominant = "sell"
    else:
        dominant = "mixed"
    return {
        "buy_volume": total_buy if total_vol else None,
        "sell_volume": total_sell if total_vol else None,
        "buy_volume_pct": buy_pct,
        "sell_volume_pct": sell_pct,
        "buy_sell_ratio": ratio,
        "dominant_side": dominant,
    }


def _supertrend(bars: List[Bar], period: int = 10, multiplier: float = 3.0) -> Dict[str, Any]:
    if len(bars) < period + 1:
        return {}
    hl2_vals = [(b.high + b.low) / 2 for b in bars]
    atr_val = _atr(bars, period)
    if atr_val is None:
        return {}

    def _compute(idx: int, prev_upper: float, prev_lower: float, prev_trend: str) -> Dict[str, Any]:
        basic_upper = hl2_vals[idx] + multiplier * atr_val
        basic_lower = hl2_vals[idx] - multiplier * atr_val
        upper = basic_upper if (basic_upper < prev_upper or bars[idx - 1].close > prev_upper) else prev_upper
        lower = basic_lower if (basic_lower > prev_lower or bars[idx - 1].close < prev_lower) else prev_lower
        trend = prev_trend
        if bars[idx].close > prev_upper:
            trend = "up"
        elif bars[idx].close < prev_lower:
            trend = "down"
        return {"trend": trend, "upper": upper, "lower": lower}

    init_idx = len(bars) - (period + 1)
    init_basic_upper = hl2_vals[init_idx] + multiplier * atr_val
    init_basic_lower = hl2_vals[init_idx] - multiplier * atr_val
    state = {"trend": "up", "upper": init_basic_upper, "lower": init_basic_lower}
    trend_history: List[str] = []
    for i in range(init_idx + 1, len(bars)):
        state = _compute(i, state["upper"], state["lower"], state["trend"])
        trend_history.append(state["trend"])

    flips_last_50 = 0
    for i in range(1, len(trend_history)):
        if trend_history[i] != trend_history[i - 1] and (init_idx + 1 + i) >= len(bars) - 50:
            flips_last_50 += 1

    bars_since_flip = 0
    for i in range(len(trend_history) - 1, 0, -1):
        if trend_history[i] != trend_history[i - 1]:
            bars_since_flip = len(trend_history) - i
            break

    return {
        "trend": state["trend"],
        "line": state["lower"] if state["trend"] == "up" else state["upper"],
        "bars_since_flip": bars_since_flip,
        "flips_last_50_bars": flips_last_50,
        "atr": atr_val,
    }


def _microstructure(bars: List[Bar]) -> Dict[str, Any]:
    if len(bars) < 4:
        return {
            "higher_highs": None,
            "higher_lows": None,
            "lower_highs": None,
            "lower_lows": None,
            "bars_since_last_swing_high": None,
            "bars_since_last_swing_low": None,
        }
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    higher_highs = all(x > y for x, y in zip(highs[-3:], highs[-4:-1]))
    higher_lows = all(x > y for x, y in zip(lows[-3:], lows[-4:-1]))
    lower_highs = all(x < y for x, y in zip(highs[-3:], highs[-4:-1]))
    lower_lows = all(x < y for x, y in zip(lows[-3:], lows[-4:-1]))

    last_swing_high = None
    last_swing_low = None
    for idx in range(len(bars) - 2, 1, -1):
        if highs[idx] > highs[idx - 1] and highs[idx] > highs[idx + 1]:
            last_swing_high = idx
            break
    for idx in range(len(bars) - 2, 1, -1):
        if lows[idx] < lows[idx - 1] and lows[idx] < lows[idx + 1]:
            last_swing_low = idx
            break
    bars_since_high = len(bars) - last_swing_high - 1 if last_swing_high is not None else None
    bars_since_low = len(bars) - last_swing_low - 1 if last_swing_low is not None else None

    return {
        "higher_highs": higher_highs,
        "higher_lows": higher_lows,
        "lower_highs": lower_highs,
        "lower_lows": lower_lows,
        "bars_since_last_swing_high": bars_since_high,
        "bars_since_last_swing_low": bars_since_low,
    }


def _structure(bars: List[Bar]) -> Dict[str, Any]:
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    if not highs or not lows:
        return {
            "day_high": None,
            "day_low": None,
            "recent_support_levels": [],
            "recent_resistance_levels": [],
        }
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
    supertrend = _supertrend(bars)
    micro = _microstructure(bars)
    structure = _structure(bars)
    vol_split = _volume_split(bars)
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
            "supertrend": supertrend or None,
        },
        "volume": {
            "last_volume": volumes[-1] if volumes else None,
            "relative_volume": rv,
            "volume_spike": rv is not None and rv > 1.5,
            "buy_volume": vol_split.get("buy_volume"),
            "sell_volume": vol_split.get("sell_volume"),
            "buy_volume_pct": vol_split.get("buy_volume_pct"),
            "sell_volume_pct": vol_split.get("sell_volume_pct"),
            "buy_sell_ratio": vol_split.get("buy_sell_ratio"),
            "dominant_side": vol_split.get("dominant_side"),
        },
        "structure": structure,
        "structural_levels": {
            "recent_support": min(structure.get("recent_support_levels") or [None]),
            "recent_resistance": max(structure.get("recent_resistance_levels") or [None]),
            "day_low": structure.get("day_low"),
            "day_high": structure.get("day_high"),
            "distance_to_day_low": (closes[-1] - structure["day_low"]) if structure.get("day_low") is not None else None,
            "distance_to_day_high": (structure["day_high"] - closes[-1]) if structure.get("day_high") is not None else None,
            "just_broke_support": structure.get("day_low") is not None and closes and closes[-1] < structure["day_low"],
        },
        "microstructure": micro,
        "position": {"side": "flat"},
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
