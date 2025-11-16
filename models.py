from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Index,
    BigInteger,
    Date,
    ForeignKey,
)
from sqlalchemy.orm import relationship

from database import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Bar(Base):
    __tablename__ = "bars"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(32), nullable=False)
    timeframe = Column(String(24), nullable=False)
    time = Column(DateTime(timezone=True), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float)

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "time", name="uix_bars_symbol_tf_time"),
        Index("ix_bars_symbol", "symbol"),
        Index("ix_bars_symbol_tf", "symbol", "timeframe"),
    )


class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(32), nullable=False)
    timeframe = Column(String(24), nullable=False)
    mode = Column(String(16), nullable=True)
    analysis_mode = Column(String(24), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utc_now, nullable=False, index=True)
    signal = Column(String(24), nullable=False)
    entry = Column(Float)
    stop = Column(Float)
    target = Column(Float)
    rr_ratio = Column(Float)
    confidence = Column(Float)
    time_horizon = Column(String(32))
    reason_short = Column(Text)
    raw_json = Column(Text)
    context_json = Column(Text)

    __table_args__ = (
        Index("ix_signals_symbol_tf", "symbol", "timeframe"),
        Index("ix_signals_symbol_mode", "symbol", "mode"),
    )


class AlertHistory(Base):
    __tablename__ = "alert_history"

    id = Column(Integer, primary_key=True)
    timestamp_iso = Column(String, nullable=False, index=True)
    date = Column(String)
    time = Column(String)
    profile = Column(String)
    tier = Column(String)
    ticker = Column(String, index=True)
    entry = Column(Float)
    target = Column(Float)
    ai_target_price = Column(Float)
    stop = Column(Float)
    scan_price = Column(Float)
    potential_gain_pct = Column(Float)
    ai_potential_gain_pct = Column(Float)
    primary_catalyst = Column(Text)
    detailed_analysis = Column(Text)
    model_used = Column(String)


class HighGrowthCandidate(Base):
    __tablename__ = "high_growth_candidates"

    id = Column(Integer, primary_key=True)
    timestamp_iso = Column(String, nullable=False, index=True)
    ticker = Column(String(32), nullable=False, index=True)
    company_name = Column(String)
    sector = Column(String)
    market_cap = Column(String)
    growth_metric = Column(String)
    growth_period = Column(String)
    catalyst = Column(Text)
    data_source = Column(String)
    institutional_ownership = Column(String)
    risk_reward = Column(String)
    technical_indicators = Column(Text)
    notes = Column(Text)
    raw_json = Column(Text)


class TradingviewEvent(Base):
    __tablename__ = "tv_events"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(32), nullable=False, index=True)
    timeframe = Column(String(24), nullable=False, index=True)
    bar_time = Column(BigInteger)
    received_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now, index=True)
    payload_json = Column(Text)


class BreakoutWatchlist(Base):
    __tablename__ = "breakout_watchlist"

    id = Column(Integer, primary_key=True)
    as_of_date = Column(Date, nullable=False, index=True)
    symbol = Column(String(32), nullable=False, index=True)
    timeframe = Column(String(24), nullable=False, default="D")
    mode = Column(String(16))
    direction = Column(String(16), nullable=False, default="watch")
    trigger_level = Column(Float)
    stop_hint = Column(Float)
    target_hint = Column(Float)
    structure = Column(String(32))
    volume_vs_avg = Column(Float)
    notes = Column(Text)
    tags = Column(Text)
    raw_json = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now, index=True)

    __table_args__ = (
        UniqueConstraint("as_of_date", "symbol", "timeframe", name="uix_breakout_watchlist_day_symbol_tf"),
    )


class BreakoutEvent(Base):
    __tablename__ = "breakout_events"

    id = Column(Integer, primary_key=True)
    source = Column(String(32), nullable=False, default="breakout_live", index=True)
    event = Column(String(32), nullable=False, index=True)
    symbol = Column(String(32), nullable=False, index=True)
    timeframe = Column(String(24), nullable=False, index=True)
    mode = Column(String(16), index=True)
    bar_time = Column(DateTime(timezone=True), nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    break_level = Column(Float)
    trigger_price = Column(Float)
    lookback_bars = Column(Integer)
    structure = Column(String(32))
    atr_14 = Column(Float)
    atr_mult_stop = Column(Float)
    relative_volume = Column(Float)
    session_label = Column(String(16))
    confidence_hint = Column(Float)
    notes = Column(Text)
    raw_json = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now, index=True)
    signal_id = Column(Integer, ForeignKey("signals.id"), index=True)

    signal = relationship("Signal", backref="breakout_events")
