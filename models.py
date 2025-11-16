from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, UniqueConstraint, Index

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
