# TradingView Relay + Perplexity Signals

This repo now includes a relay so the AI watches *exactly* the candles that close on your TradingView chart. Pine sends the bar → Flask caches it → Perplexity reasons on the latest stack of bars and replies with a structured BUY/SELL/HOLD packet.

## 1. Environment
Set the usual scanner variables plus the TradingView knobs:

| Variable | Purpose |
| --- | --- |
| `PPLX_API_KEY` | Required for all Perplexity calls. |
| `TRADINGVIEW_WEBHOOK_TOKEN` | Shared secret that Pine includes in the alert payload. Requests without it are rejected. |
| `TRADINGVIEW_MAX_BARS` | Max bars cached per symbol/timeframe (default `320`). |
| `TRADINGVIEW_MIN_BARS` | Minimum bars before the AI will run (default `40`). |
| `TRADINGVIEW_PROMPT_BAR_LIMIT` | How many of the freshest bars are sent to Perplexity (default `120`). |
| `TRADINGVIEW_SIGNAL_MODEL` | Model alias/id, e.g. `reasoning` or `sonar-pro`. |
| `TRADINGVIEW_SIGNAL_MAX_WORKERS` | Async workers for AI calls (default `2`). |
| `TRADINGVIEW_SIGNAL_COOLDOWN_SECONDS` | Cooldown per stream to avoid spamming the model (default `60`). |
| `DATABASE_URL` | Postgres connection string (e.g., `postgresql+psycopg2://user:pass@host:5432/db`). When set, bars and AI signals are persisted. |

## 2. Pine “data relay” indicator
Create an indicator that emits JSON whenever the bar closes. Include your webhook token inside the JSON body so Flask can authenticate it.

```pinescript
//@version=5
indicator("AI Relay", overlay=false)
tf = input.timeframe("15", "AI Timeframe")
token = input.string("changeme", "Webhook Token")

isNewBar = ta.change(time(tf))
if isNewBar
    json = str.format(
         '{{"symbol":"{0}","tf":"{1}","time":{2},"open":{3},"high":{4},"low":{5},"close":{6},"volume":{7},"token":"{8}","auto_signal":true}}',
         syminfo.ticker,
         tf,
         time,
         open,
         high,
         low,
         close,
         volume,
         token
    )
    alert(json, alert.freq_once_per_bar_close)
```

Hook it up inside TradingView → Alerts → select the indicator → “Any alert() function call” → paste your Flask endpoint such as `https://app.example.com/api/tradingview/webhook` into *Webhook URL*.

## 3. Server endpoints

### POST `/api/tradingview/webhook`
*No auth header required* (TradingView can’t send it); the shared token lives in the JSON body. Required fields: `symbol`, `tf`, `time`, `open`, `high`, `low`, `close`. Optional: `volume`, `auto_signal` (defaults `true`), `force_signal` (bypasses cooldown).

Response example:
```json
{
  "success": true,
  "data": {
    "symbol": "SPY",
    "timeframe": "15",
    "bars_cached": 145,
    "queued_signal": true,
    "queue_message": null,
    "signal_inflight": false,
    "cooldown_remaining": 0.0
  }
}
```

### GET `/api/tradingview/signals`
Requires scanner auth. With `symbol` + `tf` query params it returns the last N bars plus the latest AI packet. Without params it lists every active stream.

```bash
curl -u admin:password \
  "https://app.example.com/api/tradingview/signals?symbol=SPY&tf=15&limit=80"
```

> You can trigger a fresh AI read manually with `POST /api/tradingview/webhook` and `{"force_signal": true}` in the payload if you stored enough bars already.

## 4. Signal payload
Perplexity answers with a strict JSON object (see `TRADINGVIEW_SIGNAL_RESPONSE_SCHEMA`). Fields include `signal`, `entry`, `stop`, `target`, `risk_reward`, `confidence`, 3× context bullets, invalidations, and a suggested holding window. Every response is cached so the dashboard (or Telegram, etc.) can display it immediately.

## 5. Ops notes
 - Bars and signals are cached in-memory; when `DATABASE_URL` is set they are also upserted into Postgres (`bars`/`signals`) and served as fallback if the cache is empty.
- Cooldown + inflight guards prevent duplicate model calls when TradingView fires multiple alerts quickly.
- Adjust the Pine timeframe and the `TRADINGVIEW_*` env knobs per desk (scalp vs swing) without touching Python code.
- The rest of the scanner still works (Finnhub/YFinance fallbacks, catalyst scans, etc.), so you can blend the TV signal inside the existing UI or route it to Telegram.

## 6. Dashboard tab
Launch the scanner UI and click the new **TradingView Signals** tab to see every active webhook stream, refresh the cache, and inspect the latest AI packet (entry/stop/target/context) without leaving the dashboard.

## 7. Rich day-trade payloads (optional)
You can send a full snapshot instead of one bar at a time. POST a payload like:

```json
{
  "symbol": "AAPL",
  "tf": "5",
  "bar_time": 1731604800000,
  "bars": [
    { "time": 1731601200000, "open": 194.1, "high": 194.5, "low": 193.9, "close": 194.3, "volume": 1500000 },
    { "time": 1731601500000, "open": 194.3, "high": 195.1, "low": 194.0, "close": 194.9, "volume": 1710000 }
  ],
  "trend": { "ema_fast": 195.2, "ema_mid": 194.6, "ema_slow": 192.4, "trend_regime": "up" },
  "momentum": { "rsi_14": 63.8, "macd": 1.12, "macd_signal": 0.94, "macd_hist": 0.18 },
  "volatility": { "atr_14": 1.7, "atr_percent_of_price": 0.0087 },
  "volume_info": { "relative_volume": 1.4, "vol_spike_flag": true },
  "structure": { "recent_support_levels": [192.5, 190.2], "recent_resistance_levels": [196.2, 197.8], "pattern": "higher_highs_higher_lows" },
  "session": { "session_type": "regular", "time_of_day": "mid_session" },
  "risk_settings": { "account_risk_percent": 1.0, "min_rr_ratio": 2.0 },
  "token": "tradingviewtknjeya2028",
  "auto_signal": true
}
```

The webhook will replace the cached bars for that `symbol+tf`, store the indicator blocks, and immediately queue Perplexity (respecting `TRADINGVIEW_MIN_BARS`). Send separate alerts for each timeframe (e.g., 5m/15m/1h) if you want stacked context.

That’s the complete TradingView → Flask → Perplexity loop; drop it into your infra and you’ll have AI that sees the *exact* candles you do.
