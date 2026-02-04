# 🦅 StockScanner AI

**StockScanner** is an institutional-grade algorithmic trading backend that bridges **TradingView** technicals with **Agentic AI** reasoning. It acts as a central nervous system for trading operations, ingesting real-time market data, computing advanced technicals (GEX, Volume Profile, Supertrend), and using Large Language Models to validate high-probability setups.

## 🚀 Core Capabilities

### 🧠 Hybrid AI Decision Engine
*   **Multi-Model Routing**: 
    *   **Perplexity (Sonar-Online)**: Used for real-time news checks, SEC filing lookups, and verifying "Why is this moving?".
    *   **OpenAI (GPT-4o/Mini)**: Used for quantitative signal scoring and risk/reward validation.
*   **Context-Aware**: The AI doesn't just guess; it receives a full `TechnicalContext` object containing EMAs, MACD, Volume Splits, and Market Structure (Higher Highs/Lows).
*   **News Filtering**: Automatically triggers a "News Check" if Relative Volume (RVOL) > 2.0 or Price Change > 4%.

### 📡 TradingView Relay & Webhook Handler
*   **Pine Script Integration**: Includes custom Pine Script strategies (`pine_ai_predict_pro_daytrade.pine`) that send alerts to the backend.
*   **Real-Time Ingestion**: Captures OHLCV bars via webhooks to maintain an internal "Tape" of market action.
*   **Signal Normalization**: Standardizes alerts from various strategies into a unified `SignalRecord` format.

### 📐 Advanced Quantitative Analytics
*   **Microstructure Analysis**: Detects local swing points, Higher Highs/Lows, and trend breaks.
*   **Volume Profiling**: Calculates Buy/Sell volume pressure and Relative Volume (RVOL) spikes.
*   **Supertrend & ATR**: Dynamic volatility-based stop-loss calculation.

## 🛠️ Technology Stack

*   **Core**: Python 3.9+, Flask, Gunicorn
*   **Data Persistence**: PostgreSQL (Signals/Bars), SQLite (Lightweight Alert History)
*   **ORM**: SQLAlchemy 2.0+
*   **AI Providers**: OpenAI API, Perplexity API
*   **Market Data**: TradingView Webhooks, Yahoo Finance (Fallback)

## 📦 Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Jeyamithran/stockscanner.git
    cd stockscanner
    ```

2.  **Environment Configuration**
    Create a `.env` file:
    ```bash
    # AI Keys
    OPENAI_API_KEY=sk-...
    PPLX_API_KEY=pplx-...

    # Database
    DATABASE_URL=postgresql://user:pass@localhost:5432/stockscanner

    # Authentication
    SCANNER_USERNAME=admin
    SCANNER_PASSWORD=secure_password
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Server**
    ```bash
    # Development
    python app.py

    # Production (Gunicorn)
    gunicorn -w 4 -b 0.0.0.0:8000 app:application
    ```

## 🔄 System Architecture

1.  **Ingest**: TradingView webhook hits `/api/webhook` with bar data/alert.
2.  **Process**: `services/context.py` computes 20+ technical indicators (RSI, ADX, VWAP).
3.  **Filter**: `ai_router.py` determines if the move requires a news check (e.g., is this a pump or earnings?).
4.  **Analyze**: AI generates a `SignalRecord` with entry, stop, target, and reasoning.
5.  **Persist**: Result saved to Postgres; API serves frontend dashboards.

## 🛡️ License

MIT License.
