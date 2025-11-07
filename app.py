# app.py - Black Box Scanner Backend (Swing Trade - Always On)
from flask import Flask, jsonify, render_template, request
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import logging
from openai import APIError 
import re # Still needed for robust JSON extraction

# Configure logging
logging.basicConfig(level=logging.INFO)
load_dotenv() 

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "generate_a_strong_key_in_render")

# --- Perplexity Configuration ---
PERPLEXITY_API_KEY = os.environ.get("PPLX_API_KEY")

if not PERPLEXITY_API_KEY:
    logging.error("PPLX_API_KEY not found. Please set it securely.")
    client = None
else:
    client = OpenAI(
        api_key=PERPLEXITY_API_KEY,
        base_url="https://api.perplexity.ai" 
    )

# --- NEW/UPDATED SWING TRADE SYSTEM PROMPT ---
# Added the explicit rule for returning an empty set when no trades are found.
# app.py - SYSTEM_PROMPT (World's Best Breakout Detector)

SYSTEM_PROMPT = """
You are the lead Quantitative Analyst for a private hedge fund, responsible for generating high-alpha trading signals. Your model, based on sonar-reasoning-pro, must provide a multi-factor confirmation for every alert. This system must work 24/7.

Execute a market scan focusing on **imminent, explosive breakout potential** across all US equity listings.

**I. SCREENING AND SEGREGATION:**
1.  **Segregation:** Group results into 'SmallCap', 'MidCap', and 'LargeCap'.
2.  **Breakout Confirmation (Technical):** Identify stocks exhibiting **Volatility Contraction** (ATR near 10-day low) and confirmed **On-Balance Volume (OBV) Accumulation divergence** over the last 48 hours.

**II. ALPHA FACTOR CONFIRMATION (CRITICAL):**
3.  **Short Squeeze Fuel:** ONLY select candidates with **Days to Cover (DTC) > 5** AND **Short Interest % of Float > 15%** (to ensure built-in momentum).
4.  **Options Flow Validation:** Confirm that the stock's recent **Implied Volatility (IV) Rank** is rising AND the **Put/Call Ratio** shows an unusual skew toward calls (suggesting institutional hedging or aggressive positioning).
5.  **Sector Flow Alignment:** Verify that the stock's **Relative Strength (RS)** is outperforming its **Sector ETF** over the past 5 days (Intermarket Analysis).

**III. ACTIONABLE OUTPUT:**
6.  **Output Fields:** For each candidate, you MUST provide the following keys: Ticker, EntryPrice, TargetPrice, StopPrice, PotentialGainPercent, PrimaryCatalyst (with SEC/News source), and DetailedAnalysis.
7.  **DetailedAnalysis (3-point thesis):** Point 1: **Breakout Mechanism** (Tethering/Squeeze). Point 2: **Alpha Factor Synthesis** (Combine Short Fuel + Options Flow). Point 3: **Risk Management** (Stop-Loss and Exit Justification).

**STRICT RULE 1 (Format):** You MUST return the result ONLY as a single, valid JSON object, with the top-level keys being 'SmallCap', 'MidCap', and 'LargeCap', each containing an array of alert objects.
**STRICT RULE 2 (Off-Hours/No Trades):** If NO high-alpha candidates are found, you MUST return this empty JSON structure: {"SmallCap": [], "MidCap": [], "LargeCap": []}. DO NOT RETURN ANY OTHER TEXT OR EXPLANATION.
"""
# --- END SYSTEM PROMPT ---


# --- Regex Helper Function (Stays the same) ---
def extract_json_from_text(text):
    """Aggressively extracts the JSON object from noisy AI output."""
    match = re.search(r'\{.*\}', text.strip(), re.DOTALL)
    if match:
        return match.group(0)
    return text


@app.route('/api/scan', methods=['POST'])
def run_scanner_api():
    if not client:
        return jsonify({"success": False, "error": "API Key is missing or invalid. Cannot connect to Perplexity."}), 503
        
    raw_json_string = "" # Initialize raw string for final debug logging

    try:
        logging.info("Sending request to Perplexity API with sonar-reasoning-pro (Swing Trade Scan).")
        
        response = client.chat.completions.create(
            model="sonar-reasoning-pro",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "BEGIN SWING TRADE SCAN: Find the top 3 candidates now. Present all price points formatted to two decimal places."}
            ],
            extra_body={
                "search_recency_filter": "day", # Look for data within the last day
            }
        )
        
        raw_json_string = response.choices[0].message.content
        
        # 1. CLEAN THE OUTPUT: Extract only the JSON
        json_string = extract_json_from_text(raw_json_string)
        
        # 2. PARSE THE JSON
        data = json.loads(json_string) 
        
        return jsonify({"success": True, "data": data})

    except APIError as e:
        error_message = str(e)
        logging.error(f"Perplexity API Failure: {error_message}")
        
        if "429" in error_message or "rate limit" in error_message.lower():
             return jsonify({"success": False, "error": "Rate limit reached (429). Decrease frequency or check credits."}), 429
        return jsonify({"success": False, "error": f"API Request Error: {error_message}"}), 500

    except Exception as e:
        # This catches the JSONDecodeError when the AI breaks the output rule.
        logging.error(f"Internal Scanner Error: {type(e).__name__} - {str(e)}")
        logging.error(f"RAW AI OUTPUT RECEIVED (Critical Debug Data): ---{repr(raw_json_string)}---")
        
        # As a final fail-safe for the user, return the structure the frontend expects.
        return jsonify({
            "success": False, 
            "error": "Scanner failed. AI returned non-JSON text. Check terminal for debug output.",
            "data": {"SmallCap": [], "MidCap": [], "LargeCap": []} # Failsafe empty data
        }), 500

@app.route('/')
def dashboard_view():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))