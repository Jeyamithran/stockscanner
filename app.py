# app.py - Black Box Scanner Backend (FINAL UNIFIED VERSION)
from flask import Flask, jsonify, render_template, request
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import logging
from openai import APIError 
import re 
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

# Configure logging
logging.basicConfig(level=logging.INFO)
load_dotenv() 

app = Flask(__name__)
# Secret key is needed for session management/security
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "generate_a_strong_key_in_render")

# --- AUTHENTICATION SETUP ---
auth = HTTPBasicAuth()

# CRITICAL FIX: Hash the password using a guaranteed available method (PBKDF2)
USER_DATA = {
    "username": os.environ.get("SCANNER_USERNAME", "admin"),
    "password_hash": generate_password_hash(
        os.environ.get("SCANNER_PASSWORD", "default_password"),
        method="pbkdf2:sha256:260000" # Force PBKDF2:SHA256 hash method
    ) 
}

@auth.verify_password
def verify_password(username, password):
    """Checks the submitted username/password against the stored hash."""
    if username == USER_DATA["username"]:
        if check_password_hash(USER_DATA["password_hash"], password):
            return username
    return None
# --- END AUTH SETUP ---


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

# --- SYSTEM PROMPT (World's Best Breakout Detector) ---
SYSTEM_PROMPT = """
You are a highly reliable Swing Trader Analyst. Your analysis must work 24/7, based on end-of-day or recent (24-hour) catalyst data. Your goal is to identify stocks primed for a multi-day momentum swing.

Execute a market scan focusing on imminent, explosive breakout potential across all US equity listings.

**I. SCREENING AND SEGREGATION:**
1.  **Segregation:** Group results into 'SmallCap', 'MidCap', and 'LargeCap'.
2.  **Breakout Confirmation (Technical):** Identify stocks exhibiting **Volatility Contraction (ATR near 10-day low)** AND confirmed **On-Balance Volume (OBV) Accumulation divergence** over the last 48 hours.

**II. ALPHA FACTOR CONFIRMATION (RELAXED):**
3.  **Short Squeeze Fuel (Requirement Relaxed):** Identify stocks where **Short Interest % of Float is > 10%** OR **Days to Cover (DTC) is > 3**. (Allows for more setups).
4.  **Sector Flow Alignment:** Verify that the stock's **Relative Strength (RS)** is outperforming its **Sector ETF** over the past 5 days (Intermarket Analysis).
5.  **Catalyst Validation:** Prioritize stocks with an **SEC Filing (Form 4/8-K)** or **Major News** in the last 48 hours.

**III. ACTIONABLE OUTPUT:**
6.  **Output Fields:** For each candidate, you MUST provide the following keys: Ticker, EntryPrice, TargetPrice, StopPrice, PotentialGainPercent, PrimaryCatalyst (with SEC/News source), and DetailedAnalysis.
7.  **DetailedAnalysis (3-point thesis):** Point 1: **Breakout Mechanism** (Tethering/Squeeze). Point 2: **Alpha Factor Synthesis** (Combine Short Fuel + Volume Flow). Point 3: **Risk Management** (Stop-Loss and Exit Justification).

**STRICT RULE 1 (Format):** You MUST return the result ONLY as a single, valid JSON object, with the top-level keys being 'SmallCap', 'MidCap', and 'LargeCap', each containing an array of alert objects.
**STRICT RULE 2 (No Trades):** If NO candidates are found, you MUST return this empty JSON structure: {"SmallCap": [], "MidCap": [], "LargeCap": []}. DO NOT RETURN ANY OTHER TEXT OR EXPLANATION.
"""
# --- END SYSTEM PROMPT ---


# --- Regex Helper Function (Final Working Version) ---
def extract_json_from_text(text):
    """Aggressively extracts the JSON object from noisy AI output."""
    # 1. Strip preambles (like <think> tags or markdown fences)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'```json\s*', '', text, flags=re.DOTALL)
    text = text.replace('```', '')

    # 2. Aggressively find the first opening brace and the last closing brace
    match = re.search(r'\{.*\}', text.strip(), re.DOTALL)
    
    if match:
        return match.group(0)
    
    return text


# --- PROTECTED ROUTES ---
@app.route('/api/scan', methods=['POST'])
@auth.login_required # PROTECT THE API ENDPOINT
def run_scanner_api():
    if not client:
        return jsonify({"success": False, "error": "API Key is missing or invalid. Cannot connect to Perplexity."}), 503
        
    raw_json_string = "" 
    
    try:
        logging.info("Sending request to Perplexity API with sonar-reasoning-pro (Swing Trade Scan).")
        
        response = client.chat.completions.create(
            model="sonar-reasoning-pro",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                # CHANGE MADE HERE: Requesting up to 10 candidates per tier
                {"role": "user", "content": "BEGIN SWING TRADE SCAN: Find up to 10 candidates for each market cap tier now. Present all price points formatted to two decimal places."}
            ],
            extra_body={
                "search_recency_filter": "day",
            }
        )
        
        raw_json_string = response.choices[0].message.content
        json_string = extract_json_from_text(raw_json_string)
        data = json.loads(json_string) 
        
        return jsonify({"success": True, "data": data})

    except APIError as e:
        error_message = str(e)
        logging.error(f"Perplexity API Failure: {error_message}")
        if "429" in error_message or "rate limit" in error_message.lower():
             return jsonify({"success": False, "error": "Rate limit reached (429). Decrease frequency or check credits."}), 429
        return jsonify({"success": False, "error": f"API Request Error: {error_message}"}), 500

    except Exception as e:
        # Catches JSONDecodeError
        logging.error(f"Internal Scanner Error: {type(e).__name__} - {str(e)}")
        logging.error(f"RAW AI OUTPUT RECEIVED (Critical Debug Data): ---{repr(raw_json_string)}---")
        
        return jsonify({
            "success": False, 
            "error": "Scanner failed. AI returned non-JSON text. Check terminal for debug output.",
            "data": {"SmallCap": [], "MidCap": [], "LargeCap": []} 
        }), 500


@app.route('/')
@auth.login_required # PROTECT THE MAIN DASHBOARD
def dashboard_view():
    return render_template('index.html')


@app.errorhandler(401)
def unauthorized(error):
    return 'Login Required: Please enter your credentials to access the Black Box Scanner.', 401


if __name__ == '__main__':
    # NOTE: You MUST have your environment variables set for this to run locally or on Render.
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))