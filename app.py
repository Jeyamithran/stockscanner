# app.py - Black Box Scanner Backend (With Login Gate)
from flask import Flask, jsonify, render_template, request
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import logging
from openai import APIError 
import re 
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash # werkzeug is where the hashing lives

# Configure logging
logging.basicConfig(level=logging.INFO)
load_dotenv() 

app = Flask(__name__)
# Secret key is needed for session management/security
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "generate_a_strong_key_in_render")

# --- AUTHENTICATION SETUP ---
auth = HTTPBasicAuth()

# CRITICAL FIX: Hash the password using a guaranteed available method (PBKDF2)
# This solves the AttributeError: module 'hashlib' has no attribute 'scrypt'
USER_DATA = {
    "username": os.environ.get("SCANNER_USERNAME", "admin"),
    "password_hash": generate_password_hash(
        os.environ.get("SCANNER_PASSWORD", "default_password"),
        method="pbkdf2:sha256:260000" # Force PBKDF2:SHA256 hash method
    ) 
}
# --- END AUTH SETUP ---


# --- AUTH DECORATOR ---
@auth.verify_password
def verify_password(username, password):
    """Checks the submitted username/password against the stored hash."""
    if username == USER_DATA["username"]:
        # Use check_password_hash for secure password comparison
        if check_password_hash(USER_DATA["password_hash"], password):
            return username
    return None


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

# The Professional-Grade Prompt (Reduced to essential content)
SYSTEM_PROMPT = """
You are a highly reliable Swing Trader Analyst. Your analysis must work 24/7, based on end-of-day or recent (24-hour) catalyst data. Your goal is to identify stocks primed for a multi-day momentum swing.

Execute a market scan based on the following rules:
1.  **Segregation:** Group results into 'SmallCap', 'MidCap', and 'LargeCap'.
2.  **Breakout Confirmation (Technical):** Identify stocks showing confirmed Volatility Contraction (ATR near 10-day low) and high relative volume accumulation over the last two days.
3.  **Catalyst Check:** Prioritize stocks with a recent SEC Filing, major News, or high social interest over the last 48 hours.
4.  **Actionable Output:** For each candidate, you MUST provide the following keys: Ticker, EntryPrice, TargetPrice, StopPrice, PotentialGainPercent, PrimaryCatalyst, and DetailedAnalysis (3-point bulleted list).

**STRICT RULE 1 (Format):** You MUST return the result ONLY as a single, valid JSON object, with the top-level keys being 'SmallCap', 'MidCap', and 'LargeCap', each containing an array of alert objects.
**STRICT RULE 2 (Off-Hours/No Trades):** If NO high-probability candidates are found, you MUST return this empty JSON structure: {"SmallCap": [], "MidCap": [], "LargeCap": []}. DO NOT RETURN ANY OTHER TEXT OR EXPLANATION.
"""

# --- Regex Helper Function (Stays the same) ---
# app.py (Updated extract_json_from_text function)
def extract_json_from_text(text):
    """
    Strips AI's verbose output (like <think> tags or markdown fences) 
    and aggressively extracts the first valid JSON object.
    """
    # 1. Remove the entire <think> block and all markdown fences (```json)
    # This handles the specific failure seen in the log.
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'```json\s*', '', text, flags=re.DOTALL)
    text = text.replace('```', '')

    # 2. Aggressively find the first opening brace and the last closing brace
    match = re.search(r'\{.*\}', text.strip(), re.DOTALL)
    
    if match:
        return match.group(0)
    
    # If no JSON object is found, return the original text for logging the error context
    return text


# --- PROTECTED ROUTES ---
@app.route('/api/scan', methods=['POST'])
@auth.login_required # <--- PROTECT THE API ENDPOINT
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
                {"role": "user", "content": "BEGIN SWING TRADE SCAN: Find the top 3 candidates now. Present all price points formatted to two decimal places."}
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
        logging.error(f"Internal Scanner Error: {type(e).__name__} - {str(e)}")
        logging.error(f"RAW AI OUTPUT RECEIVED (Critical Debug Data): ---{repr(raw_json_string)}---")
        return jsonify({
            "success": False, 
            "error": "Scanner failed. AI returned non-JSON text. Check terminal for debug output.",
            "data": {"SmallCap": [], "MidCap": [], "LargeCap": []} 
        }), 500


@app.route('/')
@auth.login_required # <--- PROTECT THE MAIN DASHBOARD
def dashboard_view():
    return render_template('index.html')


@app.errorhandler(401)
def unauthorized(error):
    # This ensures the authentication prompt appears for the user
    return 'Login Required: Please enter your credentials to access the Black Box Scanner.', 401


if __name__ == '__main__':
    # NOTE: Run this command in your terminal for local testing:
    # export SCANNER_USERNAME="admin" && export SCANNER_PASSWORD="mypass" && flask run
    # For production, set the variables on Render.
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))