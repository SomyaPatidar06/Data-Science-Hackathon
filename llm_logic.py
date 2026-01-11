
import os
import time
import json
import logging
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- API Key Management (Rotation) ---
# Load ALL keys (split by comma) to create a pool
raw_keys = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
API_KEY_POOL = [k.strip() for k in raw_keys.split(",") if k.strip()]
current_key_index = 0

if not API_KEY_POOL:
    logging.error("No API Keys found in environment!")
else:
    logging.info(f"Loaded {len(API_KEY_POOL)} API Keys for rotation.")

def get_current_key():
    global current_key_index
    if not API_KEY_POOL: return None
    return API_KEY_POOL[current_key_index % len(API_KEY_POOL)]

def rotate_key():
    global current_key_index, _ACTIVE_MODEL
    current_key_index += 1
    new_key = get_current_key()
    logging.info(f"🔄 Rotating to API Key #{current_key_index % len(API_KEY_POOL) + 1} (Ends in ...{new_key[-4:]})")
    genai.configure(api_key=new_key)
    # CRITICAL: Force model reload with new credentials
    _ACTIVE_MODEL = None
    return new_key

# Configure initial key
if API_KEY_POOL:
    genai.configure(api_key=get_current_key())

# Generation Config
generation_config = {
  "temperature": 0.2,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 1024,
  "response_mime_type": "application/json",
}

# Auto-Discovery of Working Model
_ACTIVE_MODEL = None

def get_working_model():
    global _ACTIVE_MODEL
    if _ACTIVE_MODEL:
        return _ACTIVE_MODEL

    candidates = [
        "gemini-2.0-flash-exp",
        "gemini-exp-1206",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-pro",
        "gemini-1.0-pro"
    ]
    
    logging.info("Starting Auto-Discovery for Working Gemini Model...")
    
    for model_name in candidates:
        try:
            logging.info(f"Testing model: {model_name}...")
            test_model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config
            )
            # Dry run to check if it exists/is accessible
            test_model.generate_content("Hello") 
            logging.info(f"SUCCESS: Found working model: {model_name}")
            _ACTIVE_MODEL = test_model
            return _ACTIVE_MODEL
        except Exception as e:
            # If we get a Quota error (429), the model EXISTS! We just hit a limit.
            # So we should use it, but wait a bit.
            if "429" in str(e) or "Quota" in str(e) or "quota" in str(e):
                logging.info(f"SUCCESS (Quota Hit): Found working model: {model_name}. Pausing 60s...")
                time.sleep(60)
                _ACTIVE_MODEL = test_model
                return _ACTIVE_MODEL
            
            logging.warning(f"Failed to load {model_name}: {e}")
            
    # If all fail:
    logging.error("CRITICAL: All model candidates failed.")
    raise RuntimeError("No available Gemini models found. Check API Key and Region.")

# Rate Limiter Global
LAST_CALL_TIME = 0

def check_consistency_llm(book_text_snippet, character, backstory):
    """
    Checks if the backstory is consistent with the book context.
    """
    global LAST_CALL_TIME
    
    # 1. Enforce Rate Limit (Safe for Free Tier)
    # Even paid tiers benefit from slight spacing to avoid burst limits
    time_since_last = time.time() - LAST_CALL_TIME
    if time_since_last < 4: 
        # Paid tier: 0s if we wanted, but let's do 4s to be safe due to current errors
        # If user is confirmed Free, we ideally need 10s.
        # Let's use 10s to be absolutely sure it finishes.
        time.sleep(10 - time_since_last)
    
    LAST_CALL_TIME = time.time()
    
    # Construct prompt
    prompt = f"""
    You are an expert literary analyst. Your task is to determine if a hypothetical backstory for a character is CONSISTENT with the provided novel text.
    
    Character: {character}
    
    Hypothetical Backstory:
    {backstory}
    
    Novel Text (Context):
    {book_text_snippet[:1000000]} 
    
    (Note: Context truncated to 1M chars if too long, but usually fits in Gemini 1.5 Pro)

    Task:
    1. Analyze the novel text to understand the character's established history, personality, and known facts.
    2. Compare the hypothetical backstory with these established facts.
    3. If the backstory explicitly CONTRADICTS the novel, output 0 (Inconsistent).
    4. If the backstory is PLAUSIBLE and does not contradict the novel (even if not explicitly mentioned), output 1 (Consistent).
    
    Output Format: JSON
    {{
        "prediction": 0 or 1,
        "rationale": "A concise explanation of why it is consistent or inconsistent."
    }}
    """
    
    # Track rotations for this specific call
    rotations_this_call = 0
    
    retries = 10 # Increase retries to allow full rotation
    for attempt in range(retries):
        try:
            # Retrieve model dynamically (handles caching internally)
            model_instance = get_working_model()
            response = model_instance.generate_content(prompt)
            # Cleanup: sometimes API returns ```json ... ``` despite instructions
            raw_text = response.text
            clean_text = raw_text.strip()
            if clean_text.startswith("```"):
                import re
                match = re.search(r"```(?:json)?\s*(.*)\s*```", clean_text, re.DOTALL)
                if match:
                    clean_text = match.group(1)
            
            return json.loads(clean_text)
        except Exception as e:
            logging.error(f"Error calling Gemini (Key ...{get_current_key()[-4:]}): {e}")
            
            # If Quota Exceeded (429), ROTATE KEY and RETRY immediately!
            if "429" in str(e) or "Quota" in str(e) or "quota" in str(e):
                logging.warning(f"⚠️ Quota Hit at index {current_key_index}!")
                
                # Check for "Total Exhaustion" (We rotated through ALL keys and they are ALL dead)
                rotations_this_call += 1
                
                if len(API_KEY_POOL) > 1:
                    # If we tried every key in the pool, we MUST sleep
                    if rotations_this_call >= len(API_KEY_POOL):
                        logging.warning("🛑 ALL KEYS EXHAUSTED! Pool needs recharge. Sleeping 60s...")
                        time.sleep(60)
                        rotations_this_call = 0 # Reset counter after sleep
                    else:
                        rotate_key()
                        time.sleep(1) 
                else:
                    # If we only have 1 key, we MUST sleep
                    logging.warning("🚫 No other keys to switch to. Sleeping 65s...")
                    time.sleep(65)
            else:
                # DEBUG: List available models to find the right name
                try:
                    logging.info("--- Available Models ---")
                    for m in genai.list_models():
                        if 'generateContent' in m.supported_generation_methods:
                            logging.info(f"Model: {m.name}")
                    logging.info("------------------------")
                except Exception as list_e:
                    logging.error(f"Could not list models: {list_e}")

                time.sleep(5 * (attempt + 1))
            
    # Fallback on failure
    return {"prediction": 0, "rationale": "API Error or Timeout"}

