
import os
import time
import json
import logging
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    logging.error("CRITICAL: No API Key found! Please set GEMINI_API_KEY or GOOGLE_API_KEY.")
else:
    # distinct last 4 chars for verification
    logging.info(f"Configuring Gemini with API Key ending in '...{api_key[-4:]}'")
    genai.configure(api_key=api_key)

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
            logging.warning(f"Failed to load {model_name}: {e}")
            
    # If all fail:
    logging.error("CRITICAL: All model candidates failed.")
    raise RuntimeError("No available Gemini models found. Check API Key and Region.")

def check_consistency_llm(book_text_snippet, character, backstory):
    """
    Checks if the backstory is consistent with the book context.
    
    Args:
        book_text_snippet: The content of the novel (or relevant chunks).
        character: Character name.
        backstory: The hypothetical backstory to check.
        
    Returns:
        dict: {"prediction": 0 or 1, "rationale": "reasoning..."}
    """
    
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
    
    retries = 3
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
            logging.error(f"Error calling Gemini (Attempt {attempt+1}/{retries}): {e}")
            
            # DEBUG: List available models to find the right name
            try:
                logging.info("--- Available Models ---")
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        logging.info(f"Model: {m.name}")
                logging.info("------------------------")
            except Exception as list_e:
                logging.error(f"Could not list models: {list_e}")

            time.sleep(2 * (attempt + 1))
            
    # Fallback on failure
    return {"prediction": 0, "rationale": "API Error or Timeout"}

