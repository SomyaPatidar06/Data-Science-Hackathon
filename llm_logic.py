
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

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
)

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
            response = model.generate_content(prompt)
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
            time.sleep(2 * (attempt + 1))
            
    # Fallback on failure
    return {"prediction": 0, "rationale": "API Error or Timeout"}

