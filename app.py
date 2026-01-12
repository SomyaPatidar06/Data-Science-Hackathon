

import pathway as pw
import os
import sys
# Startup Debugging
print("--- [STARTUP] Initializing Application... ---")
from dotenv import load_dotenv
load_dotenv()

# Handle single or multiple keys
raw_keys = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
key = raw_keys.split(",")[0].strip() if raw_keys else None

# --- Logging Setup (EARLY) ---
import logging
LOG_FILE = "app.log"
# Force re-config to override any previous init
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w' # Overwrite each run
)
print(f"Logging configured to {LOG_FILE}...")

if key:
    print(f"--- [STARTUP] API Key(s) Found. Using first key ending in ...{key[-4:]} ---")
    import google.generativeai as genai
    genai.configure(api_key=key)
    try:
        print("--- [DIAGNOSTIC] Checking Available Models ---")
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        if models:
            print(f"--- [DIAGNOSTIC] SUCCESS! Found {len(models)} models: {models} ---")
        else:
            print("--- [DIAGNOSTIC] WARNING: No models found! (Possible Region Block?) ---")
    except Exception as e:
        print(f"--- [DIAGNOSTIC] CHECK FAILED: {e} ---")
else:
    print("--- [STARTUP] CRITICAL: NO API KEY FOUND IN ENVIRONMENT ---")

import logging
from llm_logic import check_consistency_llm

# --- Configuration ---
# NOTE: Case Sensitive Path for Linux!
DATA_DIR = "./DataSet/Dataset"  # Based on git structure
BOOKS_DIR = os.path.join(DATA_DIR, "Books")
INPUT_CSV = os.path.join(DATA_DIR, "train.csv") 
OUTPUT_CSV = "results.csv"
LOG_FILE = "app.log"

# --- Logging Setup ---
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w' # Overwrite each run
)
print(f"Logging to {LOG_FILE}...")

# --- Pre-load Books into Memory ---
logging.info("Loading books into memory...")
books_content = {}

# Debugging: Print current directory and list files
logging.info(f"Current Working Directory: {os.getcwd()}")
if os.path.exists(DATA_DIR):
    logging.info(f"Found Data Directory: {DATA_DIR}")
else:
    logging.warning(f"Data Directory NOT found at {DATA_DIR}")
    # Fallback to try finding it
    if os.path.exists("./dataset/Dataset"):
        DATA_DIR = "./dataset/Dataset"
        BOOKS_DIR = os.path.join(DATA_DIR, "Books")
        INPUT_CSV = os.path.join(DATA_DIR, "train.csv")
        logging.info(f"Fallback: Found Data Directory at {DATA_DIR}")

if os.path.exists(BOOKS_DIR):
    for filename in os.listdir(BOOKS_DIR):
        if filename.endswith(".txt"):
            # Normalize to lowercase for robust matching
            book_name = filename.replace(".txt", "").strip().lower()
            path = os.path.join(BOOKS_DIR, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    books_content[book_name] = f.read()
                logging.info(f"Loaded: {book_name} (from {filename})")
            except Exception as e:
                logging.error(f"Failed to load {filename}: {e}")
else:
    logging.error(f"Books directory not found at {BOOKS_DIR}")

# --- Helper Function (Regular Python) ---
def process_row(book_name, character, backstory):
    """
    Standard Python function to process each row.
    """
    logging.info(f"Processing row for book='{book_name}', char='{character}'")
    
    # Normalize lookup
    lookup_name = str(book_name).strip().lower()
    context = books_content.get(lookup_name, "")
    
    if not context:
        logging.error(f"Book not found '{book_name}' (Looked for: '{lookup_name}')")
        logging.info(f"Available books: {list(books_content.keys())}")
        return (0, f"Book '{book_name}' not found.")

    
    if not backstory:
        logging.error("No backstory")
        return (0, "No backstory provided.")
        
    result = check_consistency_llm(context, character, backstory)
    logging.info(f"Result for '{character}': {result.get('prediction')}")
    return (result.get("prediction", 0), result.get("rationale", ""))

# --- Pathway Pipeline ---
# --- Sequential Processing (Survival Mode) ---
import pandas as pd
import time
import csv

def run_sequential_loop():
    print("--- [SURVIVAL MODE] Starting Sequential Processing ---")
    
    # 1. Load Input Data
    if not os.path.exists(INPUT_CSV):
        logging.error(f"Input file not found: {INPUT_CSV}")
        return
        
    df = pd.read_csv(INPUT_CSV)
    total_rows = len(df)
    print(f"--- Loaded {total_rows} rows from {INPUT_CSV} ---")

    # 2. Check for Resume (Load existing Output)
    processed_ids = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            existing_df = pd.read_csv(OUTPUT_CSV)
            # Ensure we only track valid IDs
            if 'id' in existing_df.columns:
                processed_ids = set(existing_df['id'].astype(str))
            elif 'record_id' in existing_df.columns:
                 processed_ids = set(existing_df['record_id'].astype(str))
            
            print(f"--- Found outputs. Resuming! Skipping {len(processed_ids)} already processed rows. ---")
        except Exception as e:
            print(f"--- Warning: Could not read existing results ({e}). Starting from scratch. ---")

    # 3. Setup CSV Writer (Append Mode)
    write_header = not os.path.exists(OUTPUT_CSV) or os.stat(OUTPUT_CSV).st_size == 0
    
    fieldnames = ['id', 'book_name', 'original_label', 'enc', 'rationale']
    
    with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        # 4. Iterate and Process
        for index, row in df.iterrows():
            row_id = str(row['id'])
            
            if row_id in processed_ids:
                # print(f"Skipping ID {row_id} (Done)")
                continue

            print(f"Processing Row {index + 1}/{total_rows} (ID: {row_id})...")
            
            # --- LOGIC CALL ---
            try:
                # Book Content Lookup
                lookup_name = str(row['book_name']).strip().lower()
                context = books_content.get(lookup_name, "")
                
                prediction = 0
                rationale = "Error/Not Found"

                if not context:
                    logging.error(f"Book not found: {row['book_name']}")
                    rationale = f"Book '{row['book_name']}' not found in library."
                elif not row['content']: # Backstory
                    rationale = "No backstory provided."
                else:
                    # Actual LLM Call
                    result = check_consistency_llm(context, row['char'], row['content'])
                    prediction = result.get("prediction", 0)
                    rationale = result.get("rationale", "")

                # SAVE IMMEDIATELY
                writer.writerow({
                    'id': row_id,
                    'book_name': row['book_name'],
                    'original_label': row['label'],
                    'enc': prediction,
                    'rationale': rationale
                })
                f.flush() # CRITICAL: Ensure it is on disk
                logging.info(f"Row {row_id} Saved. Result: {prediction}")
                
                # --- THROTTLE (Crucial for RPM Limits) ---
                print("Sleeping 12s to respect RPM...")
                time.sleep(12) 

            except Exception as e:
                logging.error(f"CRITICAL ERROR on Row {row_id}: {e}")
                print(f"Skipping Row {row_id} due to error.")

    print("--- Processing Complete! ---")

if __name__ == "__main__":
    run_sequential_loop()
