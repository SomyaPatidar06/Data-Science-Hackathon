

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
def run_pipeline():
    # 1. Read the CSV
    ds = pw.io.csv.read(
        INPUT_CSV,
        schema=pw.schema_from_csv(INPUT_CSV),
        mode="static"
    )

    # 2. Apply the LLM processing using pw.apply
    # Renaming 'id' to 'record_id' because 'id' is reserved in Pathway
    processed = ds.select(
        record_id=ds.id, 
        label=ds.label,
        book_name=ds.book_name,
        prediction_tuple=pw.apply(process_row, ds.book_name, ds.char, ds.content)
    )
    
    # 3. Flatten the tuple result
    final_table = processed.select(
        record_id=processed.record_id,
        book_name=processed.book_name,
        original_label=processed.label,
        enc=processed.prediction_tuple[0],
        rationale=processed.prediction_tuple[1]
    )

    # 4. Output to CSV
    pw.io.csv.write(final_table, OUTPUT_CSV)

    # Run!
    pw.run()

if __name__ == "__main__":
    run_pipeline()
