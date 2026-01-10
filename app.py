
import pathway as pw
import os
from llm_logic import check_consistency_llm

# --- Configuration ---
# NOTE: Case Sensitive Path for Linux!
DATA_DIR = "./DataSet/Dataset"  # Based on git structure
BOOKS_DIR = os.path.join(DATA_DIR, "Books")
INPUT_CSV = os.path.join(DATA_DIR, "train.csv") 
OUTPUT_CSV = "results.csv"

# --- Pre-load Books into Memory ---
print("Loading books into memory...")
books_content = {}

# Debugging: Print current directory and list files
print(f"Current Working Directory: {os.getcwd()}")
if os.path.exists(DATA_DIR):
    print(f"Found Data Directory: {DATA_DIR}")
else:
    print(f"WARNING: Data Directory NOT found at {DATA_DIR}")
    # Fallback to try finding it
    if os.path.exists("./dataset/Dataset"):
        DATA_DIR = "./dataset/Dataset"
        BOOKS_DIR = os.path.join(DATA_DIR, "Books")
        INPUT_CSV = os.path.join(DATA_DIR, "train.csv")
        print(f"Fallback: Found Data Directory at {DATA_DIR}")

if os.path.exists(BOOKS_DIR):
    for filename in os.listdir(BOOKS_DIR):
        if filename.endswith(".txt"):
            book_name = filename.replace(".txt", "")
            path = os.path.join(BOOKS_DIR, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    books_content[book_name] = f.read()
                print(f"Loaded: {book_name}")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
else:
    print(f"WARNING: Books directory not found at {BOOKS_DIR}")

# --- Helper Function (Regular Python) ---
def process_row(book_name, character, backstory):
    """
    Standard Python function to process each row.
    """
    context = books_content.get(book_name, "")
    if not context:
        return (0, f"Book '{book_name}' not found in loaded data ({list(books_content.keys())})")
    
    if not backstory:
        return (0, "No backstory provided.")
        
    result = check_consistency_llm(context, character, backstory)
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
