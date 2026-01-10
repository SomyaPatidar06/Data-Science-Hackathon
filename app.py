
import pathway as pw
import os
from llm_logic import check_consistency_llm

# --- Configuration ---
DATA_DIR = "./dataset/Dataset"
BOOKS_DIR = os.path.join(DATA_DIR, "Books")
INPUT_CSV = os.path.join(DATA_DIR, "train.csv") # Or test.csv for final run
OUTPUT_CSV = "results.csv"

# --- Pre-load Books into Memory ---
# Since we have only a few books and they are text files, we can load them into a global dictionary.
# This avoids complex stream joins for this specific hackathon scale.
print("Loading books into memory...")
books_content = {}
if os.path.exists(BOOKS_DIR):
    for filename in os.listdir(BOOKS_DIR):
        if filename.endswith(".txt"):
            book_name = filename.replace(".txt", "") # Assuming filename matches 'book_name' column
            path = os.path.join(BOOKS_DIR, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    books_content[book_name] = f.read()
                print(f"Loaded: {book_name}")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
else:
    print(f"WARNING: Books directory not found at {BOOKS_DIR}")

# --- Helper Wrapper for Pathway UDF ---
@pw.dof
def process_row(book_name, character, backstory):
    """
    User Defined Function to process each row.
    Looks up book content and calls LLM.
    """
    context = books_content.get(book_name, "")
    if not context:
        return 0, f"Book '{book_name}' not found in loaded data."
    
    if not backstory:
        return 0, "No backstory provided."
        
    result = check_consistency_llm(context, character, backstory)
    return result.get("prediction", 0), result.get("rationale", "")

# --- Pathway Pipeline ---
def run_pipeline():
    # 1. Read the CSV
    # schema: id, book_name, char, caption, content, label
    ds = pw.io.csv.read(
        INPUT_CSV,
        schema=pw.schema_from_csv(INPUT_CSV),
        mode="static" # Process once and exit
    )

    # 2. Apply the LLM processing
    # We use 'content' column for backstory
    processed = ds.select(
        id=ds.id,
        label=ds.label, # Keep original label execution
        book_name=ds.book_name,
        prediction_tuple=process_row(ds.book_name, ds.char, ds.content)
    )
    
    # 3. Flatten the tuple result
    final_table = processed.select(
        id=processed.id,
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
