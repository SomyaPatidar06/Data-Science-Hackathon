
from pypdf import PdfReader

try:
    reader = PdfReader("695bac0f217f3_Problem_Statement_-_Kharagpur_Data_Science_Hackathon_2026 (1).pdf")
    
    print("--- Reading Pages 1-7 ---")
    for i in range(min(7, len(reader.pages))):
        text = page = reader.pages[i].extract_text()
        print(f"--- Page {i+1} ---")
        print(text)
        
    # Also look for annotations (links)
    print("\n--- Identifying Links ---")
    for page in reader.pages:
        if "/Annots" in page:
            for annot in page["/Annots"]:
                subtype = annot.get_object().get("/Subtype")
                if subtype == "/Link":
                    uri = annot.get_object().get("/A").get("/URI")
                    if uri:
                        print(f"Found Link: {uri}")

except Exception as e:
    print(f"Error reading PDF: {e}")
