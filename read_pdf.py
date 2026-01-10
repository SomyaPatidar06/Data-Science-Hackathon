
from pypdf import PdfReader
import sys

try:
    reader = PdfReader("695bac0f217f3_Problem_Statement_-_Kharagpur_Data_Science_Hackathon_2026 (1).pdf")
    number_of_pages = len(reader.pages)
    print(f"Total Pages: {number_of_pages}")
    
    full_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        print(f"--- Page {i+1} ---")
        print(text)
        full_text += text + "\n"
        
except Exception as e:
    print(f"Error reading PDF: {e}")
