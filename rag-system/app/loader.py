import re
import os
from pypdf import PdfReader

def extract_clean_text_pdf(pdf_path):

    try:
        loader = PdfReader(pdf_path)
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return "", 0
    
    full_text = []
    for page_num, page in enumerate(loader.pages):
        page_text = page.extract_text()
        if page_text:
            full_text.append(page_text)
    
    raw_text = "\n".join(full_text)
    cleaned_text = re.sub(r'[\t]+', ' ', raw_text)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    cleaned_text = "\n".join([line.strip() for line in cleaned_text.splitlines()])

    total_length = len(cleaned_text)

    return cleaned_text, total_length