import re
from pypdf import PdfReader

def extract_clean_text_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return [], 0
    
    pages_data = []
    total_length = 0
    
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            # Clean basic formatting per page
            cleaned_page = re.sub(r'[\t]+', ' ', page_text)
            cleaned_page = re.sub(r'\n{3,}', '\n\n', cleaned_page)
            cleaned_page = "\n".join([line.strip() for line in cleaned_page.splitlines()])
            
            total_length += len(cleaned_page)
            
            # Keep track of text tied to its exact page number
            pages_data.append({
                "text": cleaned_page,
                "page_number": page_num + 1
            })
            
    return pages_data, total_length