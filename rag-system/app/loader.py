import re
from pypdf import PdfReader

def extract_and_clean_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        print(f"erro loading PDF: {e}")
        return "", 0

    raw_text_pieces = []

    for page_number, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            raw_text_pieces.append(page_text)
    
    raw_text = "\n".join(raw_text_pieces)
    cleaned_text = re.sub(r'[ \t]+', ' ', raw_text)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    cleaned_text = "\n".join([line.strip() for line in cleaned_text.splitlines()])

    total_length = len(cleaned_text)
    first_500_chars = cleaned_text[:500]
    
    return cleaned_text, total_length

# if __name__ == "__main__":
#     pdf_path = r"C:/Users/ajmel/desktop/internship-projects/rag-system/data/ArtificiaL_.pdf"
#     print(f"Reading and cleaning: {pdf_path}...\n")
#     preview, length = extract_and_clean_pdf(pdf_path)
    
#     # PRINT REQUIRED METRICS
#     print(f"=========================================")
#     print(f"TOTAL TEXT LENGTH: {length} characters")
#     print(f"=========================================\n")
#     print("FIRST 500 CHARACTERS PREVIEW:")
#     print("-" * 40)
#     print(preview)
#     print("-" * 40)


