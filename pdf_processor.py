from pypdf import PdfReader

def extract_text_from_pdf(pdf_file):
    """
    Extracts all raw text from an uploaded PDF file.
    Works with file paths or Streamlit UploadedFile objects.
    """
    text = ""
    try:
        # Initialize the PDF reader
        reader = PdfReader(pdf_file)
        
        # Loop through every page and extract text
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
                
    except Exception as e:
        print(f"Error reading PDF: {e}")
        
    return text.strip()
