from pypdf import PdfReader

def load_pdf(path:str) -> str:
    """Extract all content and text form pdf"""

    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)