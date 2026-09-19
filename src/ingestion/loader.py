"""Loads raw text out of the source PDF."""
from pypdf import PdfReader


def load_pdf(path: str) -> str:
    """Extract and concatenate all page text from a PDF file."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)