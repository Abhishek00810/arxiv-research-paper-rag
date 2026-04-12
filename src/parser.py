import fitz
from pathlib import Path
import re

def parse_pdf(pdf_path: str) -> list[dict] :
    doc = fitz.open(pdf_path)
    paper_id = Path(pdf_path).stem


    title = _extract_title(doc[0])

    pages = []
    current_section = "unknown"
    
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block["type"] != 0: # 0 = text, 1 = image
                continue
        
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    size = span["size"]
                    bold = bool(span["flags"] & 16)

                    # Heuristic: section heading = bold OR larger font, short, looks numbered
                    if _is_section_heading(text, size, bold):
                        current_section = text
        
        pages.append({
              "paper_id": paper_id,
              "title": title,
              "section": current_section,
              "page": page_num,
              "text": page.get_text(),
        })
    return pages

def _extract_title(page) -> str:
    spans = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                spans.append((span["size"], span["text"].strip()))

    
    if not spans:
        return "unknown"
    
    max_size = max(s[0] for s in spans)
    title_parts = [text for size, text in spans if size >= 0.95 * max_size and text]

    return " ".join(title_parts)


def _is_section_heading(text: str, size: float, bold: bool) -> bool:
    if len(text) > 80:
        return False
    if len(text) < 2:
        return False

    # Numbered sections: "1 Introduction", "2.1 Method"                                                                                                    
    numbered = bool(re.match(r'^\d+[\.\d]*\s+\w', text))                                                            
    return numbered or (bold and size > 10)    
