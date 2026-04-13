from fitz import f
from parser import parse_pdf
from chunker import chunk_pages
from indexer import index_chunks
from pathlib import Path


def ingest_all(papers_dir: str = "data/papers"):
    pdf_files = list(Path(papers_dir).glob("*.pdf"))

    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}...")                                                                                                                       
        pages = parse_pdf(str(pdf_path))                                                                                                                              
        chunks = chunk_pages(pages)
        index_chunks(chunks)                                                                                                                                          
        print(f"Done — {len(chunks)} chunks indexed")


if __name__ == "__main__":
    ingest_all()

    
          