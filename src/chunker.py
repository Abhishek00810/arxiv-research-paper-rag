from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import FastEmbedEmbeddings                                                               
                                                                                                                               
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-large-en-v1.5")

def chunk_pages(pages: list[dict]) -> list[dict]:
    chunker = SemanticChunker(embeddings)

    all_chunks = []
    chunk_index = 0

    for page in pages:
        splits = chunker.split_text(page["text"])
        for chunk_text in splits:
            all_chunks.append({                                                                                                      
                "text": chunk_text,
                "paper_id": page["paper_id"],                                                                                        
                "title": page["title"],
                "section": page["section"],
                "page": page["page"],
                "chunk_index": chunk_index,                                                                                          
            })
            chunk_index += 1                                                                                                         
                  

    return all_chunks