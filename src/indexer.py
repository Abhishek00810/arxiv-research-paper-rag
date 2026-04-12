from qdrant_client import QdrantClient
from fastembed import TextEmbedding
import hashlib
from qdrant_client.models import VectorParams, Distance, PointStruct

embeddings = TextEmbedding("BAAI/bge-m3")
    
client = QdrantClient(
    host = "localhost",
    port = 6333
)

client.recreate_collection(
    collection_name = "arxiv_papers",
    vectors_config = VectorParams(size = 1024, distance = Distance.COSINE)
)

def index_chunks(chunks: list[dict]) -> None:
    texts = [chunk["text"] for chunk in chunks]
    vectors = list(embeddings.embed(texts))
    points = []
    for i, chunk in enumerate(chunks):
        key = f"{chunk['paper_id']}_{chunk['page']}_{chunk['chunk_index']}"
        point_id = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
        curr_point = PointStruct(   
          id=point_id,                                                                                                         
          vector=vectors[i].tolist(),
          payload=chunk 
        )
        points.append(curr_point)

    client.upsert(collection_name="arxiv_papers", points = points)