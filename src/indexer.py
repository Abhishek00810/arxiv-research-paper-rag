from qdrant_client import QdrantClient
from fastembed import TextEmbedding, SparseTextEmbedding                                                                     
import hashlib
from qdrant_client.models import VectorParams, Distance, PointStruct, SparseVectorParams, SparseVector


embeddings = TextEmbedding("BAAI/bge-large-en-v1.5")
sparse_model = SparseTextEmbedding("Qdrant/bm25")  
client = QdrantClient(
    host = "localhost",
    port = 6333
)

# create collection only if it doesn't exist
if not client.collection_exists("arxiv_papers"):                                                                                                                      
      client.create_collection(                                                                                                                                         
          collection_name="arxiv_papers",                                                                                                                               
          vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},                                                                                  
          sparse_vectors_config={"bm25": SparseVectorParams()}                                                                                                        
      )

def index_chunks(chunks: list[dict]) -> None:
    texts = [chunk["text"] for chunk in chunks]
    vectors = list(embeddings.embed(texts))
    sparse_vectors = list(sparse_model.embed(texts)) 
    points = []
    for i, chunk in enumerate(chunks):
        key = f"{chunk['paper_id']}_{chunk['page']}_{chunk['chunk_index']}"
        point_id = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
        curr_point = PointStruct(   
          id=point_id,                                                                                                         
          vector={
           "dense": vectors[i].tolist(),
            "bm25": SparseVector(                                                                                                
              indices=sparse_vectors[i].indices.tolist(),
              values=sparse_vectors[i].values.tolist() 
            )
          },
          payload=chunk
        )
        points.append(curr_point)

    client.upsert(collection_name="arxiv_papers", points = points)