from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding  
from sentence_transformers import CrossEncoder  


client = QdrantClient(host="localhost", port=6333)
dense_model = TextEmbedding("BAAI/bge-large-en-v1.5")
sparse_model = SparseTextEmbedding("Qdrant/bm25")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")  

def retrieve(query: str) -> list[dict]:
    #this is function all the other, two returns top 5 chunks
    #dense results
    query_vector = list(dense_model.embed([query]))[0].tolist()
    dense_results = client.search(
        collection_name = "arxiv_papers",
        query_vector=("dense", query_vector),
        limit = 50
    )

    # sparse search
    query_sparse = list(sparse_model.embed([query]))[0]
    sparse_results = client.search(                                                                                                           
      collection_name="arxiv_papers",
      query_vector=("bm25", models.SparseVector(                                                                                            
          indices=query_sparse.indices.tolist(),                                                                                            
          values=query_sparse.values.tolist()
      )),                                                                                                                                   
      limit=50    
    )    

    merged_result = rrf(dense_results, sparse_results)
    
    final_rerank_list = rerank(query, merged_result[:50])
    return final_rerank_list


def rrf(dense_results, sparse_results, k = 60) -> list:
    scores = {}

    for rank, result in enumerate(dense_results):
        id = result.id
        scores[id] = scores.get(id, 0) + 1/(rank + k)

    for rank, result in enumerate(sparse_results):
        id = result.id
        scores[id] = scores.get(id, 0) + 1/(rank + k)

    return sorted(scores.items(), key = lambda x: x[1], reverse=True)

def rerank(query: str, chunks: list) -> list:
    # chunks is list of (id, rrf_score) tuples from rrf()                                                                                      
    # fetch actual chunk payloads from Qdrant by id 
    ids = [chunk[0] for chunk in chunks]
    results = client.retrieve(collection_name = "arxiv_papers", ids = ids, with_payload = True)

    #build pairs of [query, chunk_text] for cross encoder
    pairs = [[query, r.payload["text"]] for r in results]

    #score each pair
    scores = reranker.predict(pairs)

    #sort by sscore, return top 5 payloads
    ranked = sorted(zip(scores, results), key = lambda x: x[0], reverse=True)

    return [r.payload for score, r in ranked[:5]]
