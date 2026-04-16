from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from dotenv import load_dotenv
import cohere
import os
import time

load_dotenv()

client = QdrantClient(host="localhost", port=6333)
dense_model = TextEmbedding("BAAI/bge-large-en-v1.5")
sparse_model = SparseTextEmbedding("Qdrant/bm25")
co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))

def retrieve(query: str) -> list[dict]:
    t0 = time.time()
    query_vector = list(dense_model.embed([query]))[0].tolist()
    dense_results = client.query_points(
        collection_name="arxiv_papers",
        query=query_vector,
        using="dense",
        limit=50
    ).points
    print(f"[timing] dense search: {time.time()-t0:.2f}s")

    t1 = time.time()
    query_sparse = list(sparse_model.embed([query]))[0]
    sparse_results = client.query_points(
        collection_name="arxiv_papers",
        query=models.SparseVector(
            indices=query_sparse.indices.tolist(),
            values=query_sparse.values.tolist()
        ),
        using="bm25",
        limit=50
    ).points
    print(f"[timing] sparse search: {time.time()-t1:.2f}s")

    t2 = time.time()
    merged_result = rrf(dense_results, sparse_results)
    print(f"[timing] rrf merge: {time.time()-t2:.2f}s")

    t3 = time.time()
    final_rerank_list = rerank(query, merged_result[:50])
    print(f"[timing] rerank: {time.time()-t3:.2f}s")

    print(f"[timing] total retrieve: {time.time()-t0:.2f}s")
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
    results = client.retrieve(collection_name="arxiv_papers", ids=ids, with_payload=True)

    # call Cohere rerank API
    documents = [r.payload["text"] for r in results]
    response = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=documents,
        top_n=5
    )

    # response.results is sorted by relevance, each has index into original documents list
    return [results[r.index].payload for r in response.results]
