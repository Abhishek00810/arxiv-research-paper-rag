import sys
sys.path.insert(0, 'src')

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import json

load_dotenv()

# no reranker — just hybrid search
client = QdrantClient(host="localhost", port=6333)
dense_model = TextEmbedding("BAAI/bge-large-en-v1.5")
sparse_model = SparseTextEmbedding("Qdrant/bm25")
llm = ChatGroq(model="llama-3.3-70b-versatile")

def fast_retrieve(query: str) -> list[dict]:
    query_vector = list(dense_model.embed([query]))[0].tolist()
    dense_results = client.query_points(
        collection_name="arxiv_papers",
        query=query_vector,
        using="dense",
        limit=5
    ).points
    return [r.payload for r in dense_results]

def fast_answer(question: str) -> str:
    chunks = fast_retrieve(question)
    chunks_text = "\n\n".join([f"[{c['title']} | {c['section']}]\n{c['text']}" for c in chunks])
    response = llm.invoke(f"""
    Answer using ONLY the provided context.
    Question: {question}
    Context: {chunks_text}
    Provide a clear accurate answer.
    """)
    return response.content

questions = [
    "What problem does Flash Attention solve compared to standard attention?",
    "What is retrieval-augmented generation RAG?",
    "How does Flash Attention reduce memory complexity?",
    "What are the tradeoffs between LoRA and full fine-tuning?",
    "How does Firecracker design differ from traditional containers?",
    "What is the role of the KV cache in LLM inference?",
    "How does vLLM use PagedAttention to improve throughput?",
    "memory management Firecracker PagedAttention similar problems",
    "relationship original attention mechanism Flash Attention optimization",
    "LoRA parameter efficiency scaling transformer",
    "Firecracker microVM vs Linux containers isolation performance",
    "RAG complement transformer limitations attention"
]

output = {}
for i, q in enumerate(questions):
    print(f"Processing {i+1}/{len(questions)}: {q[:50]}...")
    answer = fast_answer(q)
    output[q] = answer
    print(f"Done.")

with open("evals/fast_answers.json", "w") as f:
    json.dump(output, f, indent=2)

print("Saved to evals/fast_answers.json")
